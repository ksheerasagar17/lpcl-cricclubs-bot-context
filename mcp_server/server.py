"""
Read-only MongoDB MCP Server for Cricket-Insight Agent.

Provides secure, read-only access to cricket databases through MCP protocol.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .connection import MongoConnection, SecurityError as ConnectionSecurityError
from .security import SecurityValidator, SecurityError, log_security_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadOnlyMongoMCP:
    """
    Read-only MongoDB MCP server with comprehensive security enforcement.
    
    Implements the Model Context Protocol (MCP) for secure cricket data access.
    """
    
    def __init__(self, connection_uri: Optional[str] = None):
        """
        Initialize read-only MongoDB MCP server.
        
        Args:
            connection_uri: MongoDB connection string
        """
        self.connection = MongoConnection(connection_uri)
        self.security_validator = SecurityValidator(
            max_time_ms=self.connection.max_time_ms,
            max_documents=self.connection.max_documents
        )
        
        # Server state
        self.is_connected = False
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        
        # Supported operations
        self.supported_operations = {
            "find", "aggregate", "count_documents", "distinct", "find_one"
        }
        
        logger.info("ReadOnlyMongoMCP server initialized")
    
    async def start(self) -> None:
        """
        Start the MCP server and establish database connection.
        
        Raises:
            ConnectionError: If unable to connect to MongoDB
        """
        try:
            self.connection.connect()
            self.is_connected = True
            logger.info("MCP server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise ConnectionError(f"Unable to start MCP server: {e}")
    
    async def stop(self) -> None:
        """Stop the MCP server and cleanup resources."""
        try:
            self.connection.disconnect()
            self.is_connected = False
            logger.info("MCP server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
    
    async def execute_query(
        self, 
        operation: str, 
        collection: str, 
        query: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a read-only MongoDB query with full security validation.
        
        Args:
            operation: MongoDB operation to perform
            collection: Collection name
            query: Query parameters
            options: Additional options for the operation
            
        Returns:
            Query results with metadata
            
        Raises:
            SecurityError: If operation violates security policies
            ConnectionError: If database connection fails
            ValueError: If parameters are invalid
        """
        if not self.is_connected:
            raise ConnectionError("MCP server is not connected to database")
        
        start_time = datetime.utcnow()
        self.request_count += 1
        
        try:
            # Security validation
            self.security_validator.validate_operation(operation)
            self.security_validator.validate_collection_name(collection)
            
            # Sanitize query if provided
            if query:
                query = self.security_validator.sanitize_query(query)
            
            # Create execution context with security parameters
            execution_context = self.security_validator.create_execution_context()
            
            # Execute operation
            result = await self._execute_operation(
                operation, collection, query, options, execution_context
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "operation": operation,
                "collection": collection,
                "result": result,
                "metadata": {
                    "execution_time_ms": execution_time,
                    "timestamp": start_time.isoformat(),
                    "read_only": True,
                    "security_validated": True,
                    "document_count": len(result.get("documents", [])) if isinstance(result, dict) else 0
                }
            }
            
        except (SecurityError, ConnectionSecurityError) as e:
            self.error_count += 1
            log_security_event("query_security_violation", {
                "operation": operation,
                "collection": collection,
                "error": str(e),
                "timestamp": start_time.isoformat()
            })
            raise
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Query execution failed: {e}")
            raise ValueError(f"Query execution failed: {e}")
    
    async def _execute_operation(
        self,
        operation: str,
        collection_name: str,
        query: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the actual MongoDB operation.
        
        Args:
            operation: MongoDB operation
            collection_name: Collection name
            query: Query parameters
            options: Operation options
            execution_context: Security execution context
            
        Returns:
            Operation results
        """
        collection = self.connection.get_collection(collection_name)
        
        # Merge options with security context
        exec_options = {**execution_context}
        if options:
            # Only allow safe options
            safe_options = {"limit", "skip", "sort", "projection"}
            for key, value in options.items():
                if key in safe_options:
                    exec_options[key] = value
        
        # Execute based on operation type
        if operation == "find":
            return await self._execute_find(collection, query or {}, exec_options)
        
        elif operation == "find_one":
            return await self._execute_find_one(collection, query or {}, exec_options)
        
        elif operation == "aggregate":
            if not query or "pipeline" not in query:
                raise ValueError("Aggregate operation requires 'pipeline' in query")
            pipeline = self.security_validator.validate_aggregation_pipeline(query["pipeline"])
            return await self._execute_aggregate(collection, pipeline, exec_options)
        
        elif operation == "count_documents":
            return await self._execute_count(collection, query or {}, exec_options)
        
        elif operation == "distinct":
            if not query or "field" not in query:
                raise ValueError("Distinct operation requires 'field' in query")
            return await self._execute_distinct(collection, query["field"], query.get("filter", {}), exec_options)
        
        else:
            raise SecurityError(f"Operation '{operation}' is not supported")
    
    async def _execute_find(self, collection, query: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute find operation with security limits."""
        cursor = collection.find(
            query,
            projection=options.get("projection"),
            limit=min(options.get("limit", self.connection.max_documents), self.connection.max_documents),
            skip=options.get("skip", 0),
            sort=options.get("sort"),
            max_time_ms=options["maxTimeMS"]
        )
        
        documents = list(cursor)
        
        return {
            "operation": "find",
            "documents": self._serialize_documents(documents),
            "count": len(documents),
            "query": query
        }
    
    async def _execute_find_one(self, collection, query: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute find_one operation."""
        document = collection.find_one(
            query,
            projection=options.get("projection"),
            max_time_ms=options["maxTimeMS"]
        )
        
        return {
            "operation": "find_one",
            "document": self._serialize_document(document) if document else None,
            "found": document is not None,
            "query": query
        }
    
    async def _execute_aggregate(self, collection, pipeline: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation pipeline."""
        cursor = collection.aggregate(
            pipeline,
            maxTimeMS=options["maxTimeMS"],
            allowDiskUse=options["allowDiskUse"]  # Always False for security
        )
        
        documents = list(cursor)
        
        return {
            "operation": "aggregate",
            "documents": self._serialize_documents(documents),
            "count": len(documents),
            "pipeline": pipeline
        }
    
    async def _execute_count(self, collection, query: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute count_documents operation."""
        count = collection.count_documents(
            query,
            maxTimeMS=options["maxTimeMS"]
        )
        
        return {
            "operation": "count_documents",
            "count": count,
            "query": query
        }
    
    async def _execute_distinct(self, collection, field: str, filter_query: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute distinct operation."""
        values = collection.distinct(
            field,
            filter_query,
            maxTimeMS=options["maxTimeMS"]
        )
        
        return {
            "operation": "distinct",
            "field": field,
            "values": values[:self.connection.max_documents],  # Limit results
            "count": len(values),
            "filter": filter_query
        }
    
    def _serialize_document(self, document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Serialize a single MongoDB document for JSON response."""
        if not document:
            return None
        
        # Convert ObjectId and other BSON types to JSON-serializable formats
        return self._convert_bson_types(document)
    
    def _serialize_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Serialize list of MongoDB documents for JSON response."""
        return [self._serialize_document(doc) for doc in documents]
    
    def _convert_bson_types(self, obj: Any) -> Any:
        """Convert BSON types to JSON-serializable types."""
        if hasattr(obj, '__dict__'):
            # Handle ObjectId and other BSON types
            if hasattr(obj, 'oid'):  # ObjectId
                return str(obj)
            elif hasattr(obj, 'time'):  # datetime
                return obj.isoformat()
        
        if isinstance(obj, dict):
            return {key: self._convert_bson_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_bson_types(item) for item in obj]
        else:
            return obj
    
    async def get_server_status(self) -> Dict[str, Any]:
        """
        Get comprehensive server status and health information.
        
        Returns:
            Server status dictionary
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get database health
        db_health = self.connection.health_check()
        
        return {
            "server": {
                "status": "running" if self.is_connected else "disconnected",
                "uptime_seconds": uptime,
                "read_only": True,
                "version": "1.0.0"
            },
            "statistics": {
                "requests_processed": self.request_count,
                "errors_count": self.error_count,
                "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1) * 100
            },
            "security": {
                "read_only_enforced": True,
                "max_time_ms": self.connection.max_time_ms,
                "allow_disk_use": False,
                "max_documents": self.connection.max_documents
            },
            "database": db_health,
            "supported_operations": list(self.supported_operations)
        }
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information for available collections.
        
        Returns:
            Schema information dictionary
        """
        try:
            collections = self.connection.database.list_collection_names()
            
            schema_info = {
                "database": self.connection.database_name,
                "collections": {}
            }
            
            for collection_name in collections:
                if not collection_name.startswith("system."):  # Skip system collections
                    collection = self.connection.get_collection(collection_name)
                    
                    # Get collection stats
                    try:
                        stats = self.connection.database.command("collStats", collection_name)
                        schema_info["collections"][collection_name] = {
                            "document_count": stats.get("count", 0),
                            "avg_document_size": stats.get("avgObjSize", 0),
                            "total_size": stats.get("size", 0)
                        }
                    except Exception as e:
                        logger.warning(f"Could not get stats for collection {collection_name}: {e}")
                        schema_info["collections"][collection_name] = {
                            "document_count": "unknown",
                            "avg_document_size": "unknown",
                            "total_size": "unknown"
                        }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'connection') and self.is_connected:
            try:
                self.connection.disconnect()
            except:
                pass