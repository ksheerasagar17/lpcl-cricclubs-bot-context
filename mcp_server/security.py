"""
Security validation and enforcement for read-only MongoDB MCP server.

Implements query sanitization, operation validation, and safety checks.
"""

import re
import logging
from typing import Dict, Any, List, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class SecurityValidator:
    """
    Security validator for MongoDB operations in read-only MCP server.
    
    Enforces read-only access, query sanitization, and safety checks.
    """
    
    # Allowed read-only operations
    ALLOWED_OPERATIONS: Set[str] = {
        "find", "aggregate", "count_documents", "distinct", 
        "find_one", "estimated_document_count"
    }
    
    # Forbidden operations that could modify data
    FORBIDDEN_OPERATIONS: Set[str] = {
        "insert", "insert_one", "insert_many",
        "update", "update_one", "update_many", "replace_one",
        "delete", "delete_one", "delete_many", "remove",
        "drop", "create_collection", "create_index",
        "map_reduce", "bulk_write", "find_one_and_delete",
        "find_one_and_replace", "find_one_and_update"
    }
    
    # Dangerous query operators that could cause security issues
    DANGEROUS_OPERATORS: Set[str] = {
        "$where",  # JavaScript execution
        "$regex",  # Could cause ReDoS if not properly limited
        "$eval",   # JavaScript evaluation
        "$function"  # JavaScript functions in aggregation
    }
    
    # Safe aggregation operators for analytics
    SAFE_AGGREGATION_OPERATORS: Set[str] = {
        "$match", "$project", "$group", "$sort", "$limit", "$skip",
        "$unwind", "$lookup", "$addFields", "$count", "$facet",
        "$bucket", "$bucketAuto", "$sortByCount", "$sample",
        "$unionWith", "$set", "$unset", "$replaceRoot",
        "$addToSet", "$push", "$sum", "$avg", "$min", "$max",
        "$first", "$last", "$stdDevPop", "$stdDevSamp"
    }
    
    def __init__(self, max_time_ms: int = 3000, max_documents: int = 1000):
        """
        Initialize security validator.
        
        Args:
            max_time_ms: Maximum query execution time in milliseconds
            max_documents: Maximum number of documents to return
        """
        self.max_time_ms = max_time_ms
        self.max_documents = max_documents
        
        logger.info(f"Security validator initialized with max_time_ms={max_time_ms}, max_documents={max_documents}")
    
    def validate_operation(self, operation: str) -> None:
        """
        Validate that the operation is allowed in read-only mode.
        
        Args:
            operation: MongoDB operation name
            
        Raises:
            SecurityError: If operation is not allowed
        """
        if operation.lower() in self.FORBIDDEN_OPERATIONS:
            raise SecurityError(f"Operation '{operation}' is forbidden in read-only mode")
        
        if operation.lower() not in self.ALLOWED_OPERATIONS:
            logger.warning(f"Unknown operation '{operation}' - allowing with caution")
    
    def sanitize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize MongoDB query to prevent security issues.
        
        Args:
            query: Original MongoDB query
            
        Returns:
            Sanitized query safe for execution
            
        Raises:
            SecurityError: If query contains dangerous operations
        """
        if not isinstance(query, dict):
            raise SecurityError("Query must be a dictionary")
        
        sanitized = self._sanitize_dict(query)
        
        # Add safety limits
        sanitized = self._add_safety_limits(sanitized)
        
        logger.debug(f"Query sanitized: {sanitized}")
        return sanitized
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary data.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            # Check for dangerous operators
            if key in self.DANGEROUS_OPERATORS:
                raise SecurityError(f"Dangerous operator '{key}' is not allowed")
            
            # Sanitize key
            clean_key = self._sanitize_key(key)
            
            # Recursively sanitize value
            if isinstance(value, dict):
                sanitized[clean_key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[clean_key] = self._sanitize_list(value)
            else:
                sanitized[clean_key] = self._sanitize_value(value)
        
        return sanitized
    
    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """
        Sanitize list data.
        
        Args:
            data: List to sanitize
            
        Returns:
            Sanitized list
        """
        sanitized = []
        
        for item in data:
            if isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(self._sanitize_list(item))
            else:
                sanitized.append(self._sanitize_value(item))
        
        return sanitized
    
    def _sanitize_key(self, key: str) -> str:
        """
        Sanitize dictionary key.
        
        Args:
            key: Key to sanitize
            
        Returns:
            Sanitized key
        """
        if not isinstance(key, str):
            raise SecurityError("Dictionary keys must be strings")
        
        # Check for injection patterns
        if re.search(r'[{}();]', key):
            raise SecurityError(f"Key '{key}' contains potentially dangerous characters")
        
        return key
    
    def _sanitize_value(self, value: Any) -> Any:
        """
        Sanitize individual value.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized value
        """
        # Check for dangerous string patterns
        if isinstance(value, str):
            # Check for JavaScript injection patterns
            if re.search(r'(function\s*\(|eval\s*\(|setTimeout|setInterval)', value, re.IGNORECASE):
                raise SecurityError(f"Value contains potentially dangerous JavaScript patterns")
            
            # Limit string length to prevent DoS
            if len(value) > 10000:
                raise SecurityError("String value too long (max 10000 characters)")
        
        return value
    
    def _add_safety_limits(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add safety limits to query.
        
        Args:
            query: Query to add limits to
            
        Returns:
            Query with safety limits
        """
        # For find operations, we'll add limits in the MCP server
        # This method is for future extension
        return query
    
    def validate_aggregation_pipeline(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and sanitize aggregation pipeline.
        
        Args:
            pipeline: Aggregation pipeline to validate
            
        Returns:
            Sanitized pipeline
            
        Raises:
            SecurityError: If pipeline contains dangerous operations
        """
        if not isinstance(pipeline, list):
            raise SecurityError("Aggregation pipeline must be a list")
        
        if len(pipeline) > 50:  # Reasonable limit for pipeline complexity
            raise SecurityError("Aggregation pipeline too complex (max 50 stages)")
        
        sanitized_pipeline = []
        
        for stage in pipeline:
            if not isinstance(stage, dict):
                raise SecurityError("Each pipeline stage must be a dictionary")
            
            if len(stage) != 1:
                raise SecurityError("Each pipeline stage must have exactly one operator")
            
            operator = list(stage.keys())[0]
            
            # Check if operator is safe
            if operator not in self.SAFE_AGGREGATION_OPERATORS:
                raise SecurityError(f"Aggregation operator '{operator}' is not allowed")
            
            # Sanitize stage
            sanitized_stage = self._sanitize_dict(stage)
            
            # Add safety limits for specific operators
            if operator == "$limit":
                # Ensure limit is not too high
                limit_value = sanitized_stage[operator]
                if isinstance(limit_value, int) and limit_value > self.max_documents:
                    sanitized_stage[operator] = self.max_documents
            
            sanitized_pipeline.append(sanitized_stage)
        
        # Ensure pipeline has reasonable limits
        sanitized_pipeline = self._ensure_pipeline_limits(sanitized_pipeline)
        
        return sanitized_pipeline
    
    def _ensure_pipeline_limits(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure aggregation pipeline has appropriate limits.
        
        Args:
            pipeline: Pipeline to check
            
        Returns:
            Pipeline with enforced limits
        """
        # Check if pipeline has a $limit stage
        has_limit = any("$limit" in stage for stage in pipeline)
        
        # If no limit, add one for safety
        if not has_limit:
            pipeline.append({"$limit": self.max_documents})
        
        return pipeline
    
    def validate_collection_name(self, collection_name: str) -> None:
        """
        Validate collection name for security.
        
        Args:
            collection_name: Name of collection to validate
            
        Raises:
            SecurityError: If collection name is invalid
        """
        if not isinstance(collection_name, str):
            raise SecurityError("Collection name must be a string")
        
        if not collection_name:
            raise SecurityError("Collection name cannot be empty")
        
        # Check for dangerous characters
        if re.search(r'[^a-zA-Z0-9_.-]', collection_name):
            raise SecurityError("Collection name contains invalid characters")
        
        # Check length
        if len(collection_name) > 127:
            raise SecurityError("Collection name too long (max 127 characters)")
        
        # Prevent system collection access
        if collection_name.startswith("system."):
            raise SecurityError("Access to system collections is forbidden")
    
    def create_execution_context(self) -> Dict[str, Any]:
        """
        Create execution context with security parameters.
        
        Returns:
            Dictionary with execution parameters
        """
        return {
            "maxTimeMS": self.max_time_ms,
            "allowDiskUse": False,  # CRITICAL: Never allow disk usage
            "maxDocuments": self.max_documents,
            "readConcern": {"level": "available"},  # Fast reads
            "hint": None  # No index hints to prevent abuse
        }


class SecurityError(Exception):
    """Raised when security validation fails."""
    
    def __init__(self, message: str):
        super().__init__(message)
        self.timestamp = datetime.utcnow()
        logger.error(f"Security violation: {message}")


def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Log security events for monitoring and alerting.
    
    Args:
        event_type: Type of security event
        details: Event details
    """
    logger.warning(f"Security event [{event_type}]: {details}")
    
    # In production, this could send to monitoring systems
    # like DataDog, Prometheus, or custom alerting systems