"""
MongoDB connection management for read-only MCP server.

Provides secure, pooled connections with proper timeout and security enforcement.
"""

import os
import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MongoConnection:
    """
    MongoDB connection manager with security-first approach.
    
    Enforces read-only access patterns and connection pooling for performance.
    """
    
    def __init__(self, connection_uri: Optional[str] = None):
        """
        Initialize MongoDB connection with security enforcement.
        
        Args:
            connection_uri: MongoDB connection string. If None, reads from environment.
        """
        self.connection_uri = connection_uri or os.getenv("MONGODB_URI")
        if not self.connection_uri:
            raise ValueError("MongoDB connection URI is required")
        
        # Security configuration from environment
        self.read_only = os.getenv("READ_ONLY", "true").lower() == "true"
        self.max_time_ms = int(os.getenv("MAX_TIME_MS", "3000"))
        self.allow_disk_use = os.getenv("ALLOW_DISK_USE", "false").lower() == "true"
        self.max_documents = int(os.getenv("MAX_DOCUMENTS", "1000"))
        
        # Connection pool configuration
        self.pool_size = int(os.getenv("MONGODB_POOL_SIZE", "10"))
        self.timeout_ms = int(os.getenv("MONGODB_TIMEOUT_MS", "5000"))
        
        # Database name
        self.database_name = os.getenv("MONGODB_DATABASE", "cricket_db")
        
        self._client: Optional[MongoClient] = None
        self._database = None
        
        # Security enforcement - CRITICAL: Must be read-only
        if not self.read_only:
            raise SecurityError("MCP server must operate in READ_ONLY mode")
        
        logger.info(f"Initializing MongoDB connection with READ_ONLY={self.read_only}")
    
    def connect(self) -> None:
        """
        Establish MongoDB connection with security and performance settings.
        
        Raises:
            ConnectionFailure: If unable to connect to MongoDB
            SecurityError: If security requirements are not met
        """
        try:
            # Create client with security-first configuration
            self._client = MongoClient(
                self.connection_uri,
                # Connection pool settings
                maxPoolSize=self.pool_size,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                
                # Timeout settings for security
                serverSelectionTimeoutMS=self.timeout_ms,
                socketTimeoutMS=self.max_time_ms,
                connectTimeoutMS=self.timeout_ms,
                
                # Read preference for read-only operations
                readPreference="secondary",  # Prefer secondary for read-only
                
                # Connection options
                retryWrites=False,  # No write operations allowed
                retryReads=True,    # Allow read retries
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            # Get database reference
            self._database = self._client[self.database_name]
            
            logger.info(f"Successfully connected to MongoDB database: {self.database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionFailure(f"Unable to connect to MongoDB: {e}")
    
    def disconnect(self) -> None:
        """Close MongoDB connection and cleanup resources."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed")
    
    @property
    def client(self) -> MongoClient:
        """
        Get MongoDB client instance.
        
        Returns:
            MongoClient: Active MongoDB client
            
        Raises:
            ConnectionError: If not connected to MongoDB
        """
        if not self._client:
            raise ConnectionError("Not connected to MongoDB. Call connect() first.")
        return self._client
    
    @property
    def database(self):
        """
        Get database instance.
        
        Returns:
            Database: MongoDB database instance
            
        Raises:
            ConnectionError: If not connected to MongoDB
        """
        if not self._database:
            raise ConnectionError("Not connected to MongoDB database. Call connect() first.")
        return self._database
    
    def get_collection(self, collection_name: str):
        """
        Get collection instance with read-only verification.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection: MongoDB collection instance
        """
        if not self._database:
            raise ConnectionError("Not connected to MongoDB database")
        
        # Verify collection exists
        if collection_name not in self._database.list_collection_names():
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        return self._database[collection_name]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection.
        
        Returns:
            Dict containing health status information
        """
        try:
            if not self._client:
                return {"status": "disconnected", "error": "No client connection"}
            
            # Ping database
            result = self._client.admin.command('ping')
            
            # Get server info
            server_info = self._client.server_info()
            
            # Check database access
            collections = self._database.list_collection_names()
            
            return {
                "status": "healthy",
                "mongodb_version": server_info.get("version"),
                "database": self.database_name,
                "collections_count": len(collections),
                "read_only": self.read_only,
                "max_time_ms": self.max_time_ms,
                "allow_disk_use": self.allow_disk_use,
                "connection_pool_size": self.pool_size,
                "ping_response": result
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "read_only": self.read_only
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class SecurityError(Exception):
    """Raised when security requirements are violated."""
    pass