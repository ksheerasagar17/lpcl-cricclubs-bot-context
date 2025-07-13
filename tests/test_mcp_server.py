"""
Test suite for MongoDB MCP Server functionality.

Tests security enforcement, read-only operations, and query validation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from mcp_server import ReadOnlyMongoMCP, MongoConnection, SecurityValidator
from mcp_server.security import SecurityError


class TestSecurityValidator:
    """Test SecurityValidator class for query validation and sanitization."""
    
    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance for testing."""
        return SecurityValidator(max_time_ms=3000, max_documents=1000)
    
    def test_validate_allowed_operation(self, validator):
        """Test validation of allowed operations."""
        # Should not raise exception
        validator.validate_operation("find")
        validator.validate_operation("aggregate")
        validator.validate_operation("count_documents")
        validator.validate_operation("distinct")
    
    def test_validate_forbidden_operation(self, validator):
        """Test validation rejects forbidden operations."""
        with pytest.raises(SecurityError, match="forbidden in read-only mode"):
            validator.validate_operation("insert")
        
        with pytest.raises(SecurityError, match="forbidden in read-only mode"):
            validator.validate_operation("update")
        
        with pytest.raises(SecurityError, match="forbidden in read-only mode"):
            validator.validate_operation("delete")
    
    def test_sanitize_safe_query(self, validator):
        """Test sanitization of safe queries."""
        safe_query = {
            "playerID": 286412,
            "format": "T20",
            "runs": {"$gte": 50}
        }
        
        sanitized = validator.sanitize_query(safe_query)
        assert sanitized == safe_query
    
    def test_sanitize_dangerous_query(self, validator):
        """Test rejection of dangerous query operators."""
        dangerous_query = {
            "playerID": 286412,
            "$where": "this.runs > 50"  # JavaScript execution
        }
        
        with pytest.raises(SecurityError, match="Dangerous operator"):
            validator.sanitize_query(dangerous_query)
    
    def test_sanitize_large_string(self, validator):
        """Test rejection of overly large strings."""
        large_query = {
            "description": "x" * 20000  # Too large
        }
        
        with pytest.raises(SecurityError, match="String value too long"):
            validator.sanitize_query(large_query)
    
    def test_validate_collection_name(self, validator):
        """Test collection name validation."""
        # Valid names
        validator.validate_collection_name("matches")
        validator.validate_collection_name("ball_by_ball")
        validator.validate_collection_name("player_stats")
        
        # Invalid names
        with pytest.raises(SecurityError, match="Collection name cannot be empty"):
            validator.validate_collection_name("")
        
        with pytest.raises(SecurityError, match="Access to system collections is forbidden"):
            validator.validate_collection_name("system.users")
        
        with pytest.raises(SecurityError, match="invalid characters"):
            validator.validate_collection_name("matches$evil")
    
    def test_validate_aggregation_pipeline(self, validator):
        """Test aggregation pipeline validation."""
        safe_pipeline = [
            {"$match": {"teamName": "Mumbai Indians"}},
            {"$group": {"_id": "$playerID", "totalRuns": {"$sum": "$runs"}}},
            {"$sort": {"totalRuns": -1}},
            {"$limit": 10}
        ]
        
        sanitized = validator.validate_aggregation_pipeline(safe_pipeline)
        assert len(sanitized) == 4
        assert sanitized[-1] == {"$limit": 10}
    
    def test_validate_dangerous_aggregation(self, validator):
        """Test rejection of dangerous aggregation operators."""
        dangerous_pipeline = [
            {"$match": {"teamName": "Mumbai Indians"}},
            {"$where": "this.runs > 50"}  # Not allowed
        ]
        
        with pytest.raises(SecurityError, match="not allowed"):
            validator.validate_aggregation_pipeline(dangerous_pipeline)
    
    def test_ensure_pipeline_limits(self, validator):
        """Test that pipeline limits are enforced."""
        pipeline_without_limit = [
            {"$match": {"teamName": "Mumbai Indians"}},
            {"$sort": {"runs": -1}}
        ]
        
        sanitized = validator.validate_aggregation_pipeline(pipeline_without_limit)
        
        # Should add a limit
        assert any("$limit" in stage for stage in sanitized)
        limit_stage = next(stage for stage in sanitized if "$limit" in stage)
        assert limit_stage["$limit"] <= validator.max_documents
    
    def test_create_execution_context(self, validator):
        """Test execution context creation."""
        context = validator.create_execution_context()
        
        assert context["maxTimeMS"] == 3000
        assert context["allowDiskUse"] is False
        assert context["maxDocuments"] == 1000
        assert "readConcern" in context


class TestMongoConnection:
    """Test MongoConnection class."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Create mock MongoDB client."""
        with patch('mcp_server.connection.MongoClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock admin command (ping)
            mock_instance.admin.command.return_value = {"ok": 1}
            
            # Mock database and collection
            mock_db = Mock()
            mock_instance.__getitem__.return_value = mock_db
            mock_db.list_collection_names.return_value = ["matches", "ball_by_ball", "players"]
            
            yield mock_instance
    
    def test_connection_initialization(self):
        """Test connection initialization with security enforcement."""
        conn = MongoConnection("mongodb://localhost:27017/cricket_db")
        
        assert conn.read_only is True
        assert conn.max_time_ms == 3000
        assert conn.allow_disk_use is False
        assert conn.max_documents == 1000
    
    def test_connection_security_enforcement(self):
        """Test that read-only mode is enforced."""
        with patch.dict('os.environ', {'READ_ONLY': 'false'}):
            with pytest.raises(Exception):  # Should raise SecurityError
                MongoConnection("mongodb://localhost:27017/cricket_db")
    
    def test_connection_connect(self, mock_mongo_client):
        """Test database connection."""
        with patch('mcp_server.connection.MongoClient', return_value=mock_mongo_client):
            conn = MongoConnection("mongodb://localhost:27017/cricket_db")
            conn.connect()
            
            assert conn._client is not None
            assert conn._database is not None
    
    def test_connection_health_check(self, mock_mongo_client):
        """Test connection health check."""
        with patch('mcp_server.connection.MongoClient', return_value=mock_mongo_client):
            conn = MongoConnection("mongodb://localhost:27017/cricket_db")
            conn.connect()
            
            mock_mongo_client.server_info.return_value = {"version": "5.0.0"}
            
            health = conn.health_check()
            
            assert health["status"] == "healthy"
            assert health["read_only"] is True
            assert health["max_time_ms"] == 3000
            assert "mongodb_version" in health
    
    def test_get_collection_validation(self, mock_mongo_client):
        """Test collection retrieval with validation."""
        with patch('mcp_server.connection.MongoClient', return_value=mock_mongo_client):
            conn = MongoConnection("mongodb://localhost:27017/cricket_db")
            conn.connect()
            
            # Valid collection
            collection = conn.get_collection("matches")
            assert collection is not None
            
            # Invalid collection
            with pytest.raises(ValueError, match="does not exist"):
                conn.get_collection("nonexistent_collection")


class TestReadOnlyMongoMCP:
    """Test ReadOnlyMongoMCP server class."""
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Create mock MCP server for testing."""
        with patch('mcp_server.server.MongoConnection') as mock_conn_class:
            mock_conn = Mock()
            mock_conn.max_time_ms = 3000
            mock_conn.max_documents = 1000
            mock_conn_class.return_value = mock_conn
            
            server = ReadOnlyMongoMCP("mongodb://localhost:27017/cricket_db")
            server.is_connected = True
            
            return server
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test MCP server initialization."""
        with patch('mcp_server.server.MongoConnection') as mock_conn_class:
            mock_conn = Mock()
            mock_conn.max_time_ms = 3000
            mock_conn.max_documents = 1000
            mock_conn_class.return_value = mock_conn
            
            server = ReadOnlyMongoMCP("mongodb://localhost:27017/cricket_db")
            
            assert server.connection is not None
            assert server.security_validator is not None
            assert "find" in server.supported_operations
            assert "aggregate" in server.supported_operations
    
    @pytest.mark.asyncio
    async def test_execute_find_query(self, mock_mcp_server):
        """Test find query execution."""
        # Mock collection and cursor
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter([
            {"_id": "1", "playerName": "Virat Kohli", "runs": 98},
            {"_id": "2", "playerName": "MS Dhoni", "runs": 45}
        ]))
        mock_collection.find.return_value = mock_cursor
        
        mock_mcp_server.connection.get_collection.return_value = mock_collection
        
        result = await mock_mcp_server.execute_query(
            operation="find",
            collection="players",
            query={"teamName": "India"}
        )
        
        assert result["success"] is True
        assert result["operation"] == "find"
        assert len(result["result"]["documents"]) == 2
        assert result["result"]["documents"][0]["playerName"] == "Virat Kohli"
    
    @pytest.mark.asyncio
    async def test_execute_aggregate_query(self, mock_mcp_server):
        """Test aggregation query execution."""
        # Mock collection and cursor
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter([
            {"_id": "India", "totalRuns": 2500, "averageRuns": 45.5}
        ]))
        mock_collection.aggregate.return_value = mock_cursor
        
        mock_mcp_server.connection.get_collection.return_value = mock_collection
        
        pipeline = [
            {"$match": {"format": "ODI"}},
            {"$group": {"_id": "$teamName", "totalRuns": {"$sum": "$runs"}}}
        ]
        
        result = await mock_mcp_server.execute_query(
            operation="aggregate",
            collection="matches",
            query={"pipeline": pipeline}
        )
        
        assert result["success"] is True
        assert result["operation"] == "aggregate"
        assert len(result["result"]["documents"]) == 1
        assert result["result"]["documents"][0]["_id"] == "India"
    
    @pytest.mark.asyncio
    async def test_security_validation_in_query(self, mock_mcp_server):
        """Test security validation during query execution."""
        # Test forbidden operation
        with pytest.raises(SecurityError):
            await mock_mcp_server.execute_query(
                operation="insert",
                collection="matches",
                query={"teamName": "New Team"}
            )
        
        # Test invalid collection name
        with pytest.raises(SecurityError):
            await mock_mcp_server.execute_query(
                operation="find",
                collection="system.users",
                query={}
            )
    
    @pytest.mark.asyncio
    async def test_query_timeout_enforcement(self, mock_mcp_server):
        """Test that query timeouts are enforced."""
        mock_collection = Mock()
        mock_mcp_server.connection.get_collection.return_value = mock_collection
        
        await mock_mcp_server.execute_query(
            operation="find",
            collection="matches",
            query={"teamName": "India"}
        )
        
        # Verify maxTimeMS is passed to MongoDB
        mock_collection.find.assert_called()
        call_kwargs = mock_collection.find.call_args[1]
        assert call_kwargs["max_time_ms"] == 3000
    
    @pytest.mark.asyncio
    async def test_server_status(self, mock_mcp_server):
        """Test server status reporting."""
        mock_mcp_server.request_count = 50
        mock_mcp_server.error_count = 2
        
        # Mock connection health check
        mock_mcp_server.connection.health_check.return_value = {
            "status": "healthy",
            "database": "cricket_db"
        }
        
        status = await mock_mcp_server.get_server_status()
        
        assert status["server"]["status"] == "running"
        assert status["server"]["read_only"] is True
        assert status["statistics"]["requests_processed"] == 50
        assert status["statistics"]["errors_count"] == 2
        assert status["security"]["read_only_enforced"] is True
    
    @pytest.mark.asyncio
    async def test_schema_info_retrieval(self, mock_mcp_server):
        """Test schema information retrieval."""
        # Mock database operations
        mock_mcp_server.connection.database.list_collection_names.return_value = [
            "matches", "ball_by_ball", "players", "teams"
        ]
        
        mock_mcp_server.connection.database.command.return_value = {
            "count": 1000,
            "avgObjSize": 2048,
            "size": 2048000
        }
        
        schema_info = await mock_mcp_server.get_schema_info()
        
        assert schema_info["database"] == "cricket_db"
        assert "collections" in schema_info
        assert "matches" in schema_info["collections"]
        assert schema_info["collections"]["matches"]["document_count"] == 1000


class TestSecurityIntegration:
    """Integration tests for security features."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_enforcement(self):
        """Test end-to-end security enforcement."""
        with patch('mcp_server.connection.MongoClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.admin.command.return_value = {"ok": 1}
            
            # Mock database
            mock_db = Mock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.list_collection_names.return_value = ["matches"]
            
            # Create server
            server = ReadOnlyMongoMCP("mongodb://localhost:27017/cricket_db")
            await server.start()
            
            # Test that dangerous queries are blocked
            with pytest.raises(SecurityError):
                await server.execute_query(
                    operation="find",
                    collection="matches",
                    query={"$where": "this.runs > 100"}
                )
            
            # Test that write operations are blocked
            with pytest.raises(SecurityError):
                await server.execute_query(
                    operation="insert",
                    collection="matches",
                    query={"teamName": "New Team"}
                )
    
    def test_security_logging(self):
        """Test that security violations are logged."""
        from mcp_server.security import log_security_event
        
        with patch('mcp_server.security.logger') as mock_logger:
            log_security_event("query_security_violation", {
                "operation": "insert",
                "collection": "matches",
                "error": "Forbidden operation"
            })
            
            mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])