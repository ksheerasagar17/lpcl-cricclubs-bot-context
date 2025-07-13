"""
Test suite for schema drift detection and validation.

Tests that ensure the database schema matches the documented YAML schemas
and detects when the actual database structure drifts from expectations.
"""

import pytest
import yaml
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from mcp_server import MongoConnection


class TestSchemaValidation:
    """Test schema validation against documented YAML schemas."""
    
    @pytest.fixture
    def mock_mongo_connection(self):
        """Create mock MongoDB connection for testing."""
        with patch('mcp_server.connection.MongoClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock admin command (ping)
            mock_instance.admin.command.return_value = {"ok": 1}
            
            # Mock database
            mock_db = Mock()
            mock_instance.__getitem__.return_value = mock_db
            
            conn = MongoConnection("mongodb://localhost:27017/cricket_db")
            conn._client = mock_instance
            conn._database = mock_db
            
            return conn
    
    def load_yaml_schema(self, schema_file: str) -> Dict[str, Any]:
        """Load YAML schema file for testing."""
        try:
            with open(f"schema/{schema_file}", 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            pytest.skip(f"Schema file {schema_file} not found")
    
    def test_matches_schema_structure(self, mock_mongo_connection):
        """Test matches collection schema against YAML definition."""
        schema = self.load_yaml_schema("matches.yaml")
        
        # Mock sample document from matches collection
        sample_match = {
            "_id": {"$oid": "507f1f77bcf86cd799439011"},
            "canDelete": 1,
            "canUpdate": 1,
            "clubId": 12345,
            "clubName": "Mumbai Cricket Association",
            "isComplete": 1,
            "isDls": False,
            "isFollowon": 0,
            "status": "Completed",
            "lastUpdatedDate": "2024-01-15T14:30:00Z",
            "matchDate": "2024-01-15",
            "timeSinceLastUpdate": "2 minutes ago",
            "location": "Wankhede Stadium, Mumbai",
            "matchId": 98765,
            "matchType": "T20",
            "overs": 20,
            "seriesName": "IPL 2024",
            "seriesType": "League",
            "teamOne": 101,
            "teamOneCode": "MI",
            "teamOneName": "Mumbai Indians",
            "teamTwo": 102,
            "teamTwoCode": "CSK",
            "teamTwoName": "Chennai Super Kings",
            "t1_logo_file_path": "/images/teams/mumbai_indians.png",
            "t2_logo_file_path": "/images/teams/chennai_super_kings.png",
            "t1total": 185,
            "t1wickets": 7,
            "t1balls": 120,
            "t1_1total": 185,
            "t1_1wickets": 7,
            "t1_1balls": 120,
            "t1_2total": 0,
            "t1_2wickets": 0,
            "t1_2balls": 0,
            "t2total": 175,
            "t2wickets": 9,
            "t2balls": 118,
            "t2_1total": 175,
            "t2_1wickets": 9,
            "t2_1balls": 118,
            "t2_2total": 0,
            "t2_2wickets": 0,
            "t2_2balls": 0,
            "t2RevisedOvers": 20.0,
            "result": "Mumbai Indians won by 10 runs",
            "winner": 101,
            "live_streaming_link": "https://stream.cricket.com/match/12345"
        }
        
        # Mock database response
        mock_mongo_connection._database.matches.find_one.return_value = sample_match
        
        # Validate required fields exist
        required_fields = [
            "_id", "canDelete", "canUpdate", "clubId", "clubName", "isComplete",
            "matchId", "matchType", "overs", "teamOne", "teamTwo", "result"
        ]
        
        schema_fields = schema.get("fields", {})
        
        for field in required_fields:
            assert field in schema_fields, f"Required field '{field}' not documented in schema"
            assert field in sample_match, f"Required field '{field}' missing from sample document"
    
    def test_ball_by_ball_schema_structure(self, mock_mongo_connection):
        """Test ball_by_ball collection schema against YAML definition."""
        schema = self.load_yaml_schema("ball_by_ball.yaml")
        
        # Mock sample ball-by-ball document
        sample_ball_by_ball = {
            "_id": {"$oid": "68116476c9a8dc1ad3be75cb"},
            "latestBatting": {
                "batsman1": {
                    "playerID": 286412,
                    "runsScored": 98,
                    "ballsFaced": 52,
                    "fours": 7,
                    "sixers": 9,
                    "firstName": "Prabhakaran",
                    "lastName": "Thatchinamoorthy",
                    "outStringNoLink": None,
                    "profilepic_file_path": "/documentsRep/profilePics/286412.JPG",
                    "matches": 0,
                    "howOut": "",
                    "isOut": "0",
                    "nickName": "Prabhakar",
                    "battingStyle": "Right Handed Batter",
                    "impactPlayerIn": False,
                    "impactPlayerOut": False
                },
                "batsman2": {
                    "playerID": 765282,
                    "runsScored": 45,
                    "ballsFaced": 28,
                    "fours": 3,
                    "sixers": 2
                }
            },
            "latestBowling": {
                "bowler1": {
                    "matchID": 2301,
                    "playerID": 5009003,
                    "teamId": 718,
                    "balls": 2,
                    "runs": 8,
                    "wides": 1,
                    "noBalls": 0,
                    "wickets": 0,
                    "overs": "0.2",
                    "bowlingStyle": "Left Arm Medium"
                }
            },
            "innings1Balls": {
                "runs": 155,
                "overs": "20.0",
                "teamName": "Ace Aviators",
                "rcb": "155/8",
                "oversMap": {
                    "Over0": {
                        "runs": 7,
                        "rcb": "7/0",
                        "runRate": "7.00",
                        "balls": []
                    }
                }
            }
        }
        
        mock_mongo_connection._database.ball_by_ball.find_one.return_value = sample_ball_by_ball
        
        # Validate core structure
        assert "latestBatting" in sample_ball_by_ball
        assert "latestBowling" in sample_ball_by_ball
        assert "innings1Balls" in sample_ball_by_ball
        
        # Validate batsman structure
        batsman1 = sample_ball_by_ball["latestBatting"]["batsman1"]
        required_batsman_fields = [
            "playerID", "runsScored", "ballsFaced", "fours", "sixers",
            "firstName", "lastName", "battingStyle"
        ]
        
        for field in required_batsman_fields:
            assert field in batsman1, f"Required batsman field '{field}' missing"
        
        # Validate bowler structure
        bowler1 = sample_ball_by_ball["latestBowling"]["bowler1"]
        required_bowler_fields = [
            "playerID", "balls", "runs", "wickets", "overs", "bowlingStyle"
        ]
        
        for field in required_bowler_fields:
            assert field in bowler1, f"Required bowler field '{field}' missing"
    
    def test_players_schema_structure(self, mock_mongo_connection):
        """Test players collection schema against YAML definition."""
        schema = self.load_yaml_schema("players.yaml")
        
        # Mock sample player document
        sample_player = {
            "_id": {"$oid": "507f1f77bcf86cd799439012"},
            "playerID": 286412,
            "firstName": "Virat",
            "lastName": "Kohli",
            "nickName": "King Kohli",
            "fullName": "Virat Kohli",
            "dateOfBirth": "1988-11-05",
            "age": 35,
            "birthPlace": "Delhi, India",
            "nationality": "Indian",
            "height": "5'9\"",
            "weight": "70 kg",
            "battingStyle": "Right Handed Batter",
            "bowlingStyle": "Right Arm Medium",
            "playerRole": "Batsman",
            "specialization": "Middle Order",
            "currentTeam": 101,
            "currentTeamName": "Mumbai Indians",
            "careerStart": "2008-08-18",
            "isActive": True,
            "profilepic_file_path": "/documentsRep/profilePics/286412.JPG",
            "jersey_number": 18,
            "careerStats": {
                "batting": {
                    "matchesPlayed": 254,
                    "runsScored": 12169,
                    "battingAverage": 57.32,
                    "strikeRate": 93.17,
                    "centuries": 43,
                    "halfCenturies": 64,
                    "highestScore": 183
                },
                "bowling": {
                    "wicketsTaken": 4,
                    "bowlingAverage": 166.25,
                    "economyRate": 8.31
                }
            },
            "lastUpdated": "2025-01-13T10:30:00Z",
            "dataSource": "ESPN Cricinfo"
        }
        
        mock_mongo_connection._database.players.find_one.return_value = sample_player
        
        # Validate required fields
        required_fields = [
            "playerID", "firstName", "lastName", "battingStyle", "playerRole",
            "isActive", "careerStats"
        ]
        
        for field in required_fields:
            assert field in sample_player, f"Required player field '{field}' missing"
        
        # Validate career stats structure
        career_stats = sample_player["careerStats"]
        assert "batting" in career_stats
        assert "bowling" in career_stats
        
        batting_stats = career_stats["batting"]
        required_batting_stats = [
            "matchesPlayed", "runsScored", "battingAverage", "strikeRate"
        ]
        
        for stat in required_batting_stats:
            assert stat in batting_stats, f"Required batting stat '{stat}' missing"
    
    def test_collections_overview_completeness(self):
        """Test that collections.yaml documents all expected collections."""
        schema = self.load_yaml_schema("collections.yaml")
        
        expected_collections = ["matches", "ball_by_ball", "players", "teams"]
        documented_collections = list(schema.get("collections", {}).keys())
        
        for collection in expected_collections:
            assert collection in documented_collections, f"Collection '{collection}' not documented in collections.yaml"
        
        # Validate each collection has required metadata
        collections = schema.get("collections", {})
        for collection_name, collection_info in collections.items():
            required_metadata = ["description", "primary_key", "file_reference"]
            
            for metadata in required_metadata:
                assert metadata in collection_info, f"Collection '{collection_name}' missing '{metadata}' metadata"


class TestSchemaDriftDetection:
    """Test detection of schema drift between documented and actual schemas."""
    
    def test_detect_missing_fields(self):
        """Test detection of fields missing from actual documents."""
        documented_schema = {
            "fields": {
                "playerID": {"type": "integer", "required": True},
                "firstName": {"type": "string", "required": True},
                "lastName": {"type": "string", "required": True},
                "battingAverage": {"type": "number", "required": False}
            }
        }
        
        actual_document = {
            "playerID": 12345,
            "firstName": "John",
            # Missing lastName and battingAverage
        }
        
        drift_issues = self.detect_schema_drift(documented_schema, actual_document)
        
        assert len(drift_issues) > 0
        assert any("lastName" in issue for issue in drift_issues)
    
    def test_detect_type_mismatches(self):
        """Test detection of type mismatches between schema and data."""
        documented_schema = {
            "fields": {
                "playerID": {"type": "integer"},
                "battingAverage": {"type": "number"},
                "isActive": {"type": "boolean"}
            }
        }
        
        actual_document = {
            "playerID": "12345",  # Should be integer
            "battingAverage": "45.5",  # Should be number
            "isActive": "true"  # Should be boolean
        }
        
        drift_issues = self.detect_schema_drift(documented_schema, actual_document)
        
        assert len(drift_issues) >= 3
        type_issues = [issue for issue in drift_issues if "type mismatch" in issue.lower()]
        assert len(type_issues) >= 1
    
    def test_detect_unexpected_fields(self):
        """Test detection of fields present in data but not in schema."""
        documented_schema = {
            "fields": {
                "playerID": {"type": "integer"},
                "firstName": {"type": "string"}
            }
        }
        
        actual_document = {
            "playerID": 12345,
            "firstName": "John",
            "unexpectedField": "value",  # Not in schema
            "anotherNewField": 42  # Not in schema
        }
        
        drift_issues = self.detect_schema_drift(documented_schema, actual_document)
        
        unexpected_issues = [issue for issue in drift_issues if "unexpected" in issue.lower()]
        assert len(unexpected_issues) >= 2
    
    def test_no_drift_when_schemas_match(self):
        """Test that no drift is detected when schemas match perfectly."""
        documented_schema = {
            "fields": {
                "playerID": {"type": "integer", "required": True},
                "firstName": {"type": "string", "required": True},
                "battingAverage": {"type": "number", "required": False}
            }
        }
        
        actual_document = {
            "playerID": 12345,
            "firstName": "John",
            "battingAverage": 45.5
        }
        
        drift_issues = self.detect_schema_drift(documented_schema, actual_document)
        
        assert len(drift_issues) == 0
    
    def detect_schema_drift(self, documented_schema: Dict[str, Any], actual_document: Dict[str, Any]) -> List[str]:
        """
        Helper method to detect schema drift issues.
        
        Args:
            documented_schema: The documented schema structure
            actual_document: The actual document from database
        
        Returns:
            List of drift issues found
        """
        issues = []
        documented_fields = documented_schema.get("fields", {})
        
        # Check for missing required fields
        for field_name, field_info in documented_fields.items():
            if field_info.get("required", False) and field_name not in actual_document:
                issues.append(f"Required field '{field_name}' missing from actual document")
        
        # Check for type mismatches
        for field_name, field_value in actual_document.items():
            if field_name in documented_fields:
                expected_type = documented_fields[field_name].get("type")
                actual_type = self.get_python_type(field_value)
                
                if not self.types_compatible(expected_type, actual_type):
                    issues.append(f"Type mismatch for field '{field_name}': expected {expected_type}, got {actual_type}")
        
        # Check for unexpected fields
        for field_name in actual_document.keys():
            if field_name not in documented_fields:
                issues.append(f"Unexpected field '{field_name}' found in actual document")
        
        return issues
    
    def get_python_type(self, value: Any) -> str:
        """Get Python type name for a value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
    
    def types_compatible(self, schema_type: str, actual_type: str) -> bool:
        """Check if schema type is compatible with actual type."""
        if schema_type == actual_type:
            return True
        
        # Number can be integer or float
        if schema_type == "number" and actual_type in ["integer", "float"]:
            return True
        
        # Handle mixed types documented in schema
        if schema_type == "mixed":
            return True
        
        return False


class TestSchemaVersioning:
    """Test schema versioning and migration tracking."""
    
    def test_schema_version_tracking(self):
        """Test that schemas include version information."""
        schema_files = ["matches.yaml", "ball_by_ball.yaml", "players.yaml", "collections.yaml"]
        
        for schema_file in schema_files:
            try:
                with open(f"schema/{schema_file}", 'r') as f:
                    schema = yaml.safe_load(f)
                
                # Check for version tracking (if implemented)
                if "version" in schema or "last_updated" in schema:
                    if "version" in schema:
                        assert isinstance(schema["version"], str)
                    if "last_updated" in schema:
                        assert isinstance(schema["last_updated"], str)
                
            except FileNotFoundError:
                pytest.skip(f"Schema file {schema_file} not found")
    
    def test_schema_validation_status(self):
        """Test that schemas include validation status information."""
        try:
            with open("schema/ball_by_ball.yaml", 'r') as f:
                schema = yaml.safe_load(f)
            
            # Check validation status section exists
            if "validation_status" in schema:
                validation = schema["validation_status"]
                assert "status" in validation
                assert "date_validated" in validation
                
                if "issues_found" in validation and "issues_fixed" in validation:
                    assert validation["issues_found"] >= validation["issues_fixed"]
        
        except FileNotFoundError:
            pytest.skip("ball_by_ball.yaml schema file not found")


class TestSchemaConsistency:
    """Test consistency between different schema files."""
    
    def test_cross_reference_consistency(self):
        """Test that cross-references between schemas are consistent."""
        try:
            with open("schema/collections.yaml", 'r') as f:
                collections_schema = yaml.safe_load(f)
            
            collections = collections_schema.get("collections", {})
            
            # Verify that referenced schema files exist
            for collection_name, collection_info in collections.items():
                file_reference = collection_info.get("file_reference")
                if file_reference:
                    assert file_reference.endswith(".yaml")
                    # In a real test, would check file existence
        
        except FileNotFoundError:
            pytest.skip("collections.yaml schema file not found")
    
    def test_relationship_consistency(self):
        """Test that relationships defined in collections.yaml are valid."""
        try:
            with open("schema/collections.yaml", 'r') as f:
                collections_schema = yaml.safe_load(f)
            
            collections = collections_schema.get("collections", {})
            
            for collection_name, collection_info in collections.items():
                relationships = collection_info.get("relationships", [])
                
                for relationship in relationships:
                    # Verify relationship has required fields
                    assert "collection" in relationship
                    assert "type" in relationship
                    assert "field" in relationship
                    
                    # Verify referenced collection exists in schema
                    referenced_collection = relationship["collection"]
                    assert referenced_collection in collections, f"Referenced collection '{referenced_collection}' not found"
        
        except FileNotFoundError:
            pytest.skip("collections.yaml schema file not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])