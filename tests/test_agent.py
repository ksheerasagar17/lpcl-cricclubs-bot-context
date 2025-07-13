"""
Test suite for Cricket-Insight Agent functionality.

Tests the LangChain agent integration, tool usage, and cricket-specific functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from cricket_agent import CricketInsightAgent, CricketConfig
from cricket_agent.tools import get_available_tools


class TestCricketConfig:
    """Test CricketConfig class."""
    
    def test_config_initialization_with_defaults(self):
        """Test configuration initialization with environment variables."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'sk-test123',
            'LLM_MODEL': 'gpt-4o-mini',
            'LLM_TEMPERATURE': '0.1'
        }):
            config = CricketConfig()
            assert config.openai_api_key == 'sk-test123'
            assert config.llm_model == 'gpt-4o-mini'
            assert config.llm_temperature == 0.1
    
    def test_config_validation_invalid_api_key(self):
        """Test config validation with invalid API key."""
        with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
            CricketConfig(openai_api_key="invalid-key")
    
    def test_config_validation_missing_api_key(self):
        """Test config validation with missing API key."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            CricketConfig(openai_api_key="")
    
    def test_mongodb_config_generation(self):
        """Test MongoDB configuration generation."""
        config = CricketConfig(
            openai_api_key="sk-test123",
            mongodb_uri="mongodb://localhost:27017/test",
            read_only=True,
            max_time_ms=3000
        )
        
        mongo_config = config.get_mongodb_config()
        assert mongo_config['read_only'] is True
        assert mongo_config['max_time_ms'] == 3000
        assert mongo_config['uri'] == "mongodb://localhost:27017/test"
    
    def test_llm_config_generation(self):
        """Test LLM configuration generation."""
        config = CricketConfig(
            openai_api_key="sk-test123",
            llm_model="gpt-4o-mini",
            llm_temperature=0.2
        )
        
        llm_config = config.get_llm_config()
        assert llm_config['model'] == "gpt-4o-mini"
        assert llm_config['temperature'] == 0.2
        assert llm_config['api_key'] == "sk-test123"


class TestCricketInsightAgent:
    """Test CricketInsightAgent class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return CricketConfig(
            openai_api_key="sk-test123",
            mcp_uri="http://localhost:8000/mcp",
            llm_model="gpt-4o-mini",
            verbose_logging=False
        )
    
    @pytest.fixture
    def mock_agent(self, mock_config):
        """Create mock agent for testing."""
        with patch('cricket_agent.agent.ChatOpenAI') as mock_llm, \
             patch('cricket_agent.agent.get_available_tools') as mock_tools, \
             patch('cricket_agent.agent.create_openai_tools_agent') as mock_create_agent, \
             patch('cricket_agent.agent.AgentExecutor') as mock_executor:
            
            mock_tools.return_value = []
            mock_agent_instance = Mock()
            mock_create_agent.return_value = mock_agent_instance
            mock_executor_instance = Mock()
            mock_executor.return_value = mock_executor_instance
            
            agent = CricketInsightAgent(
                config=mock_config,
                enable_streaming=True,
                enable_analytics_helpers=True
            )
            
            return agent
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.config is not None
        assert mock_agent.enable_streaming is True
        assert mock_agent.enable_analytics_helpers is True
        assert hasattr(mock_agent, 'llm')
        assert hasattr(mock_agent, 'tools')
        assert hasattr(mock_agent, 'agent')
        assert hasattr(mock_agent, 'executor')
    
    @pytest.mark.asyncio
    async def test_agent_query_execution(self, mock_agent):
        """Test synchronous query execution."""
        # Mock the executor response
        mock_agent.executor.ainvoke = AsyncMock(return_value={
            "output": "Virat Kohli has a batting average of 57.32 in ODI cricket."
        })
        
        result = await mock_agent.aquery("What is Virat Kohli's batting average?")
        
        assert result["success"] is True
        assert "57.32" in result["response"]
        assert "metadata" in result
        assert result["metadata"]["model"] == "gpt-4o-mini"
    
    @pytest.mark.asyncio
    async def test_agent_query_error_handling(self, mock_agent):
        """Test error handling in query execution."""
        # Mock the executor to raise an exception
        mock_agent.executor.ainvoke = AsyncMock(side_effect=Exception("Test error"))
        
        result = await mock_agent.aquery("Test query")
        
        assert result["success"] is False
        assert "Test error" in result["error"]
        assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_agent_streaming_response(self, mock_agent):
        """Test streaming response functionality."""
        # Mock streaming chunks
        mock_chunks = [
            {"actions": [{"tool": "test_tool", "input": "test"}]},
            {"steps": [{"action": "test_action", "observation": "test_result"}]},
            {"output": "Final response from agent"}
        ]
        
        async def mock_astream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        mock_agent.executor.astream = mock_astream
        
        chunks = []
        async for chunk in mock_agent.astream_response("Test query"):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0]["type"] == "tool_call"
        assert chunks[1]["type"] == "tool_result"
        assert chunks[2]["type"] == "final_response"
        assert chunks[2]["data"] == "Final response from agent"
    
    def test_get_available_tools(self, mock_agent):
        """Test getting available tools information."""
        # Mock some tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_agent.tools = [mock_tool]
        
        tools_info = mock_agent.get_available_tools()
        
        assert len(tools_info) == 1
        assert tools_info[0]["name"] == "test_tool"
        assert tools_info[0]["description"] == "Test tool description"
    
    def test_get_system_info(self, mock_agent):
        """Test getting system information."""
        # Mock some tools
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_agent.tools = [mock_tool]
        
        system_info = mock_agent.get_system_info()
        
        assert "agent" in system_info
        assert "tools" in system_info
        assert "configuration" in system_info
        assert system_info["agent"]["model"] == "gpt-4o-mini"
        assert system_info["tools"]["total_count"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_agent):
        """Test health check functionality."""
        # Mock LLM response
        mock_agent.llm.ainvoke = AsyncMock(return_value=Mock())
        
        health_status = await mock_agent.health_check()
        
        assert "agent" in health_status
        assert "llm" in health_status
        assert "tools" in health_status
        assert "timestamp" in health_status
        assert health_status["llm"] == "healthy"


class TestToolsIntegration:
    """Test tools integration and functionality."""
    
    def test_get_available_tools_without_mcp(self):
        """Test getting tools without MCP server."""
        tools = get_available_tools(mcp_uri=None, include_analytics=True)
        
        # Should have analytics tools but no MCP tools
        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "batting_analysis" in tool_names
        assert "bowling_analysis" in tool_names
        assert "match_analysis" in tool_names
        assert "team_analysis" in tool_names
    
    @patch('cricket_agent.tools.ReadOnlyMongoMCP')
    def test_get_available_tools_with_mcp(self, mock_mcp_class):
        """Test getting tools with MCP server."""
        mock_mcp_instance = Mock()
        mock_mcp_class.return_value = mock_mcp_instance
        
        tools = get_available_tools(
            mcp_uri="http://localhost:8000/mcp",
            include_analytics=True
        )
        
        # Should have both MCP and analytics tools
        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "mongodb_query" in tool_names
        assert "batting_analysis" in tool_names
    
    def test_analytics_tools_structure(self):
        """Test that analytics tools have proper structure."""
        tools = get_available_tools(mcp_uri=None, include_analytics=True)
        
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, '_run')
            assert tool.name.startswith('batting_analysis') or \
                   tool.name.startswith('bowling_analysis') or \
                   tool.name.startswith('match_analysis') or \
                   tool.name.startswith('team_analysis')


class TestAsyncHelpers:
    """Test async helper functions."""
    
    @pytest.mark.asyncio
    async def test_create_cricket_agent(self):
        """Test agent creation helper function."""
        with patch('cricket_agent.agent.CricketInsightAgent') as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            from cricket_agent.agent import create_cricket_agent
            
            agent = await create_cricket_agent(
                openai_api_key="sk-test123",
                mcp_uri="http://localhost:8000/mcp"
            )
            
            assert agent == mock_agent_instance
            mock_agent_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quick_cricket_query(self):
        """Test quick query helper function."""
        with patch('cricket_agent.agent.create_cricket_agent') as mock_create_agent:
            mock_agent = Mock()
            mock_agent.aquery = AsyncMock(return_value={
                "success": True,
                "response": "Test response"
            })
            mock_create_agent.return_value = mock_agent
            
            from cricket_agent.agent import quick_cricket_query
            
            response = await quick_cricket_query(
                "Test query",
                openai_api_key="sk-test123"
            )
            
            assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_quick_cricket_query_error(self):
        """Test quick query helper with error."""
        with patch('cricket_agent.agent.create_cricket_agent') as mock_create_agent:
            mock_agent = Mock()
            mock_agent.aquery = AsyncMock(return_value={
                "success": False,
                "error": "Test error"
            })
            mock_create_agent.return_value = mock_agent
            
            from cricket_agent.agent import quick_cricket_query
            
            response = await quick_cricket_query(
                "Test query",
                openai_api_key="sk-test123"
            )
            
            assert "Error: Test error" in response


class TestStreamingCallbackHandler:
    """Test streaming callback handler."""
    
    @pytest.mark.asyncio
    async def test_callback_handler_initialization(self):
        """Test callback handler initialization."""
        from cricket_agent.agent import StreamingCallbackHandler
        
        handler = StreamingCallbackHandler()
        
        assert handler.response_chunks == []
        assert handler.current_tool_calls == []
    
    @pytest.mark.asyncio
    async def test_callback_handler_tool_tracking(self):
        """Test tool call tracking in callback handler."""
        from cricket_agent.agent import StreamingCallbackHandler
        
        handler = StreamingCallbackHandler()
        
        # Simulate tool start
        await handler.on_tool_start(
            {"name": "test_tool"},
            "test input"
        )
        
        assert len(handler.current_tool_calls) == 1
        assert handler.current_tool_calls[0]["tool"] == "test_tool"
        assert handler.current_tool_calls[0]["input"] == "test input"
        
        # Simulate tool end
        await handler.on_tool_end("test output")
        
        assert handler.current_tool_calls[0]["output"] == "test output"
        assert "duration" in handler.current_tool_calls[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])