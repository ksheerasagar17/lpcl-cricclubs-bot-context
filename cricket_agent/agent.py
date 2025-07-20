"""
Cricket-Insight Agent with LangChain and OpenAI gpt-4o-mini integration.

This module implements the main agent that provides intelligent cricket data analysis
through natural language interface with MCP server integration.
"""

import asyncio
import logging
import os
import yaml
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import CricketConfig
# from .tools import MCPTool, AnalyticsHelperTool, get_available_tools
from .prompts import get_cricket_system_prompt

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming responses."""
    
    def __init__(self):
        self.response_chunks = []
        self.current_tool_calls = []
        
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        logger.debug("LLM processing started")
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when new token is generated."""
        self.response_chunks.append(token)
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool execution starts."""
        tool_name = serialized.get("name", "unknown")
        self.current_tool_calls.append({
            "tool": tool_name,
            "input": input_str,
            "start_time": datetime.utcnow()
        })
        logger.info(f"Tool execution started: {tool_name}")
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool execution ends."""
        if self.current_tool_calls:
            tool_call = self.current_tool_calls[-1]
            tool_call["output"] = output
            tool_call["end_time"] = datetime.utcnow()
            tool_call["duration"] = (tool_call["end_time"] - tool_call["start_time"]).total_seconds()
            logger.info(f"Tool execution completed: {tool_call['tool']} ({tool_call['duration']:.2f}s)")


async def get_mcp_tools_with_schema(client):
    tools = await client.get_tools()
    
    # for tool in tools:
    #     if tool.name in ['find', 'aggregate', 'find_one']:
    #         # Add schema context to tool description
    #         tool.description = f"""{tool.description}
            
    #         Schema Context:
    #         {yaml.dump(schema_data, default_flow_style=False)}"""
    
    return tools

class CricketInsightAgent:
    """
    Cricket-Insight Agent with LangChain and OpenAI gpt-4o-mini integration.
    
    Provides intelligent cricket data analysis through natural language queries
    with secure MCP server integration and analytics helper tools.
    """
    
    def __init__(
        self,
        config: Optional[CricketConfig] = None,
        enable_streaming: bool = True,
        enable_analytics_helpers: bool = True
    ):
        """
        Initialize Cricket-Insight Agent.
        
        Args:
            config: Configuration object (uses default if None)
            enable_streaming: Enable streaming responses
            enable_analytics_helpers: Enable analytics helper tools
        """
        self.config = config or CricketConfig()
        self.enable_streaming = enable_streaming
        self.enable_analytics_helpers = enable_analytics_helpers

        self.mcp_client = MultiServerMCPClient(
            {
            "mongodb": {
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "-e",
                    f"MDB_MCP_CONNECTION_STRING={self.config.mcp_uri}/{self.config.mongodb_database}",
                    "mongodb/mongodb-mcp-server:latest",
                ],
                "transport": "stdio",
            }
        }
        )
        
        # Initialize OpenAI model with 2025 tools API
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            api_key=self.config.openai_api_key,
            temperature=self.config.llm_temperature,
            streaming=enable_streaming,
            max_tokens=2000,  # Reasonable limit for cricket analysis
            timeout=30.0  # 30 second timeout
        )
        
        # Initialize tools
        async with self.mcp_client.session("mongodb") as session:
            self.tools = await load_mcp_tools(session)
            
        # Create agent with tools (2025 API pattern)
        self.agent = self._create_agent()
        
        # Create agent executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.config.verbose_logging,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent infinite loops
            early_stopping_method="generate"
        )
        
        logger.info(f"Cricket-Insight Agent initialized with {len(self.tools)} tools")
    
    async def _initialize_tools(self) -> List[BaseTool]:
        """
        Initialize available tools for the agent.
        
        Returns:
            List of initialized tools
        """
        tools = []
        
        try:
            logger.info("loading MCP tools")
            # Get MCP tools
            tools = await get_mcp_tools_with_schema(self.mcp_client)
            # mcp_tools = get_available_tools(
            #     mcp_uri=self.config.mcp_uri,
            #     include_analytics=self.enable_analytics_helpers
            # )
            # tools.extend(mcp_tools)
            
            logger.info(f"Loaded {len(mcp_tools)} MCP tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
            # Continue without MCP tools - agent can still work with analytics helpers

        return tools
    
    def _create_agent(self):
        """
        Create LangChain agent with cricket-specific prompts.
        
        Returns:
            Configured LangChain agent
        """
        # Get cricket-specific system prompt
        system_prompt = get_cricket_system_prompt(
            tools_available=len(self.tools),
            mcp_enabled = False,
            # mcp_enabled=any(isinstance(tool, MCPTool) for tool in self.tools),
            analytics_enabled=self.enable_analytics_helpers
        )
        
        # Create prompt template with cricket context
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent with OpenAI tools (2025 API)
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    async def aquery(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a cricket analysis query asynchronously.
        
        Args:
            query: Natural language cricket query
            chat_history: Previous conversation history
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with query results and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Prepare input
            agent_input = {
                "input": query,
                "chat_history": chat_history or []
            }
            
            # Execute query
            result = await self.executor.ainvoke(agent_input)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query,
                "response": result.get("output", ""),
                "metadata": {
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat(),
                    "tools_used": self._extract_tools_used(result),
                    "model": self.config.llm_model
                }
            }
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Query execution failed: {e}")
            
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "metadata": {
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat(),
                    "model": self.config.llm_model
                }
            }
    
    async def astream_response(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream cricket analysis response asynchronously.
        
        Args:
            query: Natural language cricket query
            chat_history: Previous conversation history
            **kwargs: Additional parameters
        
        Yields:
            Dictionary chunks with response data
        """
        if not self.enable_streaming:
            # Fallback to regular query if streaming disabled
            result = await self.aquery(query, chat_history, **kwargs)
            yield result
            return
        
        start_time = datetime.utcnow()
        callback_handler = StreamingCallbackHandler()
        
        try:
            # Prepare input
            agent_input = {
                "input": query,
                "chat_history": chat_history or []
            }
            
            # Stream response
            async for chunk in self.executor.astream(
                agent_input,
                config={"callbacks": [callback_handler]}
            ):
                # Process different chunk types
                if "actions" in chunk:
                    # Tool calls in progress
                    yield {
                        "type": "tool_call",
                        "data": chunk["actions"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                elif "steps" in chunk:
                    # Tool results
                    yield {
                        "type": "tool_result", 
                        "data": chunk["steps"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                elif "output" in chunk:
                    # Final response
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    yield {
                        "type": "final_response",
                        "data": chunk["output"],
                        "metadata": {
                            "execution_time_seconds": execution_time,
                            "timestamp": start_time.isoformat(),
                            "tools_used": callback_handler.current_tool_calls,
                            "model": self.config.llm_model
                        }
                    }
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Streaming query failed: {e}")
            
            yield {
                "type": "error",
                "error": str(e),
                "metadata": {
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat(),
                    "model": self.config.llm_model
                }
            }
    
    def query(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for cricket analysis query.
        
        Args:
            query: Natural language cricket query
            chat_history: Previous conversation history
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with query results and metadata
        """
        return asyncio.run(self.aquery(query, chat_history, **kwargs))
    
    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract list of tools used during query execution.
        
        Args:
            result: Agent execution result
        
        Returns:
            List of tool names used
        """
        tools_used = []
        
        # Extract from intermediate steps if available
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if hasattr(step, 'tool') and step.tool:
                    tools_used.append(step.tool)
        
        return tools_used
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about available tools.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "type": type(tool).__name__
            }
            for tool in self.tools
        ]
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration.
        
        Returns:
            Dictionary with system information
        """
        return {
            "agent": {
                "version": "1.0.0",
                "model": self.config.llm_model,
                "streaming_enabled": self.enable_streaming,
                "analytics_helpers_enabled": self.enable_analytics_helpers
            },
            "tools": {
                "total_count": len(self.tools),
                "available_tools": self.get_available_tools()
            },
            "configuration": {
                "mcp_uri": self.config.mcp_uri,
                "verbose_logging": self.config.verbose_logging,
                "temperature": self.config.llm_temperature
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on agent and dependencies.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "agent": "healthy",
            "llm": "unknown",
            "tools": "unknown",
            "mcp_server": "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Test LLM connectivity
            test_response = await self.llm.ainvoke([HumanMessage(content="Test")])
            health_status["llm"] = "healthy"
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            health_status["llm"] = "unhealthy"
            health_status["agent"] = "degraded"
        
        try:
            # Test tools availability
            if self.tools:
                health_status["tools"] = "healthy"
            else:
                health_status["tools"] = "no_tools_available"
                
        except Exception as e:
            logger.error(f"Tools health check failed: {e}")
            health_status["tools"] = "unhealthy"
        
        try:
            # Test MCP server if available
            # mcp_tools = [tool for tool in self.tools if isinstance(tool, MCPTool)]
            if mcp_tools:
                # Test one MCP tool
                # This would be implemented based on actual MCP tool interface
                health_status["mcp_server"] = "healthy"
            else:
                health_status["mcp_server"] = "not_configured"
                
        except Exception as e:
            logger.error(f"MCP server health check failed: {e}")
            health_status["mcp_server"] = "unhealthy"
        
        return health_status
    
    def __del__(self):
        """Cleanup on destruction."""
        logger.info("Cricket-Insight Agent shutting down")


# Convenience functions for easy usage

async def create_cricket_agent(
    openai_api_key: Optional[str] = None,
    mcp_uri: Optional[str] = None,
    enable_streaming: bool = True
) -> CricketInsightAgent:
    """
    Create and initialize a Cricket-Insight Agent.
    
    Args:
        openai_api_key: OpenAI API key (uses env var if None)
        mcp_uri: MCP server URI (uses env var if None)
        enable_streaming: Enable streaming responses
    
    Returns:
        Initialized CricketInsightAgent
    """
    config = CricketConfig(
        openai_api_key=openai_api_key,
        mcp_uri=mcp_uri
    )
    
    return CricketInsightAgent(
        config=config,
        enable_streaming=enable_streaming
    )


async def quick_cricket_query(
    query: str,
    openai_api_key: Optional[str] = None,
    mcp_uri: Optional[str] = None
) -> str:
    """
    Quick cricket analysis query without agent setup.
    
    Args:
        query: Cricket analysis query
        openai_api_key: OpenAI API key
        mcp_uri: MCP server URI
    
    Returns:
        Analysis response string
    """
    agent = await create_cricket_agent(
        openai_api_key=openai_api_key,
        mcp_uri=mcp_uri,
        enable_streaming=False
    )
    
    result = await agent.aquery(query)
    
    if result["success"]:
        return result["response"]
    else:
        return f"Error: {result['error']}"