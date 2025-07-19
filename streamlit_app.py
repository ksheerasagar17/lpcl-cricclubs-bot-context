"""
Cricket-Insight Agent Streamlit Web UI.

Interactive chat interface for cricket data analysis with real-time streaming,
tool visibility, and cricket-specific UI elements.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
from streamlit_chat import message

from cricket_agent import CricketInsightAgent, CricketConfig

# Configure page
st.set_page_config(
    page_title="🏏 Cricket-Insight Agent",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cricket theme
st.markdown("""
<style>
.cricket-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.tool-call {
    background-color: #f0f8ff;
    border-left: 4px solid #1e90ff;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.cricket-stat {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
}

.team-colors {
    display: flex;
    justify-content: space-around;
    margin: 10px 0;
}

.team-badge {
    background-color: #007bff;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
}

.performance-good { color: #28a745; font-weight: bold; }
.performance-average { color: #ffc107; font-weight: bold; }
.performance-poor { color: #dc3545; font-weight: bold; }

.stSpinner > div {
    border-top-color: #1e90ff !important;
}
</style>
""", unsafe_allow_html=True)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "agent_config" not in st.session_state:
        st.session_state.agent_config = None
    
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = []
    
    if "system_info" not in st.session_state:
        st.session_state.system_info = None
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False


def create_agent_from_config() -> CricketInsightAgent:
    """
    Create Cricket-Insight Agent from CricketConfig environment loading.
    
    Returns:
        Initialized CricketInsightAgent
    """
    try:
        # Load configuration from environment variables
        config = CricketConfig.from_env()
        config.validate_required_settings()
        
        agent = CricketInsightAgent(
            config=config,
            enable_streaming=True,
            enable_analytics_helpers=True
        )
        
        return agent, config
        
    except Exception as e:
        st.error(f"Failed to create agent: {e}")
        logger.error(f"Agent creation failed: {e}")
        return None, None


def display_cricket_header():
    """Display cricket-themed header."""
    st.markdown("""
    <div class="cricket-header">
        <h1>🏏 Cricket-Insight Agent</h1>
        <p>AI-powered cricket data analysis with natural language queries</p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display configuration sidebar."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Agent status
        if st.session_state.agent:
            st.success("🟢 Agent Ready")
            
            # System information
            if st.session_state.system_info:
                with st.expander("📊 System Info"):
                    info = st.session_state.system_info
                    st.write(f"**Model**: {info['agent']['model']}")
                    st.write(f"**Tools**: {info['tools']['total_count']}")
                    st.write(f"**Streaming**: {'✅' if info['agent']['streaming_enabled'] else '❌'}")
                    st.write(f"**Analytics**: {'✅' if info['agent']['analytics_helpers_enabled'] else '❌'}")
                
                # Configuration details - separate expander to avoid nesting
                if st.session_state.agent_config:
                    with st.expander("🔧 Config Details"):
                        config_display = st.session_state.agent_config.to_dict()
                        for key, value in config_display.items():
                            if key != "openai_api_key":  # Skip sensitive data
                                st.write(f"**{key}**: {value}")
        else:
            st.error("🔴 Agent Not Initialized")
            st.write("Check your environment variables:")
            st.code("""
OPENAI_API_KEY=your_api_key_here
MCP_URI=http://localhost:8000/mcp
            """)
            
            if st.button("🔄 Retry Initialization"):
                st.rerun()
        
        # Query examples
        st.subheader("🎯 Example Queries")
        
        example_queries = [
            "What is TopGuns Elite's batting performance in LPCL T20?",
            "Show me the top 5 run scorers in the LPCL Spring Leather T20",
            "Compare TopGuns Elite vs Austin Super Kings head-to-head record",
            "Analyze the powerplay performance of BraveHearts this season",
            "Which bowlers from Deccan Chargers have the best economy rate in death overs?",
            "Show partnership analysis for Austin Chargers in their highest T20 chase",
            "Get team statistics for Laidback Legends in LPCL tournaments",
            "How did Invaders perform in the 2024 Fall Leather tournament?",
            "Compare Eklavya's bowling attack with other LPCL teams"
        ]
        
        for query in example_queries:
            if st.button(f"💬 {query[:50]}{'...' if len(query) > 50 else ''}", 
                        key=f"example_{hash(query)}",
                        help=query):
                if st.session_state.agent:
                    st.session_state.pending_query = query
                    st.rerun()
                else:
                    st.error("Please initialize the agent first")


def display_tool_call(tool_call: Dict[str, Any]):
    """
    Display tool call information.
    
    Args:
        tool_call: Tool call data
    """
    with st.container():
        st.markdown(f"""
        <div class="tool-call">
            <strong>🛠 Tool: {tool_call.get('tool', 'Unknown')}</strong><br>
            <small>Started: {tool_call.get('start_time', 'Unknown')}</small>
        </div>
        """, unsafe_allow_html=True)


def display_cricket_stats(stats_data: Dict[str, Any]):
    """
    Display cricket statistics in a formatted way.
    
    Args:
        stats_data: Statistics data to display
    """
    if not stats_data:
        return
    
    # Create columns for stats display
    cols = st.columns(4)
    
    # Sample stat displays
    sample_stats = [
        {"label": "Batting Avg", "value": "45.2", "status": "good"},
        {"label": "Strike Rate", "value": "142.8", "status": "good"},
        {"label": "Economy", "value": "7.85", "status": "average"},
        {"label": "Win Rate", "value": "67%", "status": "good"}
    ]
    
    for i, stat in enumerate(sample_stats):
        with cols[i]:
            status_class = f"performance-{stat['status']}"
            st.markdown(f"""
            <div class="cricket-stat">
                <h4>{stat['label']}</h4>
                <p class="{status_class}">{stat['value']}</p>
            </div>
            """, unsafe_allow_html=True)


def format_cricket_response(response: str) -> str:
    """
    Format response text with cricket-specific formatting.
    
    Args:
        response: Raw response text
    
    Returns:
        Formatted response
    """
    # Add cricket emoji for common terms
    replacements = {
        "century": "💯 century",
        "wicket": "🎯 wicket",
        "boundary": "🏏 boundary",
        "six": "⚡ six",
        "four": "🎯 four",
        "runs": "🏃 runs",
        "average": "📊 average",
        "strike rate": "📈 strike rate"
    }
    
    formatted_response = response
    for term, replacement in replacements.items():
        if term in formatted_response.lower() and replacement not in formatted_response:
            formatted_response = formatted_response.replace(term, replacement)
    
    return formatted_response


async def stream_cricket_response(agent: CricketInsightAgent, query: str):
    """
    Handle streaming response from cricket agent.
    
    Args:
        agent: Cricket-Insight Agent instance
        query: User query
    """
    response_container = st.empty()
    tool_container = st.empty()
    current_response = ""
    
    try:
        async for chunk in agent.astream_response(query):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "tool_call":
                # Display tool call information
                with tool_container.container():
                    st.info(f"🛠 Using tool: {chunk['data']}")
                    display_tool_call({"tool": str(chunk['data'])})
            
            elif chunk_type == "tool_result":
                # Display tool results
                with tool_container.container():
                    st.success("✅ Tool execution completed")
            
            elif chunk_type == "final_response":
                # Display final response
                current_response = chunk["data"]
                formatted_response = format_cricket_response(current_response)
                
                with response_container.container():
                    st.markdown(formatted_response)
                    
                    # Display metadata
                    metadata = chunk.get("metadata", {})
                    if metadata:
                        with st.expander("📋 Query Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Execution Time**: {metadata.get('execution_time_seconds', 0):.2f}s")
                                st.write(f"**Model**: {metadata.get('model', 'Unknown')}")
                            with col2:
                                st.write(f"**Timestamp**: {metadata.get('timestamp', 'Unknown')}")
                                if metadata.get('tools_used'):
                                    st.write(f"**Tools Used**: {len(metadata['tools_used'])}")
                
                # Store message in session state
                st.session_state.messages.append({
                    "user": query,
                    "assistant": current_response,
                    "timestamp": datetime.now(),
                    "metadata": metadata
                })
            
            elif chunk_type == "error":
                # Handle errors
                error_msg = chunk.get("error", "Unknown error occurred")
                st.error(f"❌ Error: {error_msg}")
                
                return
    
    except Exception as e:
        st.error(f"❌ Streaming failed: {e}")
        logger.error(f"Streaming error: {e}")


def display_chat_interface():
    """Display the main chat interface."""
    if not st.session_state.agent:
        st.warning("🔧 Please configure and initialize the agent in the sidebar to start chatting!")
        return
    
    st.subheader("💬 Cricket Analysis Chat")
    
    # Chat input
    query = st.chat_input("Ask about cricket statistics, player performance, team analysis...")
    
    # Handle pending query from examples
    if hasattr(st.session_state, 'pending_query'):
        query = st.session_state.pending_query
        delattr(st.session_state, 'pending_query')
    
    # Process query
    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🏏 Analyzing cricket data..."):
                # Run async streaming
                asyncio.run(stream_cricket_response(st.session_state.agent, query))
    
    # Display chat history
    if st.session_state.messages:
        st.subheader("📜 Chat History")
        
        for i, msg in enumerate(reversed(st.session_state.messages[-10:])):  # Show last 10
            with st.expander(f"Q: {msg['user'][:100]}{'...' if len(msg['user']) > 100 else ''}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Query:**")
                    st.write(msg['user'])
                    st.write("**Response:**")
                    st.write(format_cricket_response(msg['assistant']))
                
                with col2:
                    st.write("**Details:**")
                    st.write(f"Time: {msg['timestamp'].strftime('%H:%M:%S')}")
                    if 'metadata' in msg and msg['metadata']:
                        st.write(f"Duration: {msg['metadata'].get('execution_time_seconds', 0):.2f}s")


def display_health_check():
    """Display system health check."""
    if st.session_state.agent:
        with st.expander("🏥 System Health Check"):
            if st.button("🔍 Run Health Check"):
                with st.spinner("Checking system health..."):
                    try:
                        # Run health check
                        health_status = asyncio.run(st.session_state.agent.health_check())
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            agent_status = health_status.get("agent", "unknown")
                            color = "🟢" if agent_status == "healthy" else "🔴"
                            st.write(f"{color} **Agent**: {agent_status}")
                        
                        with col2:
                            llm_status = health_status.get("llm", "unknown")
                            color = "🟢" if llm_status == "healthy" else "🔴"
                            st.write(f"{color} **LLM**: {llm_status}")
                        
                        with col3:
                            tools_status = health_status.get("tools", "unknown")
                            color = "🟢" if tools_status == "healthy" else "🔴"
                            st.write(f"{color} **Tools**: {tools_status}")
                        
                        with col4:
                            mcp_status = health_status.get("mcp_server", "unknown")
                            color = "🟢" if mcp_status == "healthy" else "🟡" if "not_configured" in mcp_status else "🔴"
                            st.write(f"{color} **MCP**: {mcp_status}")
                        
                        st.json(health_status)
                        
                    except Exception as e:
                        st.error(f"Health check failed: {e}")


def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Auto-initialize agent on first run
    if not st.session_state.agent_initialized:
        with st.spinner("🏏 Initializing Cricket-Insight Agent..."):
            try:
                agent, config = create_agent_from_config()
                if agent and config:
                    st.session_state.agent = agent
                    st.session_state.agent_config = config
                    st.session_state.system_info = agent.get_system_info()
                    st.session_state.agent_initialized = True
                    st.success("✅ LPCL Cricket Agent Ready!")
                    time.sleep(1)  # Brief pause to show success message
                    st.rerun()
                else:
                    st.session_state.agent_initialized = True  # Prevent infinite retry
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {e}")
                st.session_state.agent_initialized = True  # Prevent infinite retry
    
    # Display header
    display_cricket_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main chat interface
        display_chat_interface()
    
    with col2:
        # Health check and system info
        display_health_check()
        
        # Sample cricket stats display
        if st.session_state.agent:
            st.subheader("📊 Sample Stats")
            display_cricket_stats({})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🏏 LPCL Cricket-Insight Agent | Powered by OpenAI GPT-4o-mini & LangChain<br>
        <small>AI-driven cricket analytics for Austin's premier cricket league</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()