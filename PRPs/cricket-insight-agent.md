name: "Cricket-Insight Agent - LangChain MCP Integration with Web UI"
description: |

## Purpose
Comprehensive PRP for building a production-ready Cricket-Insight Agent with LangChain function-calling, read-only MongoDB MCP server integration, Streamlit web UI, and Docker deployment with FluxCD.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Build a complete Cricket-Insight Agent system that provides intelligent cricket data analysis through a LangChain-powered agent connected to a read-only MongoDB MCP server, featuring a Streamlit web UI for internal demos and Docker deployment ready for FluxCD integration.

## Why
- **Business value**: Provides intelligent cricket data insights without requiring deep MongoDB knowledge
- **Integration**: Demonstrates modern AI agent architecture with MCP protocol
- **Problems solved**: Enables natural language queries over cricket databases with safety through read-only access

## What
A complete system comprising:
- LangChain function-calling agent with OpenAI gpt-4o-mini
- Read-only MongoDB MCP server with find/aggregate operations only
- Static YAML schema glossary for cricket data structure
- Curated analytics helper tools for common queries
- Streamlit web UI for interactive chat demos
- Optional Chroma vector store for intelligent tool selection
- Dockerized deployment with FluxCD configuration

### Success Criteria
- [ ] LangChain agent successfully connects to MongoDB MCP server
- [ ] Agent can execute read-only find and aggregate operations
- [ ] Static YAML schema glossary provides cricket data structure
- [ ] Analytics helper tools improve query performance
- [ ] Streamlit UI provides interactive chat interface with streaming
- [ ] Docker containers deploy successfully with FluxCD
- [ ] All security flags enforced (READ_ONLY=true, MAX_TIME_MS=3000)
- [ ] Comprehensive tests cover schema drift and helper functions

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://python.langchain.com
  why: LangChain agent creation patterns, tool integration, streaming responses
  
- url: https://github.com/mongodb-js/mongodb-mcp-server  
  why: Official MongoDB MCP server setup, read-only configuration patterns
  section: Configuration with READ_ONLY=true, operations find/aggregate only
  critical: Security flags READ_ONLY=true, MAX_TIME_MS=3000, ALLOW_DISK_USE=false

- url: https://platform.openai.com/docs/guides/function-calling
  why: OpenAI function calling with tools API, streaming responses
  section: Tools vs functions (use tools), parallel_tool_calls parameter
  critical: 2025 API uses tools not functions, supports parallel execution
  
- file: use-cases/mcp-server/src/index.ts
  why: MCP server integration patterns, authentication, tool registration
  
- file: use-cases/mcp-server/examples/database-tools.ts
  why: Database tool creation patterns, error handling, security validation
  
- file: PRPs/templates/prp_base.md
  why: Standard PRP structure and validation patterns to follow
  
- file: CLAUDE.md
  why: Project conventions, testing requirements, code structure rules
  critical: Never create files longer than 500 lines, use venv_linux, type hints required

- docfile: schema/ball_by_ball.yaml
  why: Example cricket ball by ball schema glossary.
  
- docfile: schema/matches.yaml
  why: Cricket matches schema glossary
```

### Current Codebase tree
```bash
lpcl-cricclubs-bot-context/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md
│   │   └── execute-prp.md
│   └── settings.local.json
├── schema/
│   ├── ball_by_ball.yaml
│   └── matches.yaml
├── PRPs/
│   ├── templates/prp_base.md
│   └── EXAMPLE_multi_agent_prp.md
├── examples/                     # Currently empty (just .gitkeep)
├── use-cases/mcp-server/         # Reference MCP implementation patterns
│   ├── src/
│   │   ├── index.ts             # Authenticated MCP server pattern
│   │   ├── database/            # Database security patterns
│   │   └── tools/               # Tool registration system
│   └── examples/database-tools.ts # Database tool patterns
├── CLAUDE.md                     # Project rules and conventions
└── README.md
```

### Desired Codebase tree with files to be added
```bash
lpcl-cricclubs-bot-context/
├── cricket_agent/                # Main agent package
│   ├── __init__.py              # Package init
│   ├── agent.py                 # LangChain agent with MCP integration
│   ├── tools.py                 # MCP tool wrappers for agent
│   ├── prompts.py               # System prompts for cricket analysis
│   └── config.py                # Configuration management
├── mcp_server/                  # MongoDB MCP server
│   ├── __init__.py              # Package init
│   ├── server.py                # Read-only MongoDB MCP server
│   ├── connection.py            # MongoDB connection management
│   └── security.py              # Read-only validation and safety
├── analytics/                   # Curated helper tools
│   ├── __init__.py              # Package init
│   ├── batting_stats.py         # Batting analysis helpers
│   ├── bowling_stats.py         # Bowling analysis helpers
│   ├── match_analysis.py        # Match analysis helpers
│   └── team_performance.py      # Team performance helpers
├── schema/                      # Static YAML schema glossary
│   ├── collections.yaml         # Collection definitions
│   ├── ball_by_ball.yaml       # Ball-by-ball data schema
│   ├── matches.yaml             # Matches data schema
│   └── players.yaml             # Players data schema
├── vector_store/                # Optional Chroma integration
│   ├── __init__.py              # Package init
│   ├── embeddings.py            # Helper docstring embeddings
│   └── retriever.py             # Smart tool selection
├── tests/                       # Comprehensive test suite
│   ├── __init__.py              # Package init
│   ├── test_agent.py            # Agent functionality tests
│   ├── test_mcp_server.py       # MCP server tests
│   ├── test_analytics.py        # Helper tools tests
│   ├── test_schema_drift.py     # Schema validation tests
│   └── fixtures/                # Test data fixtures
├── docker/                      # Docker configuration
│   ├── Dockerfile.mcp           # MCP server container
│   ├── Dockerfile.agent         # Agent container
│   └── docker-compose.yml       # Local development setup
├── deployment/                  # FluxCD configuration
│   ├── kustomization.yaml       # Kustomize configuration
│   ├── mcp-deployment.yaml      # MCP server deployment
│   └── agent-deployment.yaml    # Agent deployment
├── streamlit_app.py             # Web UI for chat demo
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── pyproject.toml               # Project configuration
└── README.md                    # Updated with full setup
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: LangChain requires async/await patterns for streaming
# Example: agent.astream() not agent.stream() for async streaming

# CRITICAL: MongoDB MCP server requires specific connection URI format
# mongodb://username:password@host:port/database?authSource=admin

# CRITICAL: OpenAI 2025 API uses tools not functions
# Use ChatOpenAI.bind_tools() not ChatOpenAI.bind_functions()

# CRITICAL: Streamlit requires special handling for async code
# Use asyncio.run() or st.run() for async agent calls

# CRITICAL: Docker containers need proper environment variable passing
# Use --env-file for development, ConfigMaps/Secrets for production

# CRITICAL: FluxCD requires proper resource labels and selectors
# app.kubernetes.io/name and app.kubernetes.io/instance required

# CRITICAL: Chroma vector store needs persistence configuration
# Set persist_directory to avoid losing embeddings on restart

# CRITICAL: MCP security flags must be enforced
# READ_ONLY=true, MAX_TIME_MS=3000, ALLOW_DISK_USE=false mandatory

# CRITICAL: Use python-dotenv and load_dotenv() for environment variables
# Never hardcode API keys or connection strings

# CRITICAL: All analytics helpers need docstring examples for vector store
# Example: "Usage: get_batting_average(player_id='kohli', format='ODI')"
```

## Implementation Blueprint

### Data models and structure

Create the core data models to ensure type safety and cricket domain consistency.
```python
# cricket_agent/models.py - Cricket domain models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MatchFormat(str, Enum):
    TEST = "test"
    ODI = "odi" 
    T20 = "t20"
    T20I = "t20i"

class CricketQuery(BaseModel):
    query: str = Field(..., description="Natural language cricket analysis query")
    format: Optional[MatchFormat] = Field(None, description="Limit to specific match format")
    team: Optional[str] = Field(None, description="Focus on specific team")
    player: Optional[str] = Field(None, description="Focus on specific player")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")

class BattingStats(BaseModel):
    player_id: str
    runs: int
    balls_faced: int
    fours: int
    sixes: int
    strike_rate: float
    average: Optional[float] = None

class BowlingStats(BaseModel):
    player_id: str
    overs: float
    runs_conceded: int
    wickets: int
    economy_rate: float
    average: Optional[float] = None

class AnalyticsResult(BaseModel):
    query: str
    result_type: str = Field(..., description="Type of analysis performed")
    data: Dict[str, Any] = Field(..., description="Analysis results")
    execution_time_ms: int
    data_source: str = Field(default="mongodb", description="Source of data")
    cached: bool = Field(default=False, description="Whether result was cached")
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Setup Project Structure and Configuration
CREATE project structure following CLAUDE.md conventions:
  - PATTERN: Never create files longer than 500 lines
  - CREATE cricket_agent/, mcp_server/, analytics/, schema/, tests/ packages
  - CREATE .env.example with all required environment variables
  - CREATE requirements.txt with exact versions
  - CREATE pyproject.toml with project metadata

Task 2: Implement Static YAML Schema Glossary
CREATE schema/ directory with cricket data structures:
  - MIRROR: Use schema-ball_by_ball.json and schema_matches.json as reference
  - CREATE collections.yaml with all cricket collections
  - CREATE detailed schemas for ball_by_ball, matches, players, teams
  - INCLUDE: Field descriptions, data types, sample values
  - VALIDATE: Against existing JSON schemas in project

Task 3: Build Read-Only MongoDB MCP Server
CREATE mcp_server/ package with security-first approach:
  - PATTERN: Follow use-cases/mcp-server/src/index.ts security patterns
  - IMPLEMENT: Connection management with connection pooling
  - ENFORCE: READ_ONLY=true, MAX_TIME_MS=3000, ALLOW_DISK_USE=false
  - SUPPORT: Only find() and aggregate() operations
  - INCLUDE: Query validation and sanitization
  - ERROR: Structured error handling with sanitized messages

Task 4: Create Analytics Helper Tools
CREATE analytics/ package with curated cricket tools:
  - PATTERN: Each helper must include docstring usage example + unit test
  - CREATE batting_stats.py with common batting queries
  - CREATE bowling_stats.py with bowling analysis functions
  - CREATE match_analysis.py with match comparison tools  
  - CREATE team_performance.py with team statistics
  - OPTIMIZE: Pre-built aggregation pipelines for performance
  - DOCUMENT: Usage examples for vector store embeddings

Task 5: Implement LangChain Agent with MCP Integration
CREATE cricket_agent/ package with LangChain patterns:
  - PATTERN: Follow LangChain async patterns for streaming
  - IMPLEMENT: Agent with OpenAI gpt-4o-mini using tools API
  - INTEGRATE: MCP server connection via tools.py wrapper
  - DESIGN: System prompts specialized for cricket analysis
  - SUPPORT: Streaming responses for real-time interaction
  - FALLBACK: Raw MongoDB queries when helpers don't exist
  - PREFER: Analytics helpers over raw queries when available

Task 6: Build Optional Chroma Vector Store
CREATE vector_store/ package for intelligent tool selection:
  - SETUP: Chroma with persistent storage
  - EMBED: Analytics helper docstrings using OpenAI embeddings
  - IMPLEMENT: Semantic search for tool selection
  - PATTERN: Retrieve relevant helpers based on query similarity
  - FALLBACK: Direct MCP queries when no relevant helpers found

Task 7: Create Streamlit Web UI
CREATE streamlit_app.py for internal chat demo:
  - PATTERN: Streaming chat interface with tool visibility
  - IMPLEMENT: Async agent integration with proper error handling
  - DISPLAY: Tool calls and results in structured format
  - INCLUDE: Cricket-specific UI elements (team colors, stats tables)
  - SUPPORT: Query examples and help text
  - PREPARE: Modular design for Angular widget replacement

Task 8: Docker Configuration and Deployment
CREATE docker/ directory with multi-container setup:
  - CREATE Dockerfile.mcp for MongoDB MCP server container
  - CREATE Dockerfile.agent for LangChain agent container
  - CREATE docker-compose.yml for local development
  - INCLUDE: Proper environment variable handling
  - CONFIGURE: Health checks and resource limits
  - NETWORK: Container communication patterns

Task 9: FluxCD Deployment Configuration  
CREATE deployment/ directory with Kubernetes manifests:
  - CREATE kustomization.yaml for environment management
  - CREATE mcp-deployment.yaml with MongoDB MCP server
  - CREATE agent-deployment.yaml with LangChain agent
  - INCLUDE: ConfigMaps for non-sensitive configuration
  - INCLUDE: Secrets for API keys and connection strings
  - CONFIGURE: Service discovery and networking
  - IMPLEMENT: Rolling updates and health monitoring

Task 10: Comprehensive Testing Suite
CREATE tests/ package with full coverage:
  - PATTERN: Mirror main app structure in tests/
  - TEST: Agent functionality with mocked MCP responses
  - TEST: MCP server security and read-only enforcement
  - TEST: Analytics helpers with real cricket data fixtures
  - TEST: Schema drift detection for MongoDB collections
  - TEST: Integration tests with Docker containers
  - TEST: End-to-end Streamlit UI functionality
  - INCLUDE: Performance benchmarks for query response times

Task 11: Documentation and README
UPDATE README.md with comprehensive setup guide:
  - INCLUDE: Project tree diagram as required in INITIAL.md
  - DOCUMENT: One-liner Docker commands for Mongo + MCP
  - EXPLAIN: LangSmith log → helper stub → tests promotion workflow
  - PROVIDE: FluxCD bootstrap steps with examples
  - ADD: Environment setup and configuration guide
  - INCLUDE: Usage examples and common queries
```

### Per task pseudocode as needed

```python
# Task 3: MongoDB MCP Server
# mcp_server/server.py
from pymongo import MongoClient
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

class ReadOnlyMongoMCP:
    def __init__(self, connection_uri: str):
        # PATTERN: Connection pooling for performance
        self.client = MongoClient(
            connection_uri,
            maxPoolSize=10,
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=3000  # MAX_TIME_MS enforcement
        )
        # CRITICAL: Enforce read-only mode
        self.read_only = True
        self.allowed_operations = {"find", "aggregate", "count", "distinct"}
    
    async def execute_query(self, operation: str, collection: str, query: Dict[str, Any]) -> Dict[str, Any]:
        # PATTERN: Security validation first
        if operation not in self.allowed_operations:
            raise ValueError(f"Operation {operation} not allowed in read-only mode")
        
        # PATTERN: Query sanitization and validation
        sanitized_query = self._sanitize_query(query)
        
        # GOTCHA: MongoDB timeout enforcement via maxTimeMS
        if operation == "find":
            cursor = self.client.db[collection].find(
                sanitized_query,
                maxTimeMS=3000  # CRITICAL: MAX_TIME_MS=3000
            )
            # ALLOW_DISK_USE=false is default for find operations
            return {"results": list(cursor), "operation": "find"}
        
        elif operation == "aggregate":
            # CRITICAL: allowDiskUse=False enforced
            cursor = self.client.db[collection].aggregate(
                sanitized_query,
                maxTimeMS=3000,
                allowDiskUse=False  # CRITICAL: ALLOW_DISK_USE=false
            )
            return {"results": list(cursor), "operation": "aggregate"}

# Task 5: LangChain Agent
# cricket_agent/agent.py
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from .tools import MCPTool, AnalyticsHelperTool

class CricketInsightAgent:
    def __init__(self, mcp_uri: str, openai_api_key: str):
        # PATTERN: OpenAI 2025 tools API
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.1,  # Consistent cricket analysis
            streaming=True    # CRITICAL: Enable streaming
        )
        
        # CRITICAL: Use tools not functions (2025 API)
        self.tools = self._create_tools(mcp_uri)
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._get_cricket_prompt()
        )
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def astream_response(self, query: str):
        # PATTERN: Async streaming for real-time responses
        async for chunk in self.executor.astream({"input": query}):
            # GOTCHA: Handle different chunk types in streaming
            if "actions" in chunk:
                # Tool calls in progress
                yield {"type": "tool_call", "data": chunk["actions"]}
            elif "output" in chunk:
                # Final response
                yield {"type": "response", "data": chunk["output"]}
            elif "intermediate_steps" in chunk:
                # Tool results
                yield {"type": "tool_result", "data": chunk["intermediate_steps"]}

# Task 7: Streamlit UI
# streamlit_app.py
import streamlit as st
import asyncio
from cricket_agent import CricketInsightAgent

async def stream_cricket_response(agent, query):
    """Handle async agent streaming in Streamlit"""
    response_container = st.empty()
    tool_container = st.empty()
    
    async for chunk in agent.astream_response(query):
        # PATTERN: Real-time UI updates during streaming
        if chunk["type"] == "tool_call":
            tool_container.info(f"🛠 Using tool: {chunk['data']}")
        elif chunk["type"] == "response":
            response_container.markdown(chunk["data"])

def main():
    st.title("🏏 Cricket Insight Agent")
    
    # PATTERN: Session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # User input
    if prompt := st.chat_input("Ask about cricket statistics..."):
        # GOTCHA: Streamlit requires asyncio.run for async code
        with st.spinner("Analyzing cricket data..."):
            asyncio.run(stream_cricket_response(agent, prompt))
```

### Integration Points
```yaml
ENVIRONMENT:
  - add to: .env.example
  - vars: |
      # LLM Configuration
      OPENAI_API_KEY=sk-...
      LLM_MODEL=gpt-4o-mini
      
      # MongoDB MCP Server
      MCP_URI=http://localhost:8000/mcp
      MONGODB_URI=mongodb://localhost:27017/cricket_db
      
      # Security flags
      READ_ONLY=true
      MAX_TIME_MS=3000
      ALLOW_DISK_USE=false
      
      # Optional features
      VERBOSE_LOGGING=true
      CHROMA_PERSIST_DIR=./vector_store/data
      
DATABASE:
  - collections: "matches, ball_by_ball, players, teams, venues"
  - indexes: "CREATE INDEX idx_match_date ON matches(start_date)"
  - read_only: "ALL operations must be read-only find/aggregate"
  
CONFIG:
  - schema_path: "./schema/"
  - analytics_path: "./analytics/"
  - vector_store: "./vector_store/data/"
  
DOCKER:
  - base_image: "python:3.11-slim"
  - expose_ports: "8000 (MCP), 8501 (Streamlit)"
  - volumes: "schema/, vector_store/data/"
  
FLUXCD:
  - namespace: "cricket-analytics"
  - labels: "app.kubernetes.io/name=cricket-insight"
  - config_map: "cricket-config"
  - secret: "cricket-secrets"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check . --fix                  # Auto-fix style issues
mypy .                             # Type checking
pytest tests/ --no-cov -q         # Quick test run

# Expected: No errors. If errors, READ and fix.
```

### Level 2: Unit Tests
```python
# test_agent.py - Cricket agent tests
async def test_agent_batting_query():
    """Test agent handles batting statistics queries"""
    agent = CricketInsightAgent(mcp_uri="mock://", openai_api_key="test")
    
    # Mock MCP responses
    with patch('cricket_agent.tools.MCPTool.execute') as mock_mcp:
        mock_mcp.return_value = {"runs": 1000, "average": 50.0}
        
        result = await agent.astream_response("What is Kohli's batting average?")
        assert "batting average" in result.lower()
        assert "kohli" in result.lower()

# test_mcp_server.py - Security tests
def test_read_only_enforcement():
    """Test MCP server rejects write operations"""
    server = ReadOnlyMongoMCP("mongodb://localhost:27017/test")
    
    with pytest.raises(ValueError, match="not allowed in read-only mode"):
        server.execute_query("insert", "players", {"name": "test"})
    
    with pytest.raises(ValueError, match="not allowed in read-only mode"):
        server.execute_query("update", "matches", {"$set": {"result": "win"}})

# test_analytics.py - Helper tools tests  
def test_batting_average_helper():
    """Test batting average calculation with edge cases"""
    from analytics.batting_stats import get_batting_average
    
    # Happy path
    avg = get_batting_average("kohli", format="ODI", min_innings=10)
    assert isinstance(avg, float)
    assert 0 <= avg <= 200  # Reasonable cricket average range
    
    # Edge case: No data
    avg = get_batting_average("nonexistent_player", format="ODI")
    assert avg is None

# test_schema_drift.py - Schema validation
def test_schema_matches_database():
    """Test YAML schemas match actual MongoDB structure"""
    import yaml
    from pymongo import MongoClient
    
    # Load YAML schema
    with open("schema/matches.yaml") as f:
        schema = yaml.safe_load(f)
    
    # Get actual database structure
    client = MongoClient(os.getenv("MONGODB_URI"))
    sample_doc = client.cricket.matches.find_one()
    
    # Validate all schema fields exist in sample
    for field in schema["required_fields"]:
        assert field in sample_doc, f"Schema field {field} not found in database"
```

```bash
# Run comprehensive tests:
pytest tests/ -v --cov=cricket_agent --cov=mcp_server --cov=analytics --cov-report=term-missing

# Expected: >90% coverage, all tests pass
# If failing: Debug specific test, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test Docker containers
docker-compose up --build
# Expected: All containers start, health checks pass

# Test MCP server endpoint
curl http://localhost:8000/health
# Expected: {"status": "healthy", "read_only": true}

# Test Streamlit UI
open http://localhost:8501
# Expected: Cricket UI loads, can submit queries

# Test end-to-end query
python -c "
import asyncio
from cricket_agent import CricketInsightAgent
agent = CricketInsightAgent('http://localhost:8000/mcp', 'sk-test')
result = asyncio.run(agent.astream_response('Show top 5 batsmen by average'))
print(result)
"
# Expected: Agent returns cricket statistics from MongoDB
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check .`
- [ ] No type errors: `mypy .`
- [ ] MCP server enforces read-only mode
- [ ] Agent streams responses with tool visibility
- [ ] Analytics helpers improve query performance
- [ ] Streamlit UI provides interactive cricket analysis
- [ ] Docker containers build and run successfully
- [ ] FluxCD manifests deploy to Kubernetes
- [ ] Schema drift tests validate database structure
- [ ] Environment variables properly configured
- [ ] Security flags enforced (READ_ONLY, timeouts)
- [ ] Documentation includes all required sections

---

## Anti-Patterns to Avoid
- ❌ Don't create files longer than 500 lines - split into modules
- ❌ Don't allow write operations to MongoDB - enforce read-only
- ❌ Don't hardcode connection strings - use environment variables
- ❌ Don't skip schema validation - implement drift detection
- ❌ Don't ignore timeouts - enforce MAX_TIME_MS limits
- ❌ Don't use functions API - use OpenAI tools API (2025)
- ❌ Don't block Streamlit UI - use async patterns properly
- ❌ Don't skip docstring examples - required for vector embeddings

## Confidence Score: 9/10

High confidence due to:
- Clear requirements from INITIAL.md specification
- Comprehensive research of LangChain, MongoDB MCP, and OpenAI APIs
- Strong reference patterns from use-cases/mcp-server implementation
- Detailed security and read-only validation requirements
- Well-established project conventions in CLAUDE.md
- Comprehensive validation loops from syntax to deployment

Minor uncertainty around FluxCD-specific configuration details, but Kubernetes patterns are well-documented and the Docker foundation provides clear deployment path.