## FEATURE:

- **Cricket-Insight Agent**  
  - LangChain function-calling agent on top of a **read-only MongoDB MCP server** (`find`, `aggregate` only).  
  - **Static YAML schema glossary** in `schema/`—no runtime discovery.  
  - Falls back to raw pipelines; prefers **curated helper tools** in `analytics/` when they exist.  

- **Web UI**  
  - `streamlit_app.py` for an internal chat demo.  
  - Future-proof: swap Streamlit for an Angular widget once you fold it into your broader GenAI dashboard.

- **LLM & Optional Vector Index**  
  - OpenAI `gpt-4o-mini` (default).  
  - Optional Chroma store seeded with helper docstrings for smarter tool selection.

- **Dev & Ops**  
  - Dockerised MCP + Agent containers, deployed via **FluxCD**.  
  - `tests/` include schema-drift checks and helper unit tests.

---

## EXAMPLES:

`examples/` shows patterns—**don’t copy directly** (different domain).

| Path | Why glance at it |
|------|------------------|
| `examples/agent/` | Best practices for provider-agnostic agent setup, env loading, error handling. |
| `examples/schema/` | Mini YAML glossary sample—mirrors our manual-schema approach. |
| `examples/streamlit_app.py` (if present) | Pattern for streaming chat; adapt for `streamlit_app.py`. |

---

## DOCUMENTATION:

- LangChain → <https://python.langchain.com>  
- MongoDB MCP → <https://github.com/mongodb/mongodb-mcp-server>  
- OpenAI Function-calling → <https://platform.openai.com/docs/guides/function-calling>

---

## OTHER CONSIDERATIONS:

- Provide `.env.example` with  
  `OPENAI_API_KEY`, `MCP_URI`, `MONGODB_URI`, `VERBOSE_LOGGING`.  
  Load via `python-dotenv`.

- README must include:  
  1. Project tree diagram (`schema/`, `analytics/`, `.claude/commands/`, etc.)  
  2. One-liner Docker commands to start Mongo + MCP (read-only, allow-list `find,aggregate`).  
  3. How to promote slow queries to helpers (LangSmith log → helper stub → tests).  
  4. FluxCD bootstrap steps.

- Security & performance flags for MCP:  
  `READ_ONLY=true`, `MAX_TIME_MS=3000`, `ALLOW_DISK_USE=false`.

- Every helper must include:  
  *docstring usage example* + unit test.