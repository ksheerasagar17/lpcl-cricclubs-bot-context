"""
Read-only MongoDB MCP Server package.

Provides secure, read-only access to cricket databases through MCP protocol.
"""

__version__ = "0.1.0"

from .server import ReadOnlyMongoMCP
from .connection import MongoConnection

__all__ = ["ReadOnlyMongoMCP", "MongoConnection"]