"""
Vector Store package for intelligent tool selection.

Optional Chroma integration for semantic search of analytics helpers.
"""

__version__ = "0.1.0"

from .embeddings import AnalyticsEmbeddings
from .retriever import ToolRetriever

__all__ = ["AnalyticsEmbeddings", "ToolRetriever"]