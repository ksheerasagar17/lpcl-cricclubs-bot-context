"""
Cricket-Insight Agent package.

A LangChain-powered agent for intelligent cricket data analysis with MongoDB MCP integration.
"""

__version__ = "0.1.0"
__author__ = "Cricket Analytics Team"

from .agent import CricketInsightAgent
from .config import CricketConfig

__all__ = ["CricketInsightAgent", "CricketConfig"]