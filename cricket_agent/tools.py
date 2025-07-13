"""
LangChain tools wrapper for MCP server integration and analytics helpers.

This module provides tool wrappers that allow the LangChain agent to interact
with the MongoDB MCP server and analytics helper functions.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..mcp_server.server import ReadOnlyMongoMCP
from ..analytics import (
    batting_stats, bowling_stats, match_analysis, team_performance
)

logger = logging.getLogger(__name__)


class MCPQueryInput(BaseModel):
    """Input schema for MCP query tool."""
    operation: str = Field(description="MongoDB operation (find, aggregate, count_documents, distinct)")
    collection: str = Field(description="Collection name to query")
    query: Dict[str, Any] = Field(description="Query parameters", default_factory=dict)
    options: Optional[Dict[str, Any]] = Field(description="Additional query options", default=None)


class MCPTool(BaseTool):
    """
    Tool for executing MongoDB queries through MCP server.
    
    Provides secure, read-only access to cricket databases.
    """
    
    name: str = "mongodb_query"
    description: str = """
    Execute read-only MongoDB queries on cricket databases.
    
    Supports operations: find, aggregate, count_documents, distinct, find_one
    
    Collections available:
    - matches: Match records with scores, teams, and metadata
    - ball_by_ball: Detailed ball-by-ball match data
    - players: Player profiles and career statistics
    - teams: Team information and metadata
    
    Examples:
    - Find matches by team: {"operation": "find", "collection": "matches", "query": {"teamOneName": "Mumbai Indians"}}
    - Get player batting stats: {"operation": "find", "collection": "ball_by_ball", "query": {"latestBatting.batsman1.playerID": 286412}}
    - Count total matches: {"operation": "count_documents", "collection": "matches", "query": {}}
    """
    args_schema: Type[BaseModel] = MCPQueryInput
    
    def __init__(self, mcp_server: ReadOnlyMongoMCP):
        """
        Initialize MCP tool with server instance.
        
        Args:
            mcp_server: Initialized ReadOnlyMongoMCP server
        """
        super().__init__()
        self.mcp_server = mcp_server
    
    async def _arun(
        self,
        operation: str,
        collection: str,
        query: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute MCP query asynchronously.
        
        Args:
            operation: MongoDB operation
            collection: Collection name
            query: Query parameters
            options: Additional options
        
        Returns:
            JSON string with query results
        """
        try:
            result = await self.mcp_server.execute_query(
                operation=operation,
                collection=collection,
                query=query,
                options=options
            )
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"MCP query failed: {e}")
            return json.dumps({
                "error": str(e),
                "operation": operation,
                "collection": collection
            })
    
    def _run(
        self,
        operation: str,
        collection: str,
        query: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synchronous wrapper for MCP query."""
        return asyncio.run(self._arun(operation, collection, query, options))


class BattingAnalysisInput(BaseModel):
    """Input schema for batting analysis tool."""
    player_id: Union[str, int] = Field(description="Player identifier")
    format_type: Optional[str] = Field(description="Match format (T20, ODI, Test)", default=None)
    analysis_type: str = Field(description="Type of analysis (average, strike_rate, boundaries, milestones)")
    additional_params: Optional[Dict[str, Any]] = Field(description="Additional parameters", default=None)


class AnalyticsHelperTool(BaseTool):
    """
    Base class for analytics helper tools.
    
    Provides access to pre-built cricket analysis functions.
    """
    
    def __init__(self, function_name: str, module_name: str, description: str):
        """
        Initialize analytics helper tool.
        
        Args:
            function_name: Name of the analytics function
            module_name: Module containing the function
            description: Tool description
        """
        self.function_name = function_name
        self.module_name = module_name
        super().__init__(name=f"analytics_{function_name}", description=description)
    
    def _get_function(self):
        """Get the analytics function by name."""
        if self.module_name == "batting_stats":
            return getattr(batting_stats, self.function_name)
        elif self.module_name == "bowling_stats":
            return getattr(bowling_stats, self.function_name)
        elif self.module_name == "match_analysis":
            return getattr(match_analysis, self.function_name)
        elif self.module_name == "team_performance":
            return getattr(team_performance, self.function_name)
        else:
            raise ValueError(f"Unknown module: {self.module_name}")


class BattingAnalysisTool(AnalyticsHelperTool):
    """Tool for batting analysis functions."""
    
    name: str = "batting_analysis"
    description: str = """
    Perform batting analysis using pre-built analytics functions.
    
    Available analyses:
    - batting_average: Calculate player batting average
    - strike_rate: Calculate strike rate 
    - boundary_percentage: Analyze boundary scoring
    - innings_progression: Get ball-by-ball innings data
    - milestones: Get centuries, half-centuries, etc.
    - pressure_performance: Analyze performance under pressure
    
    Examples:
    - Get batting average: {"player_id": "286412", "analysis_type": "batting_average", "additional_params": {"format_type": "ODI"}}
    - Strike rate analysis: {"player_id": "286412", "analysis_type": "strike_rate", "additional_params": {"situation": "powerplay"}}
    """
    args_schema: Type[BaseModel] = BattingAnalysisInput
    
    def _run(
        self,
        player_id: Union[str, int],
        analysis_type: str,
        format_type: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute batting analysis.
        
        Args:
            player_id: Player identifier
            analysis_type: Type of analysis to perform
            format_type: Match format filter
            additional_params: Additional parameters
        
        Returns:
            JSON string with analysis results
        """
        try:
            # Map analysis type to function
            function_map = {
                "batting_average": batting_stats.get_batting_average,
                "strike_rate": batting_stats.get_strike_rate,
                "boundary_percentage": batting_stats.get_boundary_percentage,
                "innings_progression": batting_stats.get_innings_progression,
                "milestones": batting_stats.get_batting_milestones,
                "pressure_performance": batting_stats.get_batting_under_pressure,
                "partnership_analysis": batting_stats.get_batting_partnership_analysis,
                "venue_analysis": batting_stats.get_batting_venue_analysis
            }
            
            if analysis_type not in function_map:
                return json.dumps({"error": f"Unknown analysis type: {analysis_type}"})
            
            func = function_map[analysis_type]
            
            # Prepare parameters
            params = {"player_id": player_id}
            if format_type:
                params["format_type"] = format_type
            if additional_params:
                params.update(additional_params)
            
            # Execute function (placeholder - would connect to actual implementation)
            result = {
                "analysis_type": analysis_type,
                "player_id": player_id,
                "format_type": format_type,
                "status": "placeholder_implementation",
                "message": "This would return actual batting analysis results",
                "parameters": params
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Batting analysis failed: {e}")
            return json.dumps({"error": str(e), "analysis_type": analysis_type})


class BowlingAnalysisInput(BaseModel):
    """Input schema for bowling analysis tool."""
    player_id: Union[str, int] = Field(description="Player identifier")
    format_type: Optional[str] = Field(description="Match format (T20, ODI, Test)", default=None)
    analysis_type: str = Field(description="Type of analysis (average, economy, strike_rate, figures)")
    additional_params: Optional[Dict[str, Any]] = Field(description="Additional parameters", default=None)


class BowlingAnalysisTool(AnalyticsHelperTool):
    """Tool for bowling analysis functions."""
    
    name: str = "bowling_analysis"
    description: str = """
    Perform bowling analysis using pre-built analytics functions.
    
    Available analyses:
    - bowling_average: Calculate bowling average
    - economy_rate: Calculate economy rate
    - strike_rate: Calculate bowling strike rate
    - figures_analysis: Analyze bowling figures
    - spell_analysis: Analyze bowling spells
    - pressure_performance: Performance under pressure
    
    Examples:
    - Get bowling average: {"player_id": "5009003", "analysis_type": "bowling_average", "additional_params": {"format_type": "T20"}}
    - Economy analysis: {"player_id": "5009003", "analysis_type": "economy_rate", "additional_params": {"phase": "death_overs"}}
    """
    args_schema: Type[BaseModel] = BowlingAnalysisInput
    
    def _run(
        self,
        player_id: Union[str, int],
        analysis_type: str,
        format_type: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute bowling analysis."""
        try:
            # Map analysis type to function
            function_map = {
                "bowling_average": bowling_stats.get_bowling_average,
                "economy_rate": bowling_stats.get_economy_rate,
                "strike_rate": bowling_stats.get_bowling_strike_rate,
                "figures_analysis": bowling_stats.get_bowling_figures_analysis,
                "spell_analysis": bowling_stats.get_bowling_spell_analysis,
                "pressure_performance": bowling_stats.get_bowling_under_pressure,
                "partnership_breaker": bowling_stats.get_bowling_partnership_breaker
            }
            
            if analysis_type not in function_map:
                return json.dumps({"error": f"Unknown analysis type: {analysis_type}"})
            
            # Prepare parameters
            params = {"player_id": player_id}
            if format_type:
                params["format_type"] = format_type
            if additional_params:
                params.update(additional_params)
            
            # Execute function (placeholder)
            result = {
                "analysis_type": analysis_type,
                "player_id": player_id,
                "format_type": format_type,
                "status": "placeholder_implementation",
                "message": "This would return actual bowling analysis results",
                "parameters": params
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Bowling analysis failed: {e}")
            return json.dumps({"error": str(e), "analysis_type": analysis_type})


class MatchAnalysisInput(BaseModel):
    """Input schema for match analysis tool."""
    match_id: Union[str, int] = Field(description="Match identifier")
    analysis_type: str = Field(description="Type of analysis (compare_innings, momentum, partnerships)")
    additional_params: Optional[Dict[str, Any]] = Field(description="Additional parameters", default=None)


class MatchAnalysisTool(AnalyticsHelperTool):
    """Tool for match analysis functions."""
    
    name: str = "match_analysis"
    description: str = """
    Perform match analysis using pre-built analytics functions.
    
    Available analyses:
    - compare_innings: Compare first and second innings
    - momentum: Analyze match momentum shifts
    - partnerships: Analyze batting partnerships
    - powerplay_analysis: Analyze powerplay performance
    - death_overs_analysis: Analyze death overs
    - turning_points: Identify match turning points
    
    Examples:
    - Compare innings: {"match_id": "2301", "analysis_type": "compare_innings"}
    - Momentum analysis: {"match_id": "2301", "analysis_type": "momentum", "additional_params": {"window_size": 5}}
    """
    args_schema: Type[BaseModel] = MatchAnalysisInput
    
    def _run(
        self,
        match_id: Union[str, int],
        analysis_type: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute match analysis."""
        try:
            # Map analysis type to function
            function_map = {
                "compare_innings": match_analysis.compare_innings,
                "momentum": match_analysis.get_match_momentum,
                "partnerships": match_analysis.analyze_partnerships,
                "powerplay_analysis": match_analysis.get_powerplay_analysis,
                "death_overs_analysis": match_analysis.get_death_overs_analysis,
                "turning_points": match_analysis.get_match_turning_points,
                "chase_analysis": match_analysis.get_chase_analysis
            }
            
            if analysis_type not in function_map:
                return json.dumps({"error": f"Unknown analysis type: {analysis_type}"})
            
            # Prepare parameters
            params = {"match_id": match_id}
            if additional_params:
                params.update(additional_params)
            
            # Execute function (placeholder)
            result = {
                "analysis_type": analysis_type,
                "match_id": match_id,
                "status": "placeholder_implementation",
                "message": "This would return actual match analysis results",
                "parameters": params
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Match analysis failed: {e}")
            return json.dumps({"error": str(e), "analysis_type": analysis_type})


class TeamAnalysisInput(BaseModel):
    """Input schema for team analysis tool."""
    team_id: Union[str, int] = Field(description="Team identifier")
    analysis_type: str = Field(description="Type of analysis (statistics, form, comparison)")
    additional_params: Optional[Dict[str, Any]] = Field(description="Additional parameters", default=None)


class TeamAnalysisTool(AnalyticsHelperTool):
    """Tool for team analysis functions."""
    
    name: str = "team_analysis"
    description: str = """
    Perform team analysis using pre-built analytics functions.
    
    Available analyses:
    - statistics: Get comprehensive team statistics
    - form_analysis: Analyze recent team form
    - win_loss_record: Get detailed win/loss record
    - batting_analysis: Team batting performance
    - bowling_analysis: Team bowling performance
    - venue_analysis: Performance at specific venues
    
    Examples:
    - Team stats: {"team_id": "101", "analysis_type": "statistics", "additional_params": {"format_type": "T20"}}
    - Recent form: {"team_id": "101", "analysis_type": "form_analysis", "additional_params": {"num_matches": 10}}
    """
    args_schema: Type[BaseModel] = TeamAnalysisInput
    
    def _run(
        self,
        team_id: Union[str, int],
        analysis_type: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute team analysis."""
        try:
            # Map analysis type to function
            function_map = {
                "statistics": team_performance.get_team_statistics,
                "form_analysis": team_performance.get_team_form_analysis,
                "win_loss_record": team_performance.get_win_loss_record,
                "batting_analysis": team_performance.get_team_batting_analysis,
                "bowling_analysis": team_performance.get_team_bowling_analysis,
                "venue_analysis": team_performance.get_team_venue_analysis,
                "tournament_performance": team_performance.get_team_tournament_performance
            }
            
            if analysis_type not in function_map:
                return json.dumps({"error": f"Unknown analysis type: {analysis_type}"})
            
            # Prepare parameters
            params = {"team_id": team_id}
            if additional_params:
                params.update(additional_params)
            
            # Execute function (placeholder)
            result = {
                "analysis_type": analysis_type,
                "team_id": team_id,
                "status": "placeholder_implementation",
                "message": "This would return actual team analysis results",
                "parameters": params
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Team analysis failed: {e}")
            return json.dumps({"error": str(e), "analysis_type": analysis_type})


def get_available_tools(
    mcp_uri: Optional[str] = None,
    include_analytics: bool = True
) -> List[BaseTool]:
    """
    Get list of available tools for the cricket agent.
    
    Args:
        mcp_uri: MCP server URI for database access
        include_analytics: Whether to include analytics helper tools
    
    Returns:
        List of initialized tools
    """
    tools = []
    
    # Add MCP tool if URI provided
    if mcp_uri:
        try:
            # Initialize MCP server
            mcp_server = ReadOnlyMongoMCP(mcp_uri)
            # Note: In production, server would be started separately
            mcp_tool = MCPTool(mcp_server)
            tools.append(mcp_tool)
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tool: {e}")
    
    # Add analytics helper tools
    if include_analytics:
        analytics_tools = [
            BattingAnalysisTool(),
            BowlingAnalysisTool(), 
            MatchAnalysisTool(),
            TeamAnalysisTool()
        ]
        tools.extend(analytics_tools)
    
    return tools