"""
Cricket Analytics package.

Curated helper tools for common cricket analysis and statistics calculations.
"""

__version__ = "0.1.0"

from .batting_stats import *
from .bowling_stats import *
from .match_analysis import *
from .team_performance import *

__all__ = [
    # Batting functions
    "get_batting_average",
    "get_strike_rate", 
    "get_boundary_percentage",
    
    # Bowling functions
    "get_bowling_average",
    "get_economy_rate",
    "get_bowling_strike_rate",
    
    # Match analysis
    "compare_innings",
    "get_match_momentum",
    "analyze_partnerships",
    
    # Team performance
    "get_team_statistics",
    "compare_teams",
    "get_win_loss_record"
]