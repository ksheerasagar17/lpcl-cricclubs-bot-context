"""
Bowling statistics and analysis functions for cricket data.

This module provides curated helper tools for bowling performance analysis,
with optimized aggregation pipelines for efficient data retrieval.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_bowling_average(
    player_id: Union[str, int],
    format_type: Optional[str] = None,
    min_wickets: int = 5,
    date_range: Optional[Dict[str, str]] = None
) -> Optional[float]:
    """
    Calculate bowling average for a specific player.
    
    Usage: get_bowling_average(player_id='5009003', format='T20', min_wickets=10)
    
    Bowling average is calculated as total runs conceded divided by wickets taken.
    Lower averages indicate better bowling performance.
    
    Args:
        player_id: Unique identifier for the bowler
        format_type: Match format filter ('T20', 'ODI', 'Test', 'T20I')
        min_wickets: Minimum wickets taken for valid average
        date_range: Optional date range filter
    
    Returns:
        Bowling average as float, or None if insufficient data
        
    Example:
        >>> avg = get_bowling_average('5009003', format='T20')
        >>> print(f"Bowling average: {avg:.2f}")
        Bowling average: 24.85
    """
    pass


def get_economy_rate(
    player_id: Union[str, int],
    format_type: Optional[str] = None,
    phase: Optional[str] = None,
    min_overs: float = 5.0
) -> Optional[float]:
    """
    Calculate economy rate for a bowler in specific match phases.
    
    Usage: get_economy_rate(player_id='5009003', format='T20', phase='powerplay', min_overs=10.0)
    
    Economy rate shows runs conceded per over, crucial for limited-overs cricket.
    Different phases like powerplay, middle overs, death overs can be analyzed.
    
    Args:
        player_id: Unique identifier for the bowler
        format_type: Match format filter
        phase: Specific bowling phase ('powerplay', 'middle_overs', 'death_overs')
        min_overs: Minimum overs bowled for valid calculation
    
    Returns:
        Economy rate as float, or None if insufficient data
        
    Example:
        >>> econ = get_economy_rate('5009003', format='T20', phase='death_overs')
        >>> print(f"Death overs economy: {econ:.2f}")
        Death overs economy: 8.95
    """
    pass


def get_bowling_strike_rate(
    player_id: Union[str, int],
    format_type: Optional[str] = None,
    conditions: Optional[str] = None
) -> Optional[float]:
    """
    Calculate bowling strike rate (balls per wicket).
    
    Usage: get_bowling_strike_rate(player_id='5009003', format='ODI', conditions='home')
    
    Strike rate indicates how frequently a bowler takes wickets.
    Lower strike rates indicate more frequent wicket-taking ability.
    
    Args:
        player_id: Unique identifier for the bowler
        format_type: Match format filter
        conditions: Match conditions ('home', 'away', 'day', 'night')
    
    Returns:
        Bowling strike rate as float, or None if insufficient data
        
    Example:
        >>> sr = get_bowling_strike_rate('5009003', format='ODI')
        >>> print(f"Strike rate: {sr:.1f} balls per wicket")
        Strike rate: 28.5 balls per wicket
    """
    pass


def get_bowling_figures_analysis(
    player_id: Union[str, int],
    format_type: Optional[str] = None,
    figure_type: str = 'best'
) -> Dict[str, Any]:
    """
    Analyze bowling figures and performance statistics.
    
    Usage: get_bowling_figures_analysis(player_id='5009003', format='T20', figure_type='best')
    
    Provides detailed analysis of bowling figures including best performances,
    wicket hauls, and consistency metrics.
    
    Args:
        player_id: Unique identifier for the bowler
        format_type: Match format filter
        figure_type: Type of analysis ('best', 'average', 'consistent')
    
    Returns:
        Dictionary with bowling figures analysis
        
    Example:
        >>> figures = get_bowling_figures_analysis('5009003', 'T20')
        >>> print(f"Best figures: {figures['best_figures']}")
        Best figures: 4/15
    """
    pass


def get_bowling_against_batting_style(
    player_id: Union[str, int],
    batting_style: str,
    format_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze bowling performance against specific batting styles.
    
    Usage: get_bowling_against_batting_style(player_id='5009003', batting_style='Left Handed Batter', format='ODI')
    
    Evaluates effectiveness against different batting styles,
    crucial for strategic bowling planning and field placements.
    
    Args:
        player_id: Unique identifier for the bowler
        batting_style: Batting style to analyze against
        format_type: Match format filter
    
    Returns:
        Dictionary with performance against batting style
        
    Example:
        >>> vs_lefties = get_bowling_against_batting_style('5009003', 'Left Handed Batter')
        >>> print(f"vs LHB average: {vs_lefties['average']:.2f}")
        vs LHB average: 22.15
    """
    pass


def get_bowling_spell_analysis(
    player_id: Union[str, int],
    match_id: Union[str, int],
    innings_number: int = 1
) -> List[Dict[str, Any]]:
    """
    Analyze bowling spell progression and over-by-over performance.
    
    Usage: get_bowling_spell_analysis(player_id='5009003', match_id='2301', innings_number=1)
    
    Provides detailed spell analysis including pressure creation,
    wicket-taking patterns, and economy fluctuations.
    
    Args:
        player_id: Unique identifier for the bowler
        match_id: Specific match identifier
        innings_number: Which innings bowled in
    
    Returns:
        List of over-by-over bowling data
        
    Example:
        >>> spell = get_bowling_spell_analysis('5009003', '2301')
        >>> print(f"Wickets in spell: {len([over for over in spell if over['wickets'] > 0])}")
        Wickets in spell: 3
    """
    pass


def compare_bowling_performance(
    player_ids: List[Union[str, int]],
    format_type: Optional[str] = None,
    metrics: List[str] = ['average', 'economy', 'strike_rate']
) -> Dict[str, Dict[str, float]]:
    """
    Compare bowling performance between multiple players.
    
    Usage: compare_bowling_performance(['5009003', '43329'], format='T20', metrics=['economy', 'strike_rate'])
    
    Provides side-by-side comparison for team selection and tactical decisions.
    
    Args:
        player_ids: List of bowler identifiers to compare
        format_type: Match format for comparison
        metrics: List of metrics to compare
    
    Returns:
        Dictionary with bowler comparisons
        
    Example:
        >>> comparison = compare_bowling_performance(['5009003', '43329'])
        >>> print(f"Bowler 1 economy: {comparison['5009003']['economy']:.2f}")
        Bowler 1 economy: 7.45
    """
    pass


def get_bowling_under_pressure(
    player_id: Union[str, int],
    pressure_situations: List[str] = ['death_overs', 'defending_low', 'powerplay'],
    format_type: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze bowling performance under pressure situations.
    
    Usage: get_bowling_under_pressure(player_id='5009003', pressure_situations=['death_overs', 'defending_low'], format='T20')
    
    Evaluates performance in high-pressure scenarios crucial for identifying
    death bowlers and pressure specialists.
    
    Args:
        player_id: Unique identifier for the bowler
        pressure_situations: List of pressure scenarios
        format_type: Match format filter
    
    Returns:
        Dictionary with performance under pressure
        
    Example:
        >>> pressure_stats = get_bowling_under_pressure('5009003', ['death_overs'])
        >>> print(f"Death overs economy: {pressure_stats['death_overs']['economy']:.2f}")
        Death overs economy: 9.25
    """
    pass


def get_bowling_partnership_breaker(
    player_id: Union[str, int],
    partnership_threshold: int = 50,
    format_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze ability to break batting partnerships.
    
    Usage: get_bowling_partnership_breaker(player_id='5009003', partnership_threshold=75, format='ODI')
    
    Identifies bowlers who are effective at breaking dangerous partnerships,
    a crucial skill in limited-overs cricket.
    
    Args:
        player_id: Unique identifier for the bowler
        partnership_threshold: Minimum partnership runs to consider
        format_type: Match format filter
    
    Returns:
        Dictionary with partnership breaking statistics
        
    Example:
        >>> breaker_stats = get_bowling_partnership_breaker('5009003')
        >>> print(f"Partnerships broken: {breaker_stats['partnerships_broken']}")
        Partnerships broken: 12
    """
    pass


def get_bowling_variation_analysis(
    player_id: Union[str, int],
    format_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze bowling variations and their effectiveness.
    
    Usage: get_bowling_variation_analysis(player_id='5009003', format='T20')
    
    Studies different delivery types and their success rates,
    useful for tactical planning and skill development.
    
    Args:
        player_id: Unique identifier for the bowler
        format_type: Match format filter
    
    Returns:
        Dictionary with variation analysis
        
    Example:
        >>> variations = get_bowling_variation_analysis('5009003')
        >>> print(f"Most effective variation: {variations['most_effective']}")
        Most effective variation: yorker
    """
    pass


def get_bowling_conditions_analysis(
    player_id: Union[str, int],
    condition_type: str = 'pitch',
    format_type: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze bowling performance under different conditions.
    
    Usage: get_bowling_conditions_analysis(player_id='5009003', condition_type='weather', format='ODI')
    
    Evaluates performance across different pitch conditions, weather,
    and time of day for strategic team composition.
    
    Args:
        player_id: Unique identifier for the bowler
        condition_type: Type of condition ('pitch', 'weather', 'time')
        format_type: Match format filter
    
    Returns:
        Dictionary with condition-based performance
        
    Example:
        >>> conditions = get_bowling_conditions_analysis('5009003', 'pitch')
        >>> print(f"Best on: {conditions['best_condition']['type']}")
        Best on: spin-friendly
    """
    pass


def get_bowling_workload_analysis(
    player_id: Union[str, int],
    timeframe: str = 'season',
    format_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze bowling workload and fatigue patterns.
    
    Usage: get_bowling_workload_analysis(player_id='5009003', timeframe='month', format='T20')
    
    Monitors bowling workload for injury prevention and performance optimization.
    
    Args:
        player_id: Unique identifier for the bowler
        timeframe: Analysis timeframe ('week', 'month', 'season')
        format_type: Match format filter
    
    Returns:
        Dictionary with workload analysis
        
    Example:
        >>> workload = get_bowling_workload_analysis('5009003')
        >>> print(f"Average overs per match: {workload['avg_overs_per_match']:.1f}")
        Average overs per match: 3.8
    """
    pass


# Utility functions for advanced bowling analytics

def calculate_bowling_consistency(
    economy_rates: List[float],
    strike_rates: List[float]
) -> float:
    """
    Calculate bowling consistency index based on economy and strike rate variance.
    
    Usage: calculate_bowling_consistency([7.2, 8.1, 6.8, 7.9], [24, 18, 30, 22])
    
    Args:
        economy_rates: List of economy rates from different matches
        strike_rates: List of strike rates from different matches
    
    Returns:
        Consistency index (0-1, higher is more consistent)
    """
    if not economy_rates or not strike_rates:
        return 0.0
    
    # Calculate coefficient of variation for both metrics
    def cv(values):
        if not values:
            return 1.0
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return 1.0
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return (variance ** 0.5) / mean_val
    
    econ_cv = cv(economy_rates)
    sr_cv = cv(strike_rates)
    
    # Combined consistency (inverse of average CV)
    avg_cv = (econ_cv + sr_cv) / 2
    return max(0, 1 - avg_cv)


def determine_bowling_role(
    bowling_style: str,
    avg_economy: float,
    avg_strike_rate: float,
    wickets_per_match: float
) -> str:
    """
    Determine optimal bowling role based on performance metrics.
    
    Usage: determine_bowling_role('Right Arm Fast', 7.2, 18.5, 1.8)
    
    Args:
        bowling_style: Style of bowling
        avg_economy: Average economy rate
        avg_strike_rate: Average strike rate (balls per wicket)
        wickets_per_match: Average wickets per match
    
    Returns:
        Suggested bowling role
    """
    if 'Fast' in bowling_style or 'Medium' in bowling_style:
        if avg_economy < 7.0 and avg_strike_rate < 20:
            return "Death Bowler"
        elif avg_economy < 6.5:
            return "Economic Pacer"
        elif wickets_per_match > 1.5:
            return "Strike Bowler"
        else:
            return "Support Pacer"
    
    elif 'Spin' in bowling_style:
        if avg_economy < 6.0:
            return "Economic Spinner"
        elif wickets_per_match > 1.2:
            return "Strike Spinner"
        else:
            return "Containing Spinner"
    
    else:
        return "All-rounder Bowler"


def calculate_bowling_impact(
    wickets: int,
    economy_rate: float,
    match_situation: str,
    overs_bowled: float
) -> float:
    """
    Calculate bowling impact score for a specific performance.
    
    Usage: calculate_bowling_impact(3, 6.5, 'death_overs', 4.0)
    
    Args:
        wickets: Wickets taken
        economy_rate: Economy rate in the spell
        match_situation: Situation when bowling
        overs_bowled: Overs bowled
    
    Returns:
        Impact score (0-100)
    """
    base_score = wickets * 20  # Base points for wickets
    
    # Economy bonus/penalty
    if economy_rate < 6.0:
        econ_bonus = 15
    elif economy_rate < 8.0:
        econ_bonus = 5
    else:
        econ_bonus = -10
    
    # Situation multiplier
    situation_multiplier = {
        'death_overs': 1.5,
        'powerplay': 1.3,
        'middle_overs': 1.0,
        'pressure': 1.4
    }.get(match_situation, 1.0)
    
    # Workload consideration
    workload_bonus = min(overs_bowled * 2, 8)  # Max 8 points for workload
    
    impact_score = (base_score + econ_bonus + workload_bonus) * situation_multiplier
    return min(100, max(0, impact_score))  # Cap between 0-100