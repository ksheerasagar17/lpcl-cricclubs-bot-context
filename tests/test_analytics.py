"""
Test suite for analytics helper functions.

Tests batting, bowling, match, and team analysis functions for correctness and edge cases.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from analytics import (
    batting_stats, bowling_stats, match_analysis, team_performance
)


class TestBattingStatsUtilities:
    """Test utility functions in batting_stats module."""
    
    def test_calculate_consistency_index_normal_scores(self):
        """Test consistency index calculation with normal scores."""
        scores = [45, 67, 23, 89, 12, 78, 34, 56]
        consistency = batting_stats.calculate_consistency_index(scores)
        
        assert 0 <= consistency <= 1
        assert isinstance(consistency, float)
    
    def test_calculate_consistency_index_consistent_scores(self):
        """Test consistency index with very consistent scores."""
        consistent_scores = [50, 51, 49, 50, 52, 48, 50]
        consistency = batting_stats.calculate_consistency_index(consistent_scores)
        
        # Should be high consistency
        assert consistency > 0.8
    
    def test_calculate_consistency_index_inconsistent_scores(self):
        """Test consistency index with very inconsistent scores."""
        inconsistent_scores = [5, 150, 0, 89, 3, 120, 1]
        consistency = batting_stats.calculate_consistency_index(inconsistent_scores)
        
        # Should be low consistency
        assert consistency < 0.5
    
    def test_calculate_consistency_index_edge_cases(self):
        """Test consistency index edge cases."""
        # Empty list
        assert batting_stats.calculate_consistency_index([]) == 0.0
        
        # Single score
        assert batting_stats.calculate_consistency_index([50]) == 0.0
        
        # Two scores
        assert batting_stats.calculate_consistency_index([50, 60]) == 0.0
        
        # All zeros
        assert batting_stats.calculate_consistency_index([0, 0, 0, 0]) == 0.0
    
    def test_determine_batting_role_opener(self):
        """Test batting role determination for openers."""
        # Aggressive opener
        role = batting_stats.determine_batting_role(
            avg_strike_rate=140.5,
            avg_position=1.5,
            boundary_percentage=65.0
        )
        assert role == "Aggressive Opener"
        
        # Anchor opener
        role = batting_stats.determine_batting_role(
            avg_strike_rate=115.2,
            avg_position=2.0,
            boundary_percentage=45.0
        )
        assert role == "Anchor Opener"
    
    def test_determine_batting_role_middle_order(self):
        """Test batting role determination for middle order."""
        # Power hitter
        role = batting_stats.determine_batting_role(
            avg_strike_rate=135.0,
            avg_position=3.5,
            boundary_percentage=70.0
        )
        assert role == "Power Hitter"
        
        # Anchor batsman
        role = batting_stats.determine_batting_role(
            avg_strike_rate=105.0,
            avg_position=4.0,
            boundary_percentage=40.0
        )
        assert role == "Anchor Batsman"
    
    def test_determine_batting_role_finisher(self):
        """Test batting role determination for finishers."""
        # Finisher
        role = batting_stats.determine_batting_role(
            avg_strike_rate=155.0,
            avg_position=5.5,
            boundary_percentage=75.0
        )
        assert role == "Finisher"
        
        # Middle order stabilizer
        role = batting_stats.determine_batting_role(
            avg_strike_rate=120.0,
            avg_position=6.0,
            boundary_percentage=50.0
        )
        assert role == "Middle Order Stabilizer"
    
    def test_determine_batting_role_lower_order(self):
        """Test batting role determination for lower order."""
        role = batting_stats.determine_batting_role(
            avg_strike_rate=130.0,
            avg_position=8.0,
            boundary_percentage=60.0
        )
        assert role == "Lower Order Hitter"


class TestBowlingStatsUtilities:
    """Test utility functions in bowling_stats module."""
    
    def test_calculate_bowling_consistency_good_performance(self):
        """Test bowling consistency with good, consistent performance."""
        economy_rates = [7.2, 7.1, 7.3, 6.9, 7.0]
        strike_rates = [18, 19, 17, 20, 18]
        
        consistency = bowling_stats.calculate_bowling_consistency(
            economy_rates, strike_rates
        )
        
        assert 0 <= consistency <= 1
        assert consistency > 0.8  # Should be highly consistent
    
    def test_calculate_bowling_consistency_poor_performance(self):
        """Test bowling consistency with inconsistent performance."""
        economy_rates = [5.5, 12.0, 6.8, 15.2, 7.1]
        strike_rates = [12, 36, 18, 42, 15]
        
        consistency = bowling_stats.calculate_bowling_consistency(
            economy_rates, strike_rates
        )
        
        assert 0 <= consistency <= 1
        assert consistency < 0.5  # Should be low consistency
    
    def test_calculate_bowling_consistency_edge_cases(self):
        """Test bowling consistency edge cases."""
        # Empty lists
        assert bowling_stats.calculate_bowling_consistency([], []) == 0.0
        
        # Single values
        assert bowling_stats.calculate_bowling_consistency([7.0], [18]) == 0.0
    
    def test_determine_bowling_role_fast_bowler(self):
        """Test bowling role determination for fast bowlers."""
        # Death bowler
        role = bowling_stats.determine_bowling_role(
            bowling_style="Right Arm Fast",
            avg_economy=6.8,
            avg_strike_rate=18.5,
            wickets_per_match=1.8
        )
        assert role == "Death Bowler"
        
        # Economic pacer
        role = bowling_stats.determine_bowling_role(
            bowling_style="Right Arm Medium",
            avg_economy=6.2,
            avg_strike_rate=22.0,
            wickets_per_match=1.2
        )
        assert role == "Economic Pacer"
        
        # Strike bowler
        role = bowling_stats.determine_bowling_role(
            bowling_style="Left Arm Fast",
            avg_economy=8.5,
            avg_strike_rate=16.0,
            wickets_per_match=2.1
        )
        assert role == "Strike Bowler"
    
    def test_determine_bowling_role_spinner(self):
        """Test bowling role determination for spinners."""
        # Economic spinner
        role = bowling_stats.determine_bowling_role(
            bowling_style="Right Arm Spin",
            avg_economy=5.8,
            avg_strike_rate=25.0,
            wickets_per_match=1.0
        )
        assert role == "Economic Spinner"
        
        # Strike spinner
        role = bowling_stats.determine_bowling_role(
            bowling_style="Left Arm Spin",
            avg_economy=7.2,
            avg_strike_rate=20.0,
            wickets_per_match=1.5
        )
        assert role == "Strike Spinner"
    
    def test_calculate_bowling_impact_excellent_performance(self):
        """Test bowling impact calculation for excellent performance."""
        impact = bowling_stats.calculate_bowling_impact(
            wickets=3,
            economy_rate=5.5,
            match_situation="death_overs",
            overs_bowled=4.0
        )
        
        assert 0 <= impact <= 100
        assert impact > 80  # Should be high impact
    
    def test_calculate_bowling_impact_poor_performance(self):
        """Test bowling impact calculation for poor performance."""
        impact = bowling_stats.calculate_bowling_impact(
            wickets=0,
            economy_rate=12.0,
            match_situation="middle_overs",
            overs_bowled=2.0
        )
        
        assert 0 <= impact <= 100
        assert impact < 30  # Should be low impact
    
    def test_calculate_bowling_impact_different_situations(self):
        """Test bowling impact with different match situations."""
        base_params = {
            "wickets": 2,
            "economy_rate": 7.0,
            "overs_bowled": 3.0
        }
        
        # Death overs should have higher impact
        death_impact = bowling_stats.calculate_bowling_impact(
            **base_params, match_situation="death_overs"
        )
        
        # Middle overs should have lower impact
        middle_impact = bowling_stats.calculate_bowling_impact(
            **base_params, match_situation="middle_overs"
        )
        
        assert death_impact > middle_impact


class TestMatchAnalysisUtilities:
    """Test utility functions in match_analysis module."""
    
    def test_calculate_win_probability_easy_chase(self):
        """Test win probability for easy chase scenario."""
        prob = match_analysis.calculate_win_probability(
            current_score=150,
            target_score=160,
            wickets_lost=2,
            overs_remaining=5.0,
            format_type="T20"
        )
        
        assert 0 <= prob <= 1
        assert prob > 0.7  # Should be high probability
    
    def test_calculate_win_probability_difficult_chase(self):
        """Test win probability for difficult chase scenario."""
        prob = match_analysis.calculate_win_probability(
            current_score=80,
            target_score=160,
            wickets_lost=7,
            overs_remaining=4.0,
            format_type="T20"
        )
        
        assert 0 <= prob <= 1
        assert prob < 0.3  # Should be low probability
    
    def test_calculate_win_probability_edge_cases(self):
        """Test win probability edge cases."""
        # Already achieved target
        prob = match_analysis.calculate_win_probability(
            current_score=161,
            target_score=160,
            wickets_lost=3,
            overs_remaining=2.0,
            format_type="T20"
        )
        assert prob == 1.0
        
        # No overs remaining, target not achieved
        prob = match_analysis.calculate_win_probability(
            current_score=150,
            target_score=160,
            wickets_lost=5,
            overs_remaining=0.0,
            format_type="T20"
        )
        assert prob == 0.0
        
        # All wickets lost
        prob = match_analysis.calculate_win_probability(
            current_score=100,
            target_score=160,
            wickets_lost=10,
            overs_remaining=5.0,
            format_type="T20"
        )
        assert prob == 0.0
    
    def test_identify_match_phase_t20(self):
        """Test match phase identification for T20."""
        assert match_analysis.identify_match_phase(3, 20, "T20") == "powerplay"
        assert match_analysis.identify_match_phase(8, 20, "T20") == "middle_overs"
        assert match_analysis.identify_match_phase(18, 20, "T20") == "death_overs"
    
    def test_identify_match_phase_odi(self):
        """Test match phase identification for ODI."""
        assert match_analysis.identify_match_phase(5, 50, "ODI") == "powerplay"
        assert match_analysis.identify_match_phase(25, 50, "ODI") == "middle_overs"
        assert match_analysis.identify_match_phase(45, 50, "ODI") == "death_overs"
    
    def test_identify_match_phase_test(self):
        """Test match phase identification for Test cricket."""
        assert match_analysis.identify_match_phase(10, 90, "Test") == "early_session"
        assert match_analysis.identify_match_phase(35, 90, "Test") == "middle_session"
        assert match_analysis.identify_match_phase(70, 90, "Test") == "late_session"
    
    def test_calculate_momentum_score_positive(self):
        """Test momentum calculation for positive momentum."""
        recent_overs = [
            {"runs": 12, "wickets": 0, "boundaries": 2, "sixes": 1, "dot_balls": 1},
            {"runs": 8, "wickets": 0, "boundaries": 1, "sixes": 0, "dot_balls": 3},
            {"runs": 15, "wickets": 0, "boundaries": 1, "sixes": 2, "dot_balls": 0}
        ]
        
        momentum = match_analysis.calculate_momentum_score(recent_overs)
        
        assert -1 <= momentum <= 1
        assert momentum > 0  # Should be positive momentum
    
    def test_calculate_momentum_score_negative(self):
        """Test momentum calculation for negative momentum."""
        recent_overs = [
            {"runs": 2, "wickets": 2, "boundaries": 0, "sixes": 0, "dot_balls": 4},
            {"runs": 1, "wickets": 1, "boundaries": 0, "sixes": 0, "dot_balls": 5},
            {"runs": 3, "wickets": 0, "boundaries": 0, "sixes": 0, "dot_balls": 4}
        ]
        
        momentum = match_analysis.calculate_momentum_score(recent_overs)
        
        assert -1 <= momentum <= 1
        assert momentum < 0  # Should be negative momentum
    
    def test_calculate_momentum_score_edge_cases(self):
        """Test momentum calculation edge cases."""
        # Empty list
        assert match_analysis.calculate_momentum_score([]) == 0.0
        
        # Single over
        momentum = match_analysis.calculate_momentum_score([
            {"runs": 6, "wickets": 0, "boundaries": 1, "sixes": 0, "dot_balls": 3}
        ])
        assert -1 <= momentum <= 1


class TestTeamPerformanceUtilities:
    """Test utility functions in team_performance module."""
    
    def test_calculate_team_strength_excellent_team(self):
        """Test team strength calculation for excellent team."""
        strength = team_performance.calculate_team_strength(
            batting_avg=35.2,
            bowling_avg=24.8,
            win_rate=0.75,
            recent_form=0.8
        )
        
        assert 0 <= strength <= 100
        assert strength > 70  # Should be high strength
    
    def test_calculate_team_strength_poor_team(self):
        """Test team strength calculation for poor team."""
        strength = team_performance.calculate_team_strength(
            batting_avg=22.1,
            bowling_avg=35.5,
            win_rate=0.35,
            recent_form=0.3
        )
        
        assert 0 <= strength <= 100
        assert strength < 50  # Should be low strength
    
    def test_identify_team_strengths_weaknesses_strong_batting(self):
        """Test identification of team with strong batting."""
        team_stats = {
            "batting_average": 32.5,
            "team_strike_rate": 135.0,
            "bowling_average": 28.0,
            "economy_rate": 8.2,
            "win_rate": 0.68
        }
        
        analysis = team_performance.identify_team_strengths_weaknesses(team_stats)
        
        assert "Strong batting lineup" in analysis["strengths"]
        assert "Aggressive batting approach" in analysis["strengths"]
        assert len(analysis["weaknesses"]) >= 0
    
    def test_identify_team_strengths_weaknesses_strong_bowling(self):
        """Test identification of team with strong bowling."""
        team_stats = {
            "batting_average": 25.0,
            "team_strike_rate": 108.0,
            "bowling_average": 22.5,
            "economy_rate": 7.2,
            "win_rate": 0.72
        }
        
        analysis = team_performance.identify_team_strengths_weaknesses(team_stats)
        
        assert "Effective bowling attack" in analysis["strengths"]
        assert "Economic bowling" in analysis["strengths"]
        assert "Consistent winning team" in analysis["strengths"]
    
    def test_identify_team_strengths_weaknesses_poor_team(self):
        """Test identification of team with multiple weaknesses."""
        team_stats = {
            "batting_average": 20.5,
            "team_strike_rate": 105.0,
            "bowling_average": 38.0,
            "economy_rate": 9.5,
            "win_rate": 0.32
        }
        
        analysis = team_performance.identify_team_strengths_weaknesses(team_stats)
        
        assert "Inconsistent batting" in analysis["weaknesses"]
        assert "Slow scoring rate" in analysis["weaknesses"]
        assert "Expensive bowling" in analysis["weaknesses"]
        assert "High economy rate" in analysis["weaknesses"]
        assert "Poor win record" in analysis["weaknesses"]
    
    def test_calculate_head_to_head_advantage_strong_advantage(self):
        """Test head-to-head advantage calculation with strong advantage."""
        team1_stats = {
            "batting_average": 35.0,
            "bowling_average": 25.0,
            "win_rate": 0.7,
            "recent_form": 0.8
        }
        
        team2_stats = {
            "batting_average": 28.0,
            "bowling_average": 32.0,
            "win_rate": 0.5,
            "recent_form": 0.4
        }
        
        h2h_record = {"team1_wins": 8, "team2_wins": 3}
        
        advantage = team_performance.calculate_head_to_head_advantage(
            team1_stats, team2_stats, h2h_record
        )
        
        assert advantage["advantage"] == "team1"
        assert advantage["confidence"] > 0.5
        assert advantage["h2h_win_rate"] > 0.7
    
    def test_calculate_head_to_head_advantage_neutral(self):
        """Test head-to-head advantage calculation with neutral situation."""
        team1_stats = {
            "batting_average": 30.0,
            "bowling_average": 28.0,
            "win_rate": 0.6,
            "recent_form": 0.5
        }
        
        team2_stats = {
            "batting_average": 29.0,
            "bowling_average": 29.0,
            "win_rate": 0.58,
            "recent_form": 0.6
        }
        
        h2h_record = {"team1_wins": 5, "team2_wins": 5}
        
        advantage = team_performance.calculate_head_to_head_advantage(
            team1_stats, team2_stats, h2h_record
        )
        
        assert advantage["advantage"] == "neutral"
        assert advantage["h2h_win_rate"] == 0.5
    
    def test_calculate_head_to_head_advantage_no_history(self):
        """Test head-to-head advantage with no historical matches."""
        team1_stats = {"batting_average": 30.0, "bowling_average": 28.0, "win_rate": 0.6, "recent_form": 0.5}
        team2_stats = {"batting_average": 29.0, "bowling_average": 29.0, "win_rate": 0.58, "recent_form": 0.6}
        h2h_record = {"team1_wins": 0, "team2_wins": 0}
        
        advantage = team_performance.calculate_head_to_head_advantage(
            team1_stats, team2_stats, h2h_record
        )
        
        assert advantage["advantage"] == "neutral"
        assert advantage["confidence"] == 0.0
        assert advantage["total_h2h_matches"] == 0


class TestAnalyticsPlaceholderImplementations:
    """Test that analytics functions have proper placeholder structures."""
    
    def test_batting_stats_functions_exist(self):
        """Test that all expected batting analysis functions exist."""
        expected_functions = [
            "get_batting_average",
            "get_strike_rate", 
            "get_boundary_percentage",
            "get_innings_progression",
            "compare_batting_performance",
            "get_batting_against_bowling_style",
            "get_batting_milestones",
            "get_batting_under_pressure",
            "get_batting_partnership_analysis",
            "get_batting_venue_analysis"
        ]
        
        for func_name in expected_functions:
            assert hasattr(batting_stats, func_name)
            func = getattr(batting_stats, func_name)
            assert callable(func)
    
    def test_bowling_stats_functions_exist(self):
        """Test that all expected bowling analysis functions exist."""
        expected_functions = [
            "get_bowling_average",
            "get_economy_rate",
            "get_bowling_strike_rate",
            "get_bowling_figures_analysis",
            "get_bowling_against_batting_style",
            "get_bowling_spell_analysis",
            "compare_bowling_performance",
            "get_bowling_under_pressure",
            "get_bowling_partnership_breaker",
            "get_bowling_variation_analysis",
            "get_bowling_conditions_analysis",
            "get_bowling_workload_analysis"
        ]
        
        for func_name in expected_functions:
            assert hasattr(bowling_stats, func_name)
            func = getattr(bowling_stats, func_name)
            assert callable(func)
    
    def test_match_analysis_functions_exist(self):
        """Test that all expected match analysis functions exist."""
        expected_functions = [
            "compare_innings",
            "get_match_momentum",
            "analyze_partnerships",
            "get_powerplay_analysis",
            "get_death_overs_analysis",
            "get_match_turning_points",
            "compare_team_strategies",
            "get_individual_impact_analysis",
            "get_match_context_analysis",
            "get_chase_analysis",
            "compare_similar_matches"
        ]
        
        for func_name in expected_functions:
            assert hasattr(match_analysis, func_name)
            func = getattr(match_analysis, func_name)
            assert callable(func)
    
    def test_team_performance_functions_exist(self):
        """Test that all expected team analysis functions exist."""
        expected_functions = [
            "get_team_statistics",
            "compare_teams",
            "get_win_loss_record",
            "get_team_batting_analysis",
            "get_team_bowling_analysis",
            "get_team_form_analysis",
            "get_team_vs_bowling_style",
            "get_team_venue_analysis",
            "get_team_tournament_performance",
            "get_team_player_contributions",
            "get_team_tactical_analysis",
            "compare_team_eras"
        ]
        
        for func_name in expected_functions:
            assert hasattr(team_performance, func_name)
            func = getattr(team_performance, func_name)
            assert callable(func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])