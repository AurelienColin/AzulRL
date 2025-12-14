"""Tests for baseline agents (RandomAgent, GreedyAgent) and evaluation harness.

These tests verify that baseline agents:
1. Produce valid actions
2. Can complete full games without errors
3. Maintain the expected performance hierarchy (Greedy > Random)
"""
import typing
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.obj.random_agent import RandomAgent
from src.obj.greedy_agent import GreedyAgent
from src.obj.game import Game
from src.obj.player import Player
from src.config import config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def game_4p() -> Game:
    """Create a 4-player game."""
    return Game(n_players=4)


@pytest.fixture
def random_agent() -> RandomAgent:
    """Create a RandomAgent with index 0."""
    return RandomAgent(index=0)


@pytest.fixture
def greedy_agent() -> GreedyAgent:
    """Create a GreedyAgent with index 0."""
    return GreedyAgent(index=0)


@pytest.fixture
def game_state(game_4p: Game) -> np.ndarray:
    """Get initial game state."""
    return game_4p.get_state()


# =============================================================================
# RandomAgent Tests
# =============================================================================

class TestRandomAgent:
    """Tests for RandomAgent."""

    def test_displayed_name(self, random_agent: RandomAgent) -> None:
        """Test agent has correct display name."""
        assert "RandomAgent" in random_agent.displayed_name

    def test_internal_choice_returns_valid_tuple(
        self, game_4p: Game, random_agent: RandomAgent
    ) -> None:
        """Test internal_choice returns 4-tuple."""
        game_4p._players = [random_agent] + [RandomAgent(i) for i in range(1, 4)]
        game_4p.players[0].is_first = True
        state = game_4p.get_state()

        result = random_agent.internal_choice(state)

        assert isinstance(result, tuple)
        assert len(result) == 4
        i_plate, i_color, i_row, n_tiles = result
        assert isinstance(i_plate, (int, np.integer))
        assert isinstance(i_color, (int, np.integer))
        assert isinstance(i_row, (int, np.integer))
        assert isinstance(n_tiles, (int, np.integer))

    def test_internal_choice_selects_valid_plate_color(
        self, game_4p: Game, random_agent: RandomAgent
    ) -> None:
        """Test that selected plate/color has tiles available."""
        game_4p._players = [random_agent] + [RandomAgent(i) for i in range(1, 4)]
        game_4p.players[0].is_first = True
        state = game_4p.get_state()

        for _ in range(10):  # Test multiple times due to randomness
            i_plate, i_color, i_row, n_tiles = random_agent.internal_choice(state)

            # n_tiles should be > 0 for valid plate/color selection
            assert n_tiles > 0, "Selected plate/color should have tiles"

    def test_internal_choice_selects_valid_row(
        self, game_4p: Game, random_agent: RandomAgent
    ) -> None:
        """Test that selected row index is valid."""
        game_4p._players = [random_agent] + [RandomAgent(i) for i in range(1, 4)]
        game_4p.players[0].is_first = True
        state = game_4p.get_state()

        for _ in range(10):
            i_plate, i_color, i_row, n_tiles = random_agent.internal_choice(state)

            assert 0 <= i_row < config.n_colors, "Row index should be in valid range"

    def test_can_play_full_game(self, game_4p: Game) -> None:
        """Test RandomAgent can play a complete game without errors."""
        # Replace all players with RandomAgents
        for i in range(4):
            game_4p.players[i] = RandomAgent(index=i)
        game_4p.players[0].is_first = True

        # Play game
        i_round = 0
        max_rounds = 15
        while not game_4p.has_ended() and i_round < max_rounds:
            i_round += 1
            game_4p.round(printing=False)
            game_4p.end_of_round(printing=False)

        # Should complete without exception
        assert i_round <= max_rounds


# =============================================================================
# GreedyAgent Tests
# =============================================================================

class TestGreedyAgent:
    """Tests for GreedyAgent."""

    def test_displayed_name(self, greedy_agent: GreedyAgent) -> None:
        """Test agent has correct display name."""
        assert "GreedyAgent" in greedy_agent.displayed_name

    def test_internal_choice_returns_valid_tuple(
        self, game_4p: Game, greedy_agent: GreedyAgent
    ) -> None:
        """Test internal_choice returns 4-tuple."""
        game_4p._players = [greedy_agent] + [GreedyAgent(i) for i in range(1, 4)]
        game_4p.players[0].is_first = True
        state = game_4p.get_state()

        result = greedy_agent.internal_choice(state)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_internal_choice_selects_valid_action(
        self, game_4p: Game, greedy_agent: GreedyAgent
    ) -> None:
        """Test that selected action is valid."""
        game_4p._players = [greedy_agent] + [GreedyAgent(i) for i in range(1, 4)]
        game_4p.players[0].is_first = True
        state = game_4p.get_state()

        i_plate, i_color, i_row, n_tiles = greedy_agent.internal_choice(state)

        # Should be valid
        assert 0 <= i_plate <= config.n_plates
        assert 0 <= i_color < config.n_colors
        assert 0 <= i_row < config.n_colors
        assert n_tiles > 0

    def test_evaluate_action_prefers_valid_placements(
        self, greedy_agent: GreedyAgent
    ) -> None:
        """Test action evaluation prefers valid placements over penalties."""
        # Setup: empty left panel
        greedy_agent.left.state[:, 0] = 0  # No tiles
        greedy_agent.left.state[:, 1] = -1  # No colors
        greedy_agent.right.state[:] = -1  # Empty right panel

        # Evaluate action placing 2 tiles in row 1 (capacity 2)
        score_good = greedy_agent._evaluate_action(0, 0, 1, 2, None)

        # Evaluate action placing 2 tiles in row 0 (capacity 1 -> 1 penalty)
        score_bad = greedy_agent._evaluate_action(0, 0, 0, 2, None)

        assert score_good > score_bad, "Should prefer actions without penalties"

    def test_can_play_full_game(self, game_4p: Game) -> None:
        """Test GreedyAgent can play a complete game without errors."""
        # Replace all players with GreedyAgents
        for i in range(4):
            game_4p.players[i] = GreedyAgent(index=i)
        game_4p.players[0].is_first = True

        # Play game
        i_round = 0
        max_rounds = 15
        while not game_4p.has_ended() and i_round < max_rounds:
            i_round += 1
            game_4p.round(printing=False)
            game_4p.end_of_round(printing=False)

        # Should complete without exception
        assert i_round <= max_rounds


# =============================================================================
# Performance Hierarchy Tests
# =============================================================================

class TestPerformanceHierarchy:
    """Tests verifying Greedy > Random performance hierarchy."""

    @pytest.mark.parametrize("n_games", [10])
    def test_greedy_beats_random(self, n_games: int) -> None:
        """Test that Greedy agent wins more often than Random against Random opponents."""
        greedy_wins = 0
        random_wins = 0

        for _ in range(n_games):
            # Game with 1 Greedy + 3 Random
            game = Game(n_players=4)
            game.players[0] = GreedyAgent(index=0)
            for i in range(1, 4):
                game.players[i] = RandomAgent(index=i)
            game.players[0].is_first = True

            # Play game
            i_round = 0
            while not game.has_ended() and i_round < 15:
                i_round += 1
                game.round(printing=False)
                game.end_of_round(printing=False)

            scores = [p.score for p in game.players]
            if scores[0] == max(scores):
                greedy_wins += 1
            else:
                random_wins += 1

        # Greedy should win most games against random opponents
        greedy_win_rate = greedy_wins / n_games
        assert greedy_win_rate > 0.5, f"Greedy win rate {greedy_win_rate:.1%} should be > 50%"


# =============================================================================
# Evaluation Harness Tests
# =============================================================================

class TestEvaluationHarness:
    """Tests for the evaluation harness functions."""

    def test_create_agent_random(self) -> None:
        """Test create_agent creates RandomAgent."""
        from src.scripts.evaluate import create_agent

        agent = create_agent("random", 0)
        assert isinstance(agent, RandomAgent)
        assert agent.index == 0

    def test_create_agent_greedy(self) -> None:
        """Test create_agent creates GreedyAgent."""
        from src.scripts.evaluate import create_agent

        agent = create_agent("greedy", 0)
        assert isinstance(agent, GreedyAgent)
        assert agent.index == 0

    def test_create_agent_human(self) -> None:
        """Test create_agent creates Human Player."""
        from src.scripts.evaluate import create_agent

        agent = create_agent("human", 0)
        assert isinstance(agent, Player)
        assert agent.index == 0

    def test_create_agent_invalid(self) -> None:
        """Test create_agent raises error for unknown type."""
        from src.scripts.evaluate import create_agent

        with pytest.raises(ValueError):
            create_agent("unknown", 0)

    def test_run_game_returns_scores(self) -> None:
        """Test run_game returns score dictionary."""
        from src.scripts.evaluate import run_game

        players = [RandomAgent(i) for i in range(4)]
        scores = run_game(players, printing=False)

        assert isinstance(scores, dict)
        assert len(scores) == 4
        for i in range(4):
            assert i in scores
            assert isinstance(scores[i], (int, float))

    def test_evaluate_agents_returns_result(self) -> None:
        """Test evaluate_agents returns EvaluationResult."""
        from src.scripts.evaluate import evaluate_agents, EvaluationResult

        # Game requires 3 or 4 players
        result = evaluate_agents(["random", "random", "random"], n_games=5)

        assert isinstance(result, EvaluationResult)
        assert result.n_games == 5
        assert len(result.agent_names) == 3
        assert sum(result.win_rates.values()) == pytest.approx(1.0, abs=0.1)

    def test_evaluation_result_to_dict(self) -> None:
        """Test EvaluationResult can be serialized to dict."""
        from src.scripts.evaluate import evaluate_agents

        # Game requires 3 or 4 players
        result = evaluate_agents(["random", "random", "random"], n_games=3)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "n_games" in result_dict
        assert "win_rates" in result_dict
        assert "avg_scores" in result_dict


# =============================================================================
# Integration with run_rl.py Tests
# =============================================================================

class TestTrainingIntegration:
    """Tests for baseline evaluation integration in training loop."""

    def test_evaluate_vs_baseline_imports(self) -> None:
        """Test evaluate_vs_baseline can be imported from run_rl."""
        from src.scripts.run_rl import evaluate_vs_baseline
        assert callable(evaluate_vs_baseline)

    def test_training_metrics_has_baseline_fields(self) -> None:
        """Test TrainingMetrics has baseline win rate fields."""
        from src.scripts.run_rl import TrainingMetrics

        metrics = TrainingMetrics()
        assert hasattr(metrics, 'win_rate_vs_random')
        assert hasattr(metrics, 'win_rate_vs_greedy')

    def test_record_episode_accepts_baseline_rates(self) -> None:
        """Test record_episode can store baseline win rates."""
        from src.scripts.run_rl import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_episode(
            episode_num=1,
            loss_choose=0.1,
            loss_eor=0.2,
            avg_score=-100.0,
            game_length=5,
            n_turns=20,
            exploration_rate=0.5,
            learning_rate=0.001,
            buffer_size_choose=100,
            buffer_size_eor=50,
            win_rate=0.5,
            invalid_row_rate=0.1,
            invalid_col_rate=0.05,
            avg_reward=1.0,
            win_rate_vs_random=0.8,
            win_rate_vs_greedy=0.6
        )

        assert len(metrics.win_rate_vs_random) == 1
        assert len(metrics.win_rate_vs_greedy) == 1
        assert metrics.win_rate_vs_random[0] == 0.8
        assert metrics.win_rate_vs_greedy[0] == 0.6

    def test_record_episode_handles_none_baseline_rates(self) -> None:
        """Test record_episode handles None baseline win rates."""
        from src.scripts.run_rl import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_episode(
            episode_num=1,
            loss_choose=0.1,
            loss_eor=0.2,
            avg_score=-100.0,
            game_length=5,
            n_turns=20,
            exploration_rate=0.5,
            learning_rate=0.001,
            buffer_size_choose=100,
            buffer_size_eor=50,
            # Omit baseline rates (None by default)
        )

        # Should store NaN for missing values
        assert len(metrics.win_rate_vs_random) == 1
        assert len(metrics.win_rate_vs_greedy) == 1
        assert np.isnan(metrics.win_rate_vs_random[0])
        assert np.isnan(metrics.win_rate_vs_greedy[0])
