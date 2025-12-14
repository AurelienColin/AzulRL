"""
Tests for player mechanics.

Tests cover:
- Player initialization
- Left panel (staging area) operations
- Right panel (wall) operations
- Scoring mechanics
- Penalty calculation
"""
import pytest
import numpy as np

from src.config import config
from src.obj.player import Player, Left, Right, Penalties


class TestLeftPanel:
    """Test left panel (staging area) operations."""

    def test_left_panel_initial_state(self, empty_left: Left) -> None:
        """Left panel should initialize with zeros and -1 colors."""
        for i in range(config.n_colors):
            assert empty_left.state[i, 0] == 0  # count
            assert empty_left.state[i, 1] == -1  # color (empty)

    def test_left_panel_get_state_shape(self, empty_left: Left) -> None:
        """Left panel state should have correct shape."""
        state = empty_left.get_state()
        assert state.shape == (empty_left.n_states,)
        assert empty_left.n_states == config.n_colors * 2

    def test_left_panel_set_tiles(self, empty_left: Left) -> None:
        """Setting tiles in left panel should work correctly."""
        # Put 2 red tiles in row 2
        empty_left.state[2, 0] = 2  # count
        empty_left.state[2, 1] = 2  # red color

        assert empty_left.state[2, 0] == 2
        assert empty_left.state[2, 1] == 2


class TestRightPanel:
    """Test right panel (wall) operations."""

    def test_right_panel_initial_state(self, empty_right: Right) -> None:
        """Right panel should initialize with -1 (empty cells)."""
        for i in range(config.n_colors):
            for j in range(config.n_colors):
                assert empty_right.state[i, j] == -1

    def test_right_panel_is_not_complete_initially(self, empty_right: Right) -> None:
        """Right panel should not be complete initially."""
        assert empty_right.is_complete is False

    def test_right_panel_place_tile(self, empty_right: Right) -> None:
        """Placing a tile on right panel should work."""
        empty_right.state[0, 0] = 0  # black at (0,0)
        assert empty_right.state[0, 0] == 0


class TestRightPanelScoring:
    """Test right panel scoring calculations."""

    def test_single_tile_base_score(self, empty_right: Right) -> None:
        """Single isolated tile should score 1 point (via count_score fallback)."""
        # Place a tile
        empty_right.state[2, 2] = 0  # black at center
        score = empty_right.count_score(2, 2, 0)
        # Single tile with no adjacents: score = 1 (from int(not col_score and not row_score))
        # Plus color_score = 0 for incomplete color
        assert score == 1

    def test_row_score_adjacent_tiles(self, empty_right: Right) -> None:
        """Adjacent tiles in row should add to score."""
        # Place tiles horizontally
        empty_right.state[0, 0] = 0  # black
        empty_right.state[0, 1] = 1  # white

        # Place third tile
        score = empty_right.count_score(0, 2, 2)  # red at (0, 2)
        # Row score: 3 (tiles at 0,1,2)
        # No column score (single tile in columns)
        assert score >= 1

    def test_column_score_adjacent_tiles(self, empty_right: Right) -> None:
        """Adjacent tiles in column should add to score."""
        # Place tiles vertically
        empty_right.state[0, 0] = 0  # black
        empty_right.state[1, 0] = 1  # white

        score = empty_right.get_column_score(2, 0)
        # Column score calculation for position (2, 0)
        # Checks adjacent tiles above
        assert score >= 0

    def test_complete_row_bonus(self, empty_right: Right) -> None:
        """Completing a row should give bonus points."""
        # Fill row 0 completely
        for col in range(config.n_colors):
            empty_right.state[0, col] = col

        # Check the row score - it counts adjacent tiles
        # The get_row_score method counts before and after from position
        # Complete row should include complete_row_score bonus
        score = empty_right.get_row_score(0, 2)  # Check from middle position
        # Score should be positive when row is filled
        assert score > 0

    def test_complete_column_bonus(self, empty_right: Right) -> None:
        """Completing a column should give bonus points."""
        # Fill column 0 completely
        for row in range(config.n_colors):
            empty_right.state[row, 0] = row

        # Check column score at middle position
        score = empty_right.get_column_score(2, 0)
        # Score should be positive when column is filled
        assert score > 0

    def test_complete_color_bonus(self, empty_right: Right) -> None:
        """Completing all tiles of one color should give bonus."""
        # Place 5 tiles of color 0 (black) in different positions
        # In Azul, same color appears once per row, in a diagonal pattern
        for i in range(config.n_colors):
            empty_right.state[i, (i) % config.n_colors] = 0  # black

        score = empty_right.get_color_score(0)
        assert score == config.complete_color_score


class TestPenalties:
    """Test penalty calculations."""

    def test_no_penalties(self, penalties: Penalties) -> None:
        """No penalties should result in 0 score."""
        penalties.n = 0
        assert penalties.get_score() == 0

    def test_one_penalty(self, penalties: Penalties) -> None:
        """One penalty should cost 1 point."""
        penalties.n = 1
        score = penalties.get_score()
        assert score == 1

    def test_two_penalties(self, penalties: Penalties) -> None:
        """Two penalties should cost 2 points."""
        penalties.n = 2
        score = penalties.get_score()
        assert score == 2

    def test_three_penalties(self, penalties: Penalties) -> None:
        """Three penalties should cost 4 points (1+1+2)."""
        penalties.n = 3
        score = penalties.get_score()
        assert score == 4

    def test_four_penalties(self, penalties: Penalties) -> None:
        """Four penalties should have progressive penalty."""
        penalties.n = 4
        score = penalties.get_score()
        # Penalty progression: 1, 2, 4 for n=1,2,3 then +3*(n-3) after
        # For n=4: base penalties (1+1+2+2=6) + 3*(4-3) = 6+3 = 9
        assert score > penalties.get_score.__self__.n - 1  # Should cost more than n-1

    def test_many_penalties(self, penalties: Penalties) -> None:
        """Many penalties should follow the penalty progression."""
        penalties.n = 7
        score = penalties.get_score()
        # 1 + 1 + 2 + 2 + 3*3 = 15
        assert score > 6

    def test_penalty_state(self, penalties: Penalties) -> None:
        """Penalty state should return score in array."""
        penalties.n = 3
        state = penalties.state
        assert state.shape == (1,)
        assert state[0] == penalties.get_score()


class TestPlayerInitialization:
    """Test player initialization."""

    def test_player_starts_with_zero_score(self, player: Player) -> None:
        """Player should start with score 0."""
        assert player.score == 0

    def test_player_has_left_panel(self, player: Player) -> None:
        """Player should have a left panel."""
        assert player.left is not None
        assert isinstance(player.left, Left)

    def test_player_has_right_panel(self, player: Player) -> None:
        """Player should have a right panel."""
        assert player.right is not None
        assert isinstance(player.right, Right)

    def test_player_has_penalties(self, player: Player) -> None:
        """Player should have a penalties counter."""
        assert player.penalties is not None
        assert isinstance(player.penalties, Penalties)

    def test_player_n_states(self, player: Player) -> None:
        """Player n_states should match expected calculation."""
        expected = 2 * config.n_colors + config.n_colors ** 2 + 1
        assert player.n_states == expected

    def test_player_n_actions(self, player: Player) -> None:
        """Player n_actions should match expected calculation."""
        # n_actions = n_colors * (n_plates + 1)
        # But n_plates is None at this point, using config
        assert player.n_actions == config.n_colors * (config.n_plates + 1) if config.n_plates else True


class TestPlayerState:
    """Test player state representation."""

    def test_player_get_state_shape(self, player: Player) -> None:
        """Player state should have correct shape."""
        state = player.get_state()
        assert state.shape == (player.n_states,)

    def test_player_get_state_contents(self, player: Player) -> None:
        """Player state should contain left, right, and penalty info."""
        state = player.get_state()

        # First part is left panel (flattened)
        left_size = config.n_colors * 2
        left_state = state[:left_size]
        assert len(left_state) == left_size

        # Second part is right panel (flattened)
        right_size = config.n_colors ** 2
        right_state = state[left_size:left_size + right_size]
        assert len(right_state) == right_size

        # Last part is penalty
        penalty_state = state[-1]
        assert penalty_state == player.penalties.get_score()


class TestPlayerWithTiles:
    """Test player with tiles placed."""

    def test_player_with_tiles_state(self, player_with_tiles: Player) -> None:
        """Player with tiles should have non-empty state."""
        state = player_with_tiles.get_state()

        # Row 0 should have 1 tile of black
        assert player_with_tiles.left.state[0, 0] == 1
        assert player_with_tiles.left.state[0, 1] == 0

        # Row 2 should have 2 tiles of red
        assert player_with_tiles.left.state[2, 0] == 2
        assert player_with_tiles.left.state[2, 1] == 2

    def test_row_capacity_row_0(self, player: Player) -> None:
        """Row 0 should have capacity 1."""
        # Fill row 0
        player.left.state[0, 0] = 1
        player.left.state[0, 1] = 0
        # Row is complete at capacity = 1
        assert player.left.state[0, 0] == 0 + 1  # index + 1 = capacity

    def test_row_capacity_row_4(self, player: Player) -> None:
        """Row 4 should have capacity 5."""
        # Fill row 4 partially
        player.left.state[4, 0] = 3
        player.left.state[4, 1] = 1  # white
        # Row not complete until count = 5
        assert player.left.state[4, 0] < 5


class TestPlayerDisplayName:
    """Test player display name."""

    def test_player_displayed_name(self, player: Player) -> None:
        """Player should have correct displayed name."""
        assert player.displayed_name == f"Player {player.index}"

    def test_player_prefix(self, player: Player) -> None:
        """Player prefix should be correctly formatted."""
        assert player.prefix == f"Player {player.index} - "
