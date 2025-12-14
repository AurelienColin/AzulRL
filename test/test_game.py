"""
Tests for game initialization and mechanics.

Tests cover:
- Bag initialization with correct tile counts
- Plate distribution with 4 tiles each
- Game state creation and structure
- Central area mechanics
"""
import pytest
import numpy as np

from src.config import config
from src.obj.game import Game
from src.obj.bag import Bag
from src.obj.plate import Plate
from src.obj.container import Container
from src.obj.central import Central


class TestBagInitialization:
    """Test bag tile count initialization."""

    def test_bag_has_correct_total_tiles(self, fresh_bag: Bag) -> None:
        """Bag should have n_tile_per_color * n_colors tiles total."""
        expected_total = config.n_tile_per_color * config.n_colors
        assert len(fresh_bag) == expected_total

    def test_bag_has_correct_tiles_per_color(self, fresh_bag: Bag) -> None:
        """Each color should have n_tile_per_color tiles."""
        assert fresh_bag.n_blacks == config.n_tile_per_color
        assert fresh_bag.n_whites == config.n_tile_per_color
        assert fresh_bag.n_reds == config.n_tile_per_color
        assert fresh_bag.n_blues == config.n_tile_per_color
        assert fresh_bag.n_yellows == config.n_tile_per_color

    def test_bag_getitem(self, fresh_bag: Bag) -> None:
        """Test __getitem__ returns correct tile counts."""
        assert fresh_bag[0] == config.n_tile_per_color  # blacks
        assert fresh_bag[1] == config.n_tile_per_color  # whites
        assert fresh_bag[2] == config.n_tile_per_color  # reds
        assert fresh_bag[3] == config.n_tile_per_color  # blues
        assert fresh_bag[4] == config.n_tile_per_color  # yellows


class TestContainerOperations:
    """Test container base class operations."""

    def test_container_len(self, empty_container: Container) -> None:
        """Empty container should have length 0."""
        assert len(empty_container) == 0

    def test_container_len_with_tiles(self, filled_container: Container) -> None:
        """Container length should be sum of all tile counts."""
        expected = 3 + 2 + 1 + 4 + 0  # blacks + whites + reds + blues + yellows
        assert len(filled_container) == expected

    def test_container_setitem(self, empty_container: Container) -> None:
        """Test __setitem__ correctly sets tile counts."""
        empty_container[0] = 5
        empty_container[1] = 3
        empty_container[2] = 2
        empty_container[3] = 4
        empty_container[4] = 1

        assert empty_container.n_blacks == 5
        assert empty_container.n_whites == 3
        assert empty_container.n_reds == 2
        assert empty_container.n_blues == 4
        assert empty_container.n_yellows == 1

    def test_container_empty(self, filled_container: Container) -> None:
        """Test empty() clears all tiles."""
        filled_container.empty()
        assert len(filled_container) == 0


class TestPlateDistribution:
    """Test plate tile distribution."""

    def test_plates_count_for_4_players(self, game_4_players: Game) -> None:
        """4-player game should have 9 plates."""
        assert game_4_players.n_plates == 9
        assert len(game_4_players.plates) == 9

    def test_plates_count_for_3_players(self, game_3_players: Game) -> None:
        """3-player game should have 7 plates."""
        assert game_3_players.n_plates == 7
        assert len(game_3_players.plates) == 7

    def test_each_plate_has_4_tiles(self, game: Game) -> None:
        """Each plate should receive exactly 4 tiles."""
        for plate in game.plates:
            assert len(plate) == config.n_tile_per_plate

    def test_plates_deplete_bag(self, game: Game) -> None:
        """Plates should take tiles from the bag."""
        tiles_on_plates = sum(len(plate) for plate in game.plates)
        expected_remaining = (config.n_tile_per_color * config.n_colors) - tiles_on_plates
        assert len(game.bag) == expected_remaining

    def test_bag_refill_assertion(self, fresh_bag: Bag) -> None:
        """Bag.pick should assert when not enough tiles."""
        # Deplete bag to below n_tile_per_plate
        fresh_bag.n_blacks = 1
        fresh_bag.n_whites = 1
        fresh_bag.n_reds = 0
        fresh_bag.n_blues = 0
        fresh_bag.n_yellows = 0

        with pytest.raises(AssertionError):
            fresh_bag.pick(0)


class TestCentralArea:
    """Test central area mechanics."""

    def test_central_starts_empty(self, empty_central: Central) -> None:
        """Central should start with no tiles."""
        assert len(empty_central) == 0

    def test_central_has_first_player_tile(self, empty_central: Central) -> None:
        """Central should have first player tile by default."""
        assert empty_central.has_first_player_tile is True

    def test_game_central_has_first_player_tile(self, game: Game) -> None:
        """Game's central should have first player tile after initialization."""
        assert game.central.has_first_player_tile is True


class TestGameState:
    """Test game state creation and structure."""

    def test_game_state_shape(self, game: Game) -> None:
        """Game state should have correct length."""
        state = game.get_state()
        assert state.shape == (game.n_states,)

    def test_game_state_dtype(self, game: Game) -> None:
        """Game state should be integer type."""
        state = game.get_state()
        assert state.dtype == np.int64 or state.dtype == np.int32

    def test_game_state_bag_section(self, game: Game) -> None:
        """First n_colors elements should represent bag tile counts."""
        state = game.get_state()
        bag_state = state[:config.n_colors]
        # Bag section should have non-negative integer counts
        for count in bag_state:
            assert count >= 0
            assert count <= config.n_tile_per_color
        # Total tiles in bag should be reasonable (after plate distribution)
        total_in_bag = sum(bag_state)
        total_on_plates = sum(len(plate) for plate in game.plates)
        assert total_in_bag + total_on_plates <= config.n_tile_per_color * config.n_colors

    def test_game_state_graveyard_section(self, game: Game) -> None:
        """Graveyard section should be empty at start."""
        state = game.get_state()
        graveyard_start = config.n_colors
        graveyard_end = 2 * config.n_colors
        graveyard_state = state[graveyard_start:graveyard_end]
        for count in graveyard_state:
            assert count == 0

    def test_game_state_plates_section(self, game: Game) -> None:
        """Plates section should reflect plate contents."""
        state = game.get_state()
        plates_start = 2 * config.n_colors
        for i, plate in enumerate(game.plates):
            plate_start = plates_start + i * config.n_colors
            for j in range(config.n_colors):
                assert state[plate_start + j] == plate[j]

    def test_game_state_central_section(self, game: Game) -> None:
        """Central section should be empty at start."""
        state = game.get_state()
        central_start = (2 + game.n_plates) * config.n_colors
        central_end = central_start + config.n_colors
        central_state = state[central_start:central_end]
        for count in central_state:
            assert count == 0

    def test_game_state_first_player_token(self, game: Game) -> None:
        """First player token should be 1 at start."""
        state = game.get_state()
        first_player_idx = (3 + game.n_plates) * config.n_colors
        assert state[first_player_idx] == 1

    def test_game_state_player_encoding(self, game: Game) -> None:
        """Player encoding should be at end of state."""
        state = game.get_state()
        player_encoding = state[-game.n_players:]
        # Player encoding indicates current active player
        # Note: Implementation uses -player_index, so player 0 encodes at position 0
        # This test just verifies the section is integer values
        assert all(isinstance(v, (int, np.integer)) for v in player_encoding)


class TestGamePlayers:
    """Test game player initialization."""

    def test_correct_player_count(self, game: Game, n_players: int) -> None:
        """Game should have correct number of players."""
        assert len(game.players) == n_players

    def test_player_indices(self, game: Game) -> None:
        """Players should have sequential indices."""
        for i, player in enumerate(game.players):
            assert player.index == i

    def test_one_first_player(self, game: Game) -> None:
        """Exactly one player should be first at start."""
        first_players = [p for p in game.players if p.is_first]
        assert len(first_players) == 1

    def test_first_player_is_zero(self, game: Game) -> None:
        """Player 0 should be first at start."""
        assert game.players[0].is_first is True


class TestGameEndCondition:
    """Test game end condition."""

    def test_game_not_ended_at_start(self, game: Game) -> None:
        """Game should not be ended at start."""
        assert game.has_ended() is False

    def test_game_ends_with_complete_row(self, game: Game) -> None:
        """Game should end when a player completes a row on right panel."""
        # Fill a complete row on player 0's right panel
        for col in range(config.n_colors):
            game.players[0].right.state[0, col] = col  # Different colors per cell
        game.players[0].right.is_complete = True

        assert game.has_ended() is True


class TestPlatePick:
    """Test plate tile picking from bag."""

    def test_pick_returns_plate(self, fresh_bag: Bag) -> None:
        """pick() should return a Plate object."""
        plate = fresh_bag.pick(0)
        assert isinstance(plate, Plate)

    def test_pick_returns_correct_tiles(self, fresh_bag: Bag) -> None:
        """pick() should return plate with exactly n_tile_per_plate tiles."""
        initial_bag_count = len(fresh_bag)
        plate = fresh_bag.pick(0)

        assert len(plate) == config.n_tile_per_plate
        assert len(fresh_bag) == initial_bag_count - config.n_tile_per_plate

    def test_pick_plate_has_index(self, fresh_bag: Bag) -> None:
        """Picked plate should have correct index."""
        plate = fresh_bag.pick(5)
        assert plate.index == 5
