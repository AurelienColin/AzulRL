"""
Tests for state representation.

Tests cover:
- State vector construction
- State normalization
- State encoding/decoding utilities
"""
import pytest
import numpy as np

from src.config import config
from src.utils import normalize_state, to_hot_encoded
from src.obj.game import Game


class TestStateVectorConstruction:
    """Test game state vector construction."""

    def test_state_vector_length(self, game: Game) -> None:
        """State vector should have expected length."""
        state = game.get_state()
        expected_length = game.n_states
        assert len(state) == expected_length

    def test_state_vector_components(self, game: Game) -> None:
        """State vector should contain all expected components."""
        state = game.get_state()
        n_colors = config.n_colors
        n_plates = game.n_plates
        n_players = game.n_players

        # Calculate expected length
        bag_len = n_colors
        graveyard_len = n_colors
        plates_len = n_plates * n_colors
        central_len = n_colors
        first_player_len = 1
        player_state_len = game.players[0].n_states * n_players
        player_encoding_len = n_players

        expected_len = (bag_len + graveyard_len + plates_len + central_len +
                        first_player_len + player_state_len + player_encoding_len)
        assert len(state) == expected_len

    def test_state_dtype(self, game: Game) -> None:
        """State should be integer type."""
        state = game.get_state()
        assert np.issubdtype(state.dtype, np.integer)


class TestStateNormalization:
    """Test state normalization."""

    def test_normalize_returns_float(self, game: Game) -> None:
        """Normalized state should be float32."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)
        assert normalized.dtype == np.float32

    def test_normalize_preserves_shape(self, game: Game) -> None:
        """Normalization should preserve state shape."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)
        assert normalized.shape == state.shape

    def test_normalize_bag_range(self, game: Game) -> None:
        """Bag values should be normalized to [0, 1]."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        bag_section = normalized[:config.n_colors]
        assert np.all(bag_section >= 0)
        assert np.all(bag_section <= 1)

    def test_normalize_plate_range(self, game: Game) -> None:
        """Plate values should be normalized to [0, 1]."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        plate_start = 2 * config.n_colors
        plate_end = (2 + game.n_plates) * config.n_colors
        plate_section = normalized[plate_start:plate_end]

        assert np.all(plate_section >= 0)
        assert np.all(plate_section <= 1)

    def test_normalize_central_range(self, game: Game) -> None:
        """Central values should be normalized to [0, 1]."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        central_start = (2 + game.n_plates) * config.n_colors
        central_end = central_start + config.n_colors
        central_section = normalized[central_start:central_end]

        assert np.all(central_section >= 0)
        assert np.all(central_section <= 1)

    def test_normalize_does_not_modify_original(self, game: Game) -> None:
        """Normalization should not modify original state."""
        state = game.get_state()
        original = state.copy()
        normalize_state(state, config.n_colors, game.n_plates, game.n_players)
        np.testing.assert_array_equal(state, original)

    def test_normalize_first_player_token(self, game: Game) -> None:
        """First player token should remain binary."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        first_player_idx = (3 + game.n_plates) * config.n_colors
        assert normalized[first_player_idx] in [0.0, 1.0]

    def test_normalize_player_encoding(self, game: Game) -> None:
        """Player encoding should remain binary after normalization."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        player_encoding = normalized[-game.n_players:]
        # All values should be 0 or 1 (binary encoding)
        assert all(v in [0.0, 1.0] for v in player_encoding)


class TestStateNormalizationValues:
    """Test specific normalization value ranges."""

    def test_bag_full_normalization(self, game: Game) -> None:
        """Full bag should normalize to 1.0 for each color."""
        # Create state with full bag
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        # Before any tiles are taken, bag should be near full
        # (minus tiles distributed to plates)
        bag_section = normalized[:config.n_colors]
        # At least some values should be close to 1
        assert np.max(bag_section) > 0.5

    def test_empty_graveyard_normalization(self, game: Game) -> None:
        """Empty graveyard should normalize to 0.0."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        graveyard_start = config.n_colors
        graveyard_end = 2 * config.n_colors
        graveyard_section = normalized[graveyard_start:graveyard_end]

        np.testing.assert_array_equal(graveyard_section, np.zeros(config.n_colors))

    def test_plate_normalization(self, game: Game) -> None:
        """Plate values should be normalized to [0, 1] range."""
        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        plate_start = 2 * config.n_colors
        plate_end = (2 + game.n_plates) * config.n_colors

        # Each plate tile count is normalized by n_tile_per_plate (4)
        # Values should be in [0, 1] range
        plate_section = normalized[plate_start:plate_end]
        assert np.all(plate_section >= 0)
        assert np.all(plate_section <= 1)

        # At least some plates should have tiles (non-zero values)
        assert np.sum(plate_section) > 0

    def test_left_panel_color_normalization(self, game: Game) -> None:
        """Left panel colors should be normalized to [0, 1]."""
        # Set a color in player 0's left panel
        game.players[0].left.state[0, 1] = 2  # red

        state = game.get_state()
        normalized = normalize_state(state, config.n_colors, game.n_plates, game.n_players)

        # Color normalization: (color + 1) / n_colors
        # For color 2 (red): (2 + 1) / 5 = 0.6
        # Find the color value in normalized state
        player_state_start = (3 + game.n_plates) * config.n_colors + 1
        left_panel_start = player_state_start
        color_idx = left_panel_start + 1  # First row, color position

        # Use np.isclose for floating point comparison
        assert np.isclose(normalized[color_idx], 0.6, atol=1e-6)


class TestHotEncoding:
    """Test one-hot encoding utility."""

    def test_hot_encoding_shape(self) -> None:
        """Hot encoding should return correct shapes."""
        choices = [
            [(0, 1, 2), (1, 2, 3)],  # Player 0: 2 turns
            [(2, 0, 1)],             # Player 1: 1 turn
        ]

        encoded = to_hot_encoded(choices, 10, 5, 5)

        # Should have 3 subarrays (one for each dimension)
        assert len(encoded) == 3

        # First dimension (plate): max 10
        assert encoded[0][0].shape == (2, 10)  # Player 0: 2 turns
        assert encoded[0][1].shape == (1, 10)  # Player 1: 1 turn

        # Second dimension (color): max 5
        assert encoded[1][0].shape == (2, 5)
        assert encoded[1][1].shape == (1, 5)

    def test_hot_encoding_values(self) -> None:
        """Hot encoding should produce correct one-hot vectors."""
        choices = [
            [(2, 3, 1)],  # Single choice
        ]

        encoded = to_hot_encoded(choices, 10, 5, 5)

        # Check plate encoding (index 2)
        plate_encoding = encoded[0][0][0]  # Player 0, turn 0
        assert plate_encoding[2] == 1
        assert np.sum(plate_encoding) == 1

        # Check color encoding (index 3)
        color_encoding = encoded[1][0][0]
        assert color_encoding[3] == 1
        assert np.sum(color_encoding) == 1

        # Check row encoding (index 1)
        row_encoding = encoded[2][0][0]
        assert row_encoding[1] == 1
        assert np.sum(row_encoding) == 1

    def test_hot_encoding_handles_negative(self) -> None:
        """Hot encoding should handle -1 (no choice) gracefully."""
        choices = [
            [(-1, 2, -1)],
        ]

        encoded = to_hot_encoded(choices, 10, 5, 5)

        # -1 should result in all zeros
        assert np.sum(encoded[0][0][0]) == 0  # No plate selected
        assert np.sum(encoded[2][0][0]) == 0  # No row selected


class TestStateSections:
    """Test state section boundaries."""

    def test_bag_section(self, game: Game) -> None:
        """Bag section should be at start of state with valid counts."""
        state = game.get_state()
        bag_section = state[:config.n_colors]

        # Verify bag section contains valid tile counts
        for count in bag_section:
            assert count >= 0
            assert count <= config.n_tile_per_color

    def test_graveyard_section(self, game: Game) -> None:
        """Graveyard section should follow bag."""
        state = game.get_state()
        start = config.n_colors
        end = 2 * config.n_colors
        graveyard_section = state[start:end]

        for i in range(config.n_colors):
            assert graveyard_section[i] == game.graveyard[i]

    def test_plates_section(self, game: Game) -> None:
        """Plates section should follow graveyard."""
        state = game.get_state()
        start = 2 * config.n_colors

        for i_plate, plate in enumerate(game.plates):
            plate_start = start + i_plate * config.n_colors
            for i_color in range(config.n_colors):
                assert state[plate_start + i_color] == plate[i_color]

    def test_central_section(self, game: Game) -> None:
        """Central section should follow plates."""
        state = game.get_state()
        start = (2 + game.n_plates) * config.n_colors
        central_section = state[start:start + config.n_colors]

        for i in range(config.n_colors):
            assert central_section[i] == game.central[i]


class TestStateConsistency:
    """Test state consistency across operations."""

    def test_state_deterministic(self, game: Game) -> None:
        """Same game should produce same state."""
        state1 = game.get_state()
        state2 = game.get_state()
        np.testing.assert_array_equal(state1, state2)

    def test_normalized_deterministic(self, game: Game) -> None:
        """Same state should produce same normalized state."""
        state = game.get_state()
        norm1 = normalize_state(state, config.n_colors, game.n_plates, game.n_players)
        norm2 = normalize_state(state, config.n_colors, game.n_plates, game.n_players)
        np.testing.assert_array_equal(norm1, norm2)


class TestStatePlayerSection:
    """Test player state section in game state."""

    def test_player_state_size(self, game: Game) -> None:
        """Each player's state section should have correct size."""
        expected_size = game.players[0].n_states
        # 2 * n_colors (left) + n_colors^2 (right) + 1 (penalty)
        assert expected_size == 2 * config.n_colors + config.n_colors ** 2 + 1

    def test_player_state_location(self, game: Game) -> None:
        """Player state should be at expected position in game state."""
        state = game.get_state()

        # Calculate player state start position
        player_state_start = (3 + game.n_plates) * config.n_colors + 1
        player_state_size = game.players[0].n_states

        # Extract player 0's state from game state
        player_0_state = state[player_state_start:player_state_start + player_state_size]

        # Compare with player's own state
        expected_state = game.players[0].get_state()

        np.testing.assert_array_equal(player_0_state, expected_state)
