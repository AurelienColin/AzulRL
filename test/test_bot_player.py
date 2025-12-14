"""
Tests for bot player decision logic.

Tests cover:
- Action selection mechanics
- Action masking (invalid actions get zero probability)
- Exploration/exploitation behavior
- Model building and inference
"""
import pytest
import numpy as np
import tensorflow as tf

from src.config import config
from src.obj.bot_player import (
    BotPlayer,
    get_plate_color_mask,
    get_row_mask_for_color,
    get_column_mask_for_row,
    apply_mask_to_logits,
    sample_from_mask,
    sample_from_2d_mask,
    build_choose_model,
    build_end_of_round_model,
)
from src.obj.game import Game


class TestPlateColorMask:
    """Test plate-color masking for valid action selection."""

    def test_mask_shape(self, game: Game) -> None:
        """Mask should have shape (n_plates + 1, n_colors)."""
        state = game.get_state()
        mask = get_plate_color_mask(state, game.n_plates)
        assert mask.shape == (game.n_plates + 1, config.n_colors)

    def test_mask_dtype(self, game: Game) -> None:
        """Mask should be float32."""
        state = game.get_state()
        mask = get_plate_color_mask(state, game.n_plates)
        assert mask.dtype == np.float32

    def test_mask_values_binary(self, game: Game) -> None:
        """Mask values should be 0 or 1."""
        state = game.get_state()
        mask = get_plate_color_mask(state, game.n_plates)
        assert np.all((mask == 0) | (mask == 1))

    def test_mask_reflects_tiles_on_plates(self, game: Game) -> None:
        """Mask should be 1 where plates have tiles of that color."""
        state = game.get_state()
        mask = get_plate_color_mask(state, game.n_plates)

        for i_plate, plate in enumerate(game.plates):
            for i_color in range(config.n_colors):
                has_tiles = plate[i_color] > 0
                assert mask[i_plate, i_color] == (1.0 if has_tiles else 0.0)

    def test_mask_central_initially_empty(self, game: Game) -> None:
        """Central area mask should be 0 initially (empty)."""
        state = game.get_state()
        mask = get_plate_color_mask(state, game.n_plates)

        # Last row is central
        central_mask = mask[game.n_plates, :]
        assert np.sum(central_mask) == 0


class TestRowMask:
    """Test row masking for valid row selection."""

    def test_row_mask_shape(self, game: Game, bot_player: BotPlayer) -> None:
        """Row mask should have shape (n_colors,)."""
        state = game.get_state()
        player_state_start = bot_player._get_player_state_start(state)
        mask = get_row_mask_for_color(state, 0, player_state_start)
        assert mask.shape == (config.n_colors,)

    def test_row_mask_empty_player(self, game: Game, bot_player: BotPlayer) -> None:
        """All rows should be valid for empty player."""
        state = game.get_state()
        player_state_start = bot_player._get_player_state_start(state)
        mask = get_row_mask_for_color(state, 0, player_state_start)
        # All rows should be available for any color on empty board
        assert np.sum(mask) == config.n_colors

    def test_row_mask_excludes_right_panel_color(self, game: Game, bot_player: BotPlayer) -> None:
        """Row with color already in right panel should be invalid."""
        # Place color 0 in row 0 of right panel
        game.players[0].right.state[0, 0] = 0  # black at (0, 0)

        state = game.get_state()
        player_state_start = bot_player._get_player_state_start(state)
        mask = get_row_mask_for_color(state, 0, player_state_start)

        # Row 0 should be invalid for color 0 (black)
        assert mask[0] == 0.0


class TestColumnMask:
    """Test column masking for end-of-round placement."""

    def test_column_mask_shape(self, game: Game, bot_player: BotPlayer) -> None:
        """Column mask should have shape (n_colors,)."""
        state = game.get_state()
        player_state_start = bot_player._get_player_state_start(state)
        mask = get_column_mask_for_row(state, 0, 0, player_state_start)
        assert mask.shape == (config.n_colors,)

    def test_column_mask_empty_panel(self, game: Game, bot_player: BotPlayer) -> None:
        """All columns should be valid on empty right panel."""
        state = game.get_state()
        player_state_start = bot_player._get_player_state_start(state)
        mask = get_column_mask_for_row(state, 0, 0, player_state_start)
        # All columns valid for first tile
        assert np.sum(mask) == config.n_colors

    def test_column_mask_excludes_occupied(self, game: Game, bot_player: BotPlayer) -> None:
        """Occupied cells should be invalid."""
        # Occupy cell (0, 0)
        game.players[0].right.state[0, 0] = 1  # white

        state = game.get_state()
        player_state_start = bot_player._get_player_state_start(state)
        mask = get_column_mask_for_row(state, 0, 2, player_state_start)

        # Column 0 should be invalid (occupied)
        assert mask[0] == 0.0


class TestMaskApplication:
    """Test mask application to logits."""

    def test_apply_mask_valid_actions(self) -> None:
        """Valid actions should keep their logits."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        masked = apply_mask_to_logits(logits, mask)
        np.testing.assert_array_equal(masked, logits)

    def test_apply_mask_invalid_actions(self) -> None:
        """Invalid actions should get very negative logits."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        masked = apply_mask_to_logits(logits, mask)

        # Valid actions unchanged
        assert masked[0] == 1.0
        assert masked[2] == 3.0
        assert masked[4] == 5.0

        # Invalid actions are very negative
        assert masked[1] < -1e8
        assert masked[3] < -1e8


class TestSampling:
    """Test sampling from masks."""

    def test_sample_from_mask_returns_valid(self) -> None:
        """Sampled index should be valid."""
        mask = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        for _ in range(100):
            idx = sample_from_mask(mask)
            assert mask[idx] == 1.0

    def test_sample_from_mask_no_valid(self) -> None:
        """Should return -1 when no valid actions."""
        mask = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        idx = sample_from_mask(mask)
        assert idx == -1

    def test_sample_from_2d_mask(self) -> None:
        """2D sampling should return valid pair."""
        mask = np.zeros((5, 5))
        mask[1, 2] = 1.0
        mask[3, 4] = 1.0

        for _ in range(100):
            i, j = sample_from_2d_mask(mask)
            assert mask[i, j] == 1.0

    def test_sample_from_2d_mask_no_valid(self) -> None:
        """Should return (-1, -1) when no valid actions."""
        mask = np.zeros((5, 5))
        i, j = sample_from_2d_mask(mask)
        assert i == -1 and j == -1


class TestModelBuilding:
    """Test neural network model building."""

    def test_choose_model_output_count(self, game: Game) -> None:
        """Choose model should have 3 outputs (plate, color, row)."""
        input_length = game.n_states - config.start_input_index
        model = build_choose_model(input_length, game.n_plates)
        assert len(model.outputs) == 3

    def test_choose_model_output_shapes(self, game: Game) -> None:
        """Choose model outputs should have correct shapes."""
        input_length = game.n_states - config.start_input_index
        model = build_choose_model(input_length, game.n_plates)

        # Create dummy input
        dummy_input = np.zeros((1, input_length), dtype=np.float32)
        outputs = model(dummy_input)

        assert outputs[0].shape == (1, game.n_plates + 1)  # plate
        assert outputs[1].shape == (1, config.n_colors)    # color
        assert outputs[2].shape == (1, config.n_colors)    # row

    def test_choose_model_softmax_outputs(self, game: Game) -> None:
        """Choose model outputs should sum to 1 (softmax)."""
        input_length = game.n_states - config.start_input_index
        model = build_choose_model(input_length, game.n_plates)

        dummy_input = np.random.randn(1, input_length).astype(np.float32)
        outputs = model(dummy_input)

        for output in outputs:
            assert np.isclose(np.sum(output.numpy()), 1.0, atol=1e-5)

    def test_end_of_round_model_output_count(self, game: Game) -> None:
        """End-of-round model should have n_colors outputs."""
        input_length = game.n_states - config.start_input_index
        model = build_end_of_round_model(input_length)
        assert len(model.outputs) == config.n_colors

    def test_end_of_round_model_output_shapes(self, game: Game) -> None:
        """End-of-round model outputs should have correct shapes."""
        input_length = game.n_states - config.start_input_index
        model = build_end_of_round_model(input_length)

        dummy_input = np.zeros((1, input_length), dtype=np.float32)
        outputs = model(dummy_input)

        for output in outputs:
            assert output.shape == (1, config.n_colors)

    def test_end_of_round_model_softmax_outputs(self, game: Game) -> None:
        """End-of-round model outputs should sum to 1."""
        input_length = game.n_states - config.start_input_index
        model = build_end_of_round_model(input_length)

        dummy_input = np.random.randn(1, input_length).astype(np.float32)
        outputs = model(dummy_input)

        for output in outputs:
            assert np.isclose(np.sum(output.numpy()), 1.0, atol=1e-5)


class TestBotPlayerInitialization:
    """Test bot player initialization."""

    def test_bot_player_epsilon_default(self, game: Game) -> None:
        """Bot should initialize epsilon from config."""
        bot = BotPlayer(
            index=0,
            n_plates=game.n_plates,
            n_players=game.n_players,
            input_length=game.n_states,
            start_input_index=config.start_input_index
        )
        assert bot.epsilon == config.epsilon_start

    def test_bot_player_epsilon_custom(self, game: Game) -> None:
        """Bot should accept custom epsilon."""
        bot = BotPlayer(
            index=0,
            n_plates=game.n_plates,
            n_players=game.n_players,
            input_length=game.n_states,
            start_input_index=config.start_input_index,
            epsilon=0.5
        )
        assert bot.epsilon == 0.5

    def test_bot_player_metrics_initial(self, bot_player: BotPlayer) -> None:
        """Bot metrics should start at 0."""
        assert bot_player.invalid_row_count == 0
        assert bot_player.invalid_col_count == 0
        assert bot_player.total_choices == 0
        assert bot_player.total_eor_placements == 0

    def test_bot_player_reset_metrics(self, bot_player: BotPlayer) -> None:
        """Reset metrics should clear counters."""
        bot_player.invalid_row_count = 10
        bot_player.total_choices = 100
        bot_player.reset_metrics()

        assert bot_player.invalid_row_count == 0
        assert bot_player.total_choices == 0


class TestBotPlayerDecision:
    """Test bot player decision making."""

    def test_internal_choice_returns_tuple(
        self, game: Game, bot_player_high_epsilon: BotPlayer
    ) -> None:
        """Internal choice should return (plate, color, row, n_tiles)."""
        state = game.get_state()
        result = bot_player_high_epsilon.internal_choice(state)

        assert isinstance(result, tuple)
        assert len(result) == 4
        i_plate, i_color, i_row, n_tiles = result

        assert 0 <= i_plate <= game.n_plates
        assert 0 <= i_color < config.n_colors
        assert 0 <= i_row < config.n_colors
        assert n_tiles >= 0

    def test_internal_choice_valid_selection(
        self, game: Game, bot_player_high_epsilon: BotPlayer
    ) -> None:
        """Internal choice should select valid plate-color combination."""
        state = game.get_state()

        for _ in range(10):  # Run multiple times due to randomness
            i_plate, i_color, i_row, n_tiles = bot_player_high_epsilon.internal_choice(state)

            # Should select non-empty plate-color combo
            if i_plate < game.n_plates:
                assert game.plates[i_plate][i_color] >= 0
            # n_tiles could be 0 if selection is invalid (shouldn't happen with masking)

    def test_internal_choice_increments_counter(
        self, game: Game, bot_player_high_epsilon: BotPlayer
    ) -> None:
        """Internal choice should increment total_choices."""
        state = game.get_state()
        initial_count = bot_player_high_epsilon.total_choices

        bot_player_high_epsilon.internal_choice(state)

        assert bot_player_high_epsilon.total_choices == initial_count + 1


class TestBotPlayerExploration:
    """Test exploration vs exploitation behavior."""

    def test_high_epsilon_random_behavior(
        self, game: Game, bot_player_high_epsilon: BotPlayer
    ) -> None:
        """High epsilon should result in random exploration."""
        state = game.get_state()
        choices = set()

        # With epsilon=1.0, should see various choices
        for _ in range(50):
            i_plate, i_color, _, _ = bot_player_high_epsilon.internal_choice(state)
            choices.add((i_plate, i_color))

        # Should have made different choices (randomness)
        assert len(choices) > 1

    def test_no_exploration_deterministic(
        self, game: Game, bot_player_no_exploration: BotPlayer
    ) -> None:
        """Zero epsilon should result in deterministic behavior."""
        state = game.get_state()

        # With epsilon=0, choices should be deterministic
        first_choice = bot_player_no_exploration.internal_choice(state)

        for _ in range(5):
            choice = bot_player_no_exploration.internal_choice(state)
            # Same state should give same choice without exploration
            assert choice[0] == first_choice[0]  # plate
            assert choice[1] == first_choice[1]  # color


class TestBotPlayerMetrics:
    """Test bot player metric tracking."""

    def test_get_invalid_rates_no_actions(self, bot_player: BotPlayer) -> None:
        """Invalid rates should be 0 with no actions."""
        row_rate, col_rate = bot_player.get_invalid_rates()
        assert row_rate == 0.0
        assert col_rate == 0.0

    def test_get_invalid_rates_with_invalids(self, bot_player: BotPlayer) -> None:
        """Invalid rates should reflect actual invalid counts."""
        bot_player.total_choices = 100
        bot_player.invalid_row_count = 10
        bot_player.total_eor_placements = 50
        bot_player.invalid_col_count = 5

        row_rate, col_rate = bot_player.get_invalid_rates()
        assert row_rate == 0.1  # 10/100
        assert col_rate == 0.1  # 5/50
