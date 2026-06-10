import random
import typing
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from src.obj.player import Player
from src.config import config
from src.utils import normalize_state
from functools import cache

from src.lazy_property import LazyProperty
from src.logging_utils import logger
import os


# =============================================================================
# Action Masking Functions
# =============================================================================

def get_plate_color_mask(game_state: np.ndarray, n_plates: int) -> np.ndarray:
    """
    Compute a mask for valid (plate, color) combinations.

    A (plate, color) pair is valid if the plate contains at least one tile of that color.

    Args:
        game_state: The full game state array (non-normalized).
        n_plates: Number of plates (excluding central).

    Returns:
        np.ndarray: Mask of shape (n_plates + 1, n_colors) where 1 = valid, 0 = invalid.
    """
    n_colors = config.n_colors
    mask = np.zeros((n_plates + 1, n_colors), dtype=np.float32)

    # Plate contents start at index 2 * n_colors (after bag and graveyard)
    plate_start_index = 2 * n_colors

    for i_plate in range(n_plates + 1):  # +1 includes the central
        plate_offset = plate_start_index + i_plate * n_colors
        for i_color in range(n_colors):
            tile_count = game_state[plate_offset + i_color]
            if tile_count > 0:
                mask[i_plate, i_color] = 1.0

    return mask


def get_row_mask_for_color(
    game_state: np.ndarray,
    i_color: int,
    player_state_start: int
) -> np.ndarray:
    """
    Compute a mask for valid rows given a specific color.

    A row can accept tiles if:
    - The row is empty (color == -1), OR
    - The row has the same color AND has remaining space

    Additionally, we cannot place tiles of a color that already exists in
    that row of the right panel.

    Args:
        game_state: The full game state array (non-normalized).
        i_color: The color index being placed.
        player_state_start: Index where the current player's state begins.

    Returns:
        np.ndarray: Mask of shape (n_colors,) where 1 = valid row, 0 = invalid.
    """
    n_colors = config.n_colors
    mask = np.zeros(n_colors, dtype=np.float32)

    # Player state layout (from player.py):
    # left.state.flatten(): For row i -> state[i] = [count, color]
    # Flattened: [count0, color0, count1, color1, ..., count4, color4] = 10 values
    # right.state.flatten(): shape (5, 5) = 25 values
    # penalties.state: 1 value

    left_panel_start = player_state_start
    right_panel_start = player_state_start + 2 * n_colors

    for i_row in range(n_colors):
        row_count = game_state[left_panel_start + 2 * i_row]
        row_color = game_state[left_panel_start + 2 * i_row + 1]
        row_capacity = i_row + 1  # Row 0 has capacity 1, row 1 has capacity 2, etc.

        # Check if this color already exists in the right panel for this row
        right_row_start = right_panel_start + i_row * n_colors
        right_row = game_state[right_row_start:right_row_start + n_colors]
        color_in_right_row = (i_color in right_row)

        if color_in_right_row:
            # Cannot place this color in a row where it already exists on the right
            mask[i_row] = 0.0
        elif row_color == -1:
            # Row is empty, can accept any color
            mask[i_row] = 1.0
        elif row_color == i_color and row_count < row_capacity:
            # Row has same color and has space
            mask[i_row] = 1.0
        # else: row has different color or is full -> invalid (mask stays 0)

    return mask


def get_column_mask_for_row(
    game_state: np.ndarray,
    i_row: int,
    color: int,
    player_state_start: int
) -> np.ndarray:
    """
    Compute a mask for valid columns for end-of-round tile placement.

    A column is valid if:
    - The cell is empty (value == -1)
    - The color doesn't already exist in that column
    - The color doesn't already exist in that row (except the cell itself)

    Args:
        game_state: The full game state array (non-normalized).
        i_row: The row index.
        color: The color being placed.
        player_state_start: Index where the current player's state begins.

    Returns:
        np.ndarray: Mask of shape (n_colors,) where 1 = valid column, 0 = invalid.
    """
    n_colors = config.n_colors
    mask = np.zeros(n_colors, dtype=np.float32)

    right_panel_start = player_state_start + 2 * n_colors

    # Extract right panel as 2D array for easier indexing
    right_panel = game_state[right_panel_start:right_panel_start + n_colors * n_colors]
    right_panel = right_panel.reshape((n_colors, n_colors))

    for i_col in range(n_colors):
        cell_value = right_panel[i_row, i_col]

        if cell_value != -1:
            # Cell is already occupied
            continue

        # Check if color exists in this column
        if color in right_panel[:, i_col]:
            continue

        # Check if color exists in this row (excluding current cell which is -1)
        if color in right_panel[i_row, :]:
            continue

        mask[i_col] = 1.0

    return mask


def apply_mask_to_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to logits before softmax: invalid actions get -inf.

    Args:
        logits: Raw model outputs (probabilities or logits).
        mask: Binary mask where 1 = valid, 0 = invalid.

    Returns:
        np.ndarray: Masked logits with invalid actions set to very low value.
    """
    return logits * mask + (1.0 - mask) * (-1e9)


def sample_from_mask(mask: np.ndarray) -> int:
    """
    Sample a random valid index from a mask.

    Args:
        mask: Binary mask where 1 = valid, 0 = invalid.

    Returns:
        int: A randomly selected valid index, or -1 if no valid actions.
    """
    valid_indices = np.where(mask > 0)[0]
    if len(valid_indices) == 0:
        return -1
    return int(np.random.choice(valid_indices))


def sample_from_2d_mask(mask: np.ndarray) -> typing.Tuple[int, int]:
    """
    Sample a random valid (i, j) pair from a 2D mask.

    Args:
        mask: 2D binary mask where 1 = valid, 0 = invalid.

    Returns:
        Tuple[int, int]: A randomly selected valid (i, j) pair, or (-1, -1) if none.
    """
    valid_indices = np.argwhere(mask > 0)
    if len(valid_indices) == 0:
        return (-1, -1)
    idx = np.random.randint(len(valid_indices))
    return int(valid_indices[idx, 0]), int(valid_indices[idx, 1])


# =============================================================================
# Model Building Functions
# =============================================================================

def _build_dense_block(input_tensor: tf.Tensor, name_prefix: str = '') -> tf.Tensor:
    """Build a dense block with configurable layers and dropout.

    Args:
        input_tensor: Input tensor to the block
        name_prefix: Optional prefix for layer names

    Returns:
        Output tensor after dense layers and dropout
    """
    current = input_tensor
    for i, units in enumerate(config.hidden_layers):
        layer_name = f'{name_prefix}_dense_{i}' if name_prefix else None
        current = tf.keras.layers.Dense(units, activation='relu', name=layer_name)(current)
        if config.dropout_rate > 0:
            dropout_name = f'{name_prefix}_dropout_{i}' if name_prefix else None
            current = tf.keras.layers.Dropout(config.dropout_rate, name=dropout_name)(current)
    return current


@cache
def build_choose_model(input_length: int, n_plates: int) -> tf.keras.Model:
    """Build cascading choice model for plate -> color -> row selection.

    Architecture uses configurable hidden layers with dropout regularization.
    Each head (plate, color, row) receives softmax from previous heads as input.
    """
    input_layer = tf.keras.layers.Input(shape=(input_length,), dtype=tf.float32)

    # Plate selection head
    current = _build_dense_block(input_layer, name_prefix='plate_block')
    plate_head = tf.keras.layers.Dense(n_plates + 1, name='plate')(current)
    plate_softmax = tf.keras.layers.Activation('softmax', name='plate_softmax')(plate_head)

    # Color selection head (conditioned on plate)
    current = tf.keras.layers.concatenate((plate_softmax, input_layer))
    current = _build_dense_block(current, name_prefix='color_block')
    color_head = tf.keras.layers.Dense(config.n_colors, name='color')(current)
    color_softmax = tf.keras.layers.Activation('softmax', name='color_softmax')(color_head)

    # Row selection head (conditioned on plate and color)
    current = tf.keras.layers.concatenate((color_softmax, plate_softmax, input_layer))
    current = _build_dense_block(current, name_prefix='row_block')
    row_head = tf.keras.layers.Dense(config.n_colors, name='row')(current)
    row_softmax = tf.keras.layers.Activation('softmax', name='row_softmax')(row_head)

    model = tf.keras.Model(inputs=input_layer, outputs=[plate_softmax, color_softmax, row_softmax])

    return model


@cache
def build_end_of_round_model(input_length: int) -> tf.keras.Model:
    """Build end-of-round model for column selection per row.

    Architecture uses configurable hidden layers with dropout regularization.
    Outputs one softmax head per row for column selection.
    """
    input_layer = tf.keras.layers.Input(shape=(input_length,), dtype=tf.float32)

    current = _build_dense_block(input_layer, name_prefix='eor_block')

    output_heads = []
    for i in range(config.n_colors):
        head = tf.keras.layers.Dense(config.n_colors, activation='softmax', name=f'row_{i}_col')(current)
        output_heads.append(head)

    model = tf.keras.Model(inputs=input_layer, outputs=output_heads)
    return model


@dataclass
class BotPlayer(Player):
    n_plates: int = 0
    n_players: int = 0
    input_length: int = 0
    start_input_index: int = 0

    epsilon: typing.Optional[float] = None  # Initialized from config in __post_init__

    _choose_model: typing.Optional[tf.keras.models.Model] = None
    _end_of_round_model: typing.Optional[tf.keras.models.Model] = None

    # Metrics tracking for invalid actions
    invalid_row_count: int = 0    # Times no valid row existed (tiles to penalties)
    invalid_col_count: int = 0    # Times no valid column existed during EOR
    total_choices: int = 0        # Total choose actions made
    total_eor_placements: int = 0 # Total EOR placements attempted

    def __post_init__(self) -> None:
        """Initialize epsilon from config if not provided."""
        if self.epsilon is None:
            self.epsilon = config.epsilon_start

    def reset_metrics(self) -> None:
        """Reset per-game metrics counters."""
        self.invalid_row_count = 0
        self.invalid_col_count = 0
        self.total_choices = 0
        self.total_eor_placements = 0

    def get_invalid_rates(self) -> typing.Tuple[float, float]:
        """
        Get invalid action rates for this episode.

        Returns:
            Tuple of (invalid_row_rate, invalid_col_rate)
        """
        row_rate = (
            self.invalid_row_count / max(self.total_choices, 1)
        )
        col_rate = (
            self.invalid_col_count / max(self.total_eor_placements, 1)
        )
        return row_rate, col_rate

    @LazyProperty
    def choose_model(self) -> tf.keras.models.Model:
        return build_choose_model(self.input_length - self.start_input_index, self.n_plates)

    @LazyProperty
    def end_of_round_model(self) -> tf.keras.models.Model:
        return build_end_of_round_model(self.input_length - self.start_input_index)

    def _get_player_state_start(self, game_state: np.ndarray) -> int:
        """
        Compute the starting index of this player's state in the game state array.

        Game state layout:
        - [0:n_colors]: bag (5 values)
        - [n_colors:2*n_colors]: graveyard (5 values)
        - [2*n_colors:(2+n_plates)*n_colors]: plates (n_plates*5 values)
        - [(2+n_plates)*n_colors:(3+n_plates)*n_colors]: central (5 values)
        - [(3+n_plates)*n_colors]: first player token (1 value)
        - Then player states...

        Returns:
            int: Index where this player's state begins.
        """
        n_colors = config.n_colors
        # After bag, graveyard, plates, central, and first player token
        # = n_colors + n_colors + n_plates*n_colors + n_colors + 1
        # = (3 + n_plates) * n_colors + 1
        base_offset = (3 + self.n_plates) * n_colors + 1
        player_state_size = 2 * n_colors + n_colors * n_colors + 1  # left + right + penalties
        return base_offset + self.index * player_state_size

    def internal_choice(self, game_state: np.ndarray) -> typing.Tuple[int, int, int, int]:
        """
        Select an action (plate, color, row) using action masking.

        Action masking ensures only valid actions are selected:
        - Plate/color combinations where the plate has tiles of that color
        - Rows that can accept the selected color

        Returns:
            Tuple of (plate_index, color_index, row_index, n_tiles_retrieved)
        """
        # Track total choices for metrics
        self.total_choices += 1

        # Get the plate-color mask (which plate-color combos have tiles)
        plate_color_mask = get_plate_color_mask(game_state, self.n_plates)

        # Check if any valid plate-color combination exists
        if not np.any(plate_color_mask > 0):
            # No valid actions - this shouldn't happen in normal gameplay
            # Return dummy values; the game should handle empty plates
            return 0, 0, 0, 0

        # Get player state start for row masking
        player_state_start = self._get_player_state_start(game_state)

        # Track whether row selection was invalid
        no_valid_row = False

        if random.random() < self.epsilon:
            # Random exploration: sample from valid actions only
            i_plate, i_color = sample_from_2d_mask(plate_color_mask)

            # Get row mask for the selected color
            row_mask = get_row_mask_for_color(game_state, i_color, player_state_start)
            if np.any(row_mask > 0):
                i_row = sample_from_mask(row_mask)
            else:
                # No valid row - tiles will go to penalties (this is valid gameplay)
                i_row = random.randint(0, config.n_colors - 1)
                no_valid_row = True
        else:
            # Policy-based selection with masking
            normalized_state = normalize_state(
                game_state, config.n_colors, self.n_plates, self.n_players
            )
            short_game_state = normalized_state[self.start_input_index:]
            reshaped_state = tf.convert_to_tensor(
                np.expand_dims(short_game_state, axis=0), dtype=tf.float32
            )
            plate_probs, color_probs, row_probs = self.choose_model(reshaped_state)

            # Convert to numpy for masking
            plate_probs = plate_probs[0].numpy()
            color_probs = color_probs[0].numpy()

            # Flatten plate-color mask to get per-plate validity
            # A plate is valid if it has any color available
            plate_mask = np.any(plate_color_mask > 0, axis=1).astype(np.float32)

            # Apply plate mask and select best plate
            masked_plate_probs = apply_mask_to_logits(plate_probs, plate_mask)
            i_plate = int(np.argmax(masked_plate_probs))

            # Get color mask for selected plate
            color_mask = plate_color_mask[i_plate, :]

            # Apply color mask and select best color
            masked_color_probs = apply_mask_to_logits(color_probs, color_mask)
            i_color = int(np.argmax(masked_color_probs))

            # Get row mask for selected color
            row_mask = get_row_mask_for_color(game_state, i_color, player_state_start)
            row_probs = row_probs[0].numpy()

            if np.any(row_mask > 0):
                masked_row_probs = apply_mask_to_logits(row_probs, row_mask)
                i_row = int(np.argmax(masked_row_probs))
            else:
                # No valid row - select any (tiles will go to penalties)
                i_row = int(np.argmax(row_probs))
                no_valid_row = True

        # Track invalid row selections
        if no_valid_row:
            self.invalid_row_count += 1

        n_tiles_retrieved = config.get_tile_retrieved(i_plate, i_color, game_state)
        # Note: taboo_penalty removed - action masking ensures valid plate-color selection

        return i_plate, i_color, i_row, n_tiles_retrieved

    def end_of_round(self, game_state: np.ndarray) -> typing.List[int]:
        """
        Handle end-of-round tile placement with action masking.

        For each complete row in the left panel, select a valid column in the
        right panel to place the tile. Action masking ensures only valid
        placements are selected (empty cell, no duplicate color in row/column).

        Returns:
            List of column choices per row (-1 if row was not complete).
        """
        # Normalize state for model inference
        normalized_state = normalize_state(
            game_state, config.n_colors, self.n_plates, self.n_players
        )
        short_game_state = normalized_state[self.start_input_index:]
        reshaped_state = tf.convert_to_tensor(
            np.expand_dims(short_game_state, axis=0), dtype=tf.float32
        )
        estimated_scores = self.end_of_round_model(reshaped_state)

        # Get player state start for column masking
        player_state_start = self._get_player_state_start(game_state)

        choices = []
        logger(f"{self.prefix} - BEFORE EOR: penalties={self.penalties.n}")

        for i_row, col_scores in enumerate(estimated_scores):
            col_scores_np = col_scores[0].numpy()

            if i_row + 1 == self.left.state[i_row, 0]:
                # Row is complete, need to place tile
                # Track total EOR placements for metrics
                self.total_eor_placements += 1

                color = self.left.state[i_row, 1]

                # Get valid column mask for this row and color
                col_mask = get_column_mask_for_row(
                    game_state, i_row, color, player_state_start
                )

                if np.any(col_mask > 0):
                    # Valid columns exist - select one
                    if random.random() < self.epsilon:
                        i_col = sample_from_mask(col_mask)
                    else:
                        masked_col_scores = apply_mask_to_logits(col_scores_np, col_mask)
                        i_col = int(np.argmax(masked_col_scores))

                    choices.append(i_col)

                    # Place tile (valid placement guaranteed by masking)
                    self.score += self.right.count_score(i_row, i_col, color)
                    self.right.state[i_row, i_col] = color
                else:
                    # No valid column - this is a rare edge case
                    # Tile cannot be placed; add penalty
                    # Track invalid column placement
                    self.invalid_col_count += 1

                    if random.random() < self.epsilon:
                        i_col = random.randint(0, config.n_colors - 1)
                    else:
                        i_col = int(np.argmax(col_scores_np))

                    choices.append(i_col)
                    self.penalties.n += 1

                # Clear the left panel row
                self.left.state[i_row, 0] = 0
                self.left.state[i_row, 1] = -1
            else:
                choices.append(-1)

        logger(f"{self.prefix} - AFTER EOR: penalties={self.penalties.n}")

        self.score -= self.penalties.get_score()
        return choices

    def __repr__(self) -> str:
        return super().__repr__()
