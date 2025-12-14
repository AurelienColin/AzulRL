import typing
import numpy as np

from src.config import config


def normalize_state(
        state: np.ndarray,
        n_colors: int,
        n_plates: int,
        n_players: int
) -> np.ndarray:
    """
    Normalize state values to [0, 1] range for stable neural network training.

    The game state structure is:
        [0:n_colors]                           - bag tile counts (0-20)
        [n_colors:2*n_colors]                  - graveyard tile counts (0-20)
        [2*n_colors:(2+n_plates)*n_colors]     - plate tile counts (0-4 per color)
        [(2+n_plates)*n_colors:(3+n_plates)*n_colors] - central tile counts (0-20)
        [(3+n_plates)*n_colors]                - first player token (0/1)
        [... player states ...]                - left panel, right panel, penalties
        [-n_players:]                          - one-hot current player (0/1)

    Args:
        state: Raw integer game state array.
        n_colors: Number of tile colors (typically 5).
        n_plates: Number of plates in the game (depends on player count).
        n_players: Number of players in the game.

    Returns:
        Normalized state array with float values in [0, 1] range.
    """
    # Explicitly copy to avoid modifying the original state array
    normalized = state.astype(np.float32, copy=True)

    # Indices for different state sections
    bag_end = n_colors
    graveyard_end = 2 * n_colors
    plates_end = (2 + n_plates) * n_colors
    central_end = (3 + n_plates) * n_colors
    first_player_idx = central_end
    player_states_start = first_player_idx + 1
    one_hot_start = len(state) - n_players

    # Normalize bag and graveyard (0-20 range)
    normalized[:graveyard_end] /= config.n_tile_per_color

    # Normalize plate tile counts (0-4 range)
    normalized[graveyard_end:plates_end] /= config.n_tile_per_plate

    # Normalize central tile counts (0-20 range)
    normalized[plates_end:central_end] /= config.n_tile_per_color

    # First player token already 0/1 - no normalization needed

    # Normalize player states
    # Player state structure from Player.get_state():
    #   left.state.flatten(): shape (n_colors, 2) -> [count0, color0, count1, color1, ...]
    #   right.state.flatten(): shape (n_colors, n_colors) -> n_colors^2 color values
    #   penalties.state: 1 value (penalty score)
    player_state_size = 2 * n_colors + n_colors ** 2 + 1
    for i_player in range(n_players):
        player_start = player_states_start + i_player * player_state_size

        # Left panel: interleaved [count, color] pairs for each row
        # Flattened from (n_colors, 2) -> [count0, color0, count1, color1, ...]
        left_start = player_start
        for i_row in range(n_colors):
            count_idx = left_start + 2 * i_row
            color_idx = left_start + 2 * i_row + 1
            row_capacity = i_row + 1

            # Normalize count (0 to row_capacity)
            normalized[count_idx] /= row_capacity

            # Normalize color (-1 to n_colors-1) -> (0 to 1)
            # Map -1 -> 0, 0 -> 0.2, 1 -> 0.4, ..., 4 -> 1.0
            normalized[color_idx] = (normalized[color_idx] + 1) / n_colors

        # Right panel: n_colors^2 values are color indices (-1 to n_colors-1)
        # Same normalization as left panel colors
        right_start = left_start + 2 * n_colors
        right_end = right_start + n_colors ** 2
        normalized[right_start:right_end] = (
            (normalized[right_start:right_end] + 1) / n_colors
        )

        # Penalties: cap at 10 and normalize
        penalty_idx = right_end
        normalized[penalty_idx] = min(normalized[penalty_idx], 10.0) / 10.0

    # One-hot player encoding already 0/1 - no normalization needed

    return normalized


def to_hot_encoded(
        choices: typing.List[typing.List[typing.Tuple[int, ...]]],
        *vmaxs: int
) -> typing.List[typing.List[np.ndarray]]:
    subarrays = [[] for _ in range(len(vmaxs))]

    for i, vmax in enumerate(vmaxs):
        for j, player_choices in enumerate(choices):
            subarrays[i].append([])
            for turn_choice in player_choices:
                hot_encoded = np.zeros(vmax)
                if turn_choice[i] != -1:
                    hot_encoded[turn_choice[i]] = 1
                subarrays[i][j].append(hot_encoded)
            subarrays[i][j] = np.array(subarrays[i][j])
    return subarrays
