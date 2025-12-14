"""Random Agent for Azul.

Selects uniformly random valid actions. Used as a baseline for evaluating
trained RL agents.
"""
import typing
import numpy as np
from dataclasses import dataclass

from src.obj.player import Player
from src.obj.bot_player import (
    get_plate_color_mask,
    get_row_mask_for_color,
    get_column_mask_for_row,
    sample_from_mask,
    sample_from_2d_mask,
)
from src.config import config


@dataclass
class RandomAgent(Player):
    """Agent that selects uniformly random valid actions.

    This agent serves as a baseline for evaluating RL training progress.
    It always selects from valid actions only (no invalid moves).
    """

    @property
    def displayed_name(self) -> str:
        return f"RandomAgent {self.index}"

    def internal_choice(self, game_state: np.ndarray) -> typing.Tuple[int, int, int, int]:
        """
        Select a random valid action (plate, color, row).

        Uses action masking to ensure only valid actions are selected:
        1. Get mask of valid (plate, color) combinations
        2. Sample a random valid (plate, color)
        3. Get mask of valid rows for that color
        4. Sample a random valid row

        Args:
            game_state: The full game state array (non-normalized).

        Returns:
            Tuple of (plate_index, color_index, row_index, n_tiles_retrieved).
        """
        n_plates = config.n_plates

        # Step 1: Get valid (plate, color) combinations
        plate_color_mask = get_plate_color_mask(game_state, n_plates)

        # Step 2: Sample random valid (plate, color)
        i_plate, i_color = sample_from_2d_mask(plate_color_mask)

        if i_plate == -1:
            # No valid actions - should not happen in normal gameplay
            return 0, 0, 0, 0

        # Step 3: Get valid rows for this color
        # Calculate player state start index
        # Game state layout: bag + graveyard + plates + central + players
        # = 2*n_colors + (n_plates + 1)*n_colors + n_players*player_n_states
        player_state_start = (
            2 * config.n_colors +  # bag + graveyard
            (n_plates + 1) * config.n_colors +  # plates + central
            self.index * self.n_states  # previous players
        )

        row_mask = get_row_mask_for_color(game_state, i_color, player_state_start)

        # Step 4: Sample random valid row
        i_row = sample_from_mask(row_mask)

        if i_row == -1:
            # No valid row - penalties will be added (put in penalty row)
            # Choose any row, the game logic will handle penalties
            i_row = 0

        # Get number of tiles retrieved
        n_tiles_retrieved = config.get_tile_retrieved(i_plate, i_color, game_state)

        return i_plate, i_color, i_row, n_tiles_retrieved

    def end_of_round(self, game_state: np.ndarray) -> typing.List[int]:
        """
        Select random valid columns for end-of-round tile placement.

        For each completed row in the left panel, select a random valid column
        in the right panel to place the tile.

        Args:
            game_state: The full game state array (non-normalized).

        Returns:
            List of column choices for each row (-1 if row not complete).
        """
        choices = []
        n_plates = config.n_plates

        # Calculate player state start index
        player_state_start = (
            2 * config.n_colors +
            (n_plates + 1) * config.n_colors +
            self.index * self.n_states
        )

        for i_row in range(config.n_colors):
            if i_row + 1 == self.left.state[i_row, 0]:
                # Row is complete, need to choose a column
                color = self.left.state[i_row, 1]

                # Get valid columns
                col_mask = get_column_mask_for_row(
                    game_state, i_row, color, player_state_start
                )
                i_col = sample_from_mask(col_mask)

                if i_col == -1:
                    # No valid column - this shouldn't happen in valid game states
                    # Choose column 0 and let game logic handle penalties
                    i_col = 0

                choices.append(i_col)

                # Place tile and update score
                if (self.right.state[i_row, i_col] == -1 and
                        color not in self.right.state[:, i_col] and
                        color not in self.right.state[i_row, :]):
                    self.score += self.right.count_score(i_row, i_col, color)
                    self.right.state[i_row, i_col] = color
                else:
                    self.penalties.n += 1

                self.left.state[i_row, 0] = 0
                self.left.state[i_row, 1] = -1
            else:
                choices.append(-1)

        self.score -= self.penalties.get_score()
        self.penalties.n = 0
        return choices
