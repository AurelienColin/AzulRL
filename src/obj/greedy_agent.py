"""Greedy Agent for Azul.

Selects actions that maximize immediate score gain. Used as a baseline for
evaluating trained RL agents. Should perform better than random but worse
than a well-trained RL agent.
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
)
from src.config import config


@dataclass
class GreedyAgent(Player):
    """Agent that selects actions maximizing immediate score gain.

    This agent serves as a stronger baseline than RandomAgent.
    It evaluates all valid actions and selects the one with the best
    immediate reward, considering:
    - Tiles placed vs penalties incurred
    - Rows being filled (progress towards scoring)
    - Avoiding placement that would cause overflow
    """

    @property
    def displayed_name(self) -> str:
        return f"GreedyAgent {self.index}"

    def _evaluate_action(
        self,
        i_plate: int,
        i_color: int,
        i_row: int,
        n_tiles: int,
        game_state: np.ndarray
    ) -> float:
        """
        Evaluate the immediate value of an action.

        Scoring heuristics:
        - Base score: number of tiles successfully placed
        - Penalty for overflow (tiles that go to penalty row)
        - Bonus for completing a row (ready for end-of-round scoring)
        - Small bonus for rows closer to completion

        Args:
            i_plate: Plate index.
            i_color: Color index.
            i_row: Row index.
            n_tiles: Number of tiles being picked up.
            game_state: Current game state.

        Returns:
            float: Estimated value of this action.
        """
        row_capacity = i_row + 1
        current_count = self.left.state[i_row, 0]
        current_color = self.left.state[i_row, 1]

        # Check if row can accept this color
        if current_color != -1 and current_color != i_color:
            # Different color in row - all tiles go to penalty
            return -n_tiles * 2.0

        # Check if color is in right panel for this row
        if i_color in self.right.state[i_row, :]:
            # Color already placed in this row - all tiles go to penalty
            return -n_tiles * 2.0

        # Calculate how many tiles fit vs overflow
        space_available = row_capacity - current_count
        tiles_placed = min(n_tiles, space_available)
        overflow = max(0, n_tiles - space_available)

        score = 0.0

        # Reward for tiles placed
        score += tiles_placed * 1.0

        # Penalty for overflow
        score -= overflow * 2.0

        # Bonus for completing a row (it will score at end of round)
        new_count = current_count + tiles_placed
        if new_count == row_capacity:
            # Row will be complete - estimate potential points
            # A completed row scores at least 1 point, more with adjacency
            score += 3.0 + i_row * 0.5  # Higher rows are worth more

        # Small bonus for progress towards completion
        completion_ratio = new_count / row_capacity
        score += completion_ratio * 0.5

        # Slight preference for larger rows (more efficient use of tiles)
        score += i_row * 0.1

        return score

    def internal_choice(self, game_state: np.ndarray) -> typing.Tuple[int, int, int, int]:
        """
        Select the action that maximizes immediate score.

        Evaluates all valid (plate, color, row) combinations and returns
        the one with the highest estimated value.

        Args:
            game_state: The full game state array (non-normalized).

        Returns:
            Tuple of (plate_index, color_index, row_index, n_tiles_retrieved).
        """
        n_plates = config.n_plates

        # Get valid (plate, color) combinations
        plate_color_mask = get_plate_color_mask(game_state, n_plates)

        # Calculate player state start index
        player_state_start = (
            2 * config.n_colors +
            (n_plates + 1) * config.n_colors +
            self.index * self.n_states
        )

        best_action = None
        best_score = float('-inf')

        # Iterate over all valid (plate, color) combinations
        for i_plate in range(n_plates + 1):
            for i_color in range(config.n_colors):
                if plate_color_mask[i_plate, i_color] == 0:
                    continue

                n_tiles = config.get_tile_retrieved(i_plate, i_color, game_state)
                if n_tiles == 0:
                    continue

                # Get valid rows for this color
                row_mask = get_row_mask_for_color(
                    game_state, i_color, player_state_start
                )

                # Evaluate each valid row
                for i_row in range(config.n_colors):
                    if row_mask[i_row] > 0:
                        action_score = self._evaluate_action(
                            i_plate, i_color, i_row, n_tiles, game_state
                        )

                        if action_score > best_score:
                            best_score = action_score
                            best_action = (i_plate, i_color, i_row, n_tiles)

                # Also consider penalty row (no valid row)
                if np.sum(row_mask) == 0:
                    # All rows invalid - must take penalty
                    # Still might be worth picking tiles to deny opponent
                    action_score = -n_tiles * 2.0
                    if action_score > best_score:
                        best_score = action_score
                        best_action = (i_plate, i_color, 0, n_tiles)

        if best_action is None:
            # No valid actions found - return default
            return 0, 0, 0, 0

        return best_action

    def end_of_round(self, game_state: np.ndarray) -> typing.List[int]:
        """
        Select optimal columns for end-of-round tile placement.

        For each completed row, choose the column that maximizes immediate
        scoring potential (adjacency bonuses).

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

                # Find best column (maximize adjacency score)
                best_col = -1
                best_score = float('-inf')

                for i_col in range(config.n_colors):
                    if col_mask[i_col] > 0:
                        # Simulate placing tile and calculate score
                        # Use existing scoring methods from Right class
                        score = self._estimate_column_score(i_row, i_col, color)
                        if score > best_score:
                            best_score = score
                            best_col = i_col

                if best_col == -1:
                    # No valid column - use first available or 0
                    best_col = sample_from_mask(col_mask)
                    if best_col == -1:
                        best_col = 0

                i_col = best_col
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

    def _estimate_column_score(self, i_row: int, i_col: int, color: int) -> float:
        """
        Estimate the score for placing a tile at (i_row, i_col).

        Uses adjacency heuristics to estimate scoring potential.

        Args:
            i_row: Row index.
            i_col: Column index.
            color: Color being placed.

        Returns:
            float: Estimated score for this placement.
        """
        score = 1.0  # Base score for any valid placement

        # Count horizontal adjacency
        h_count = 0
        for delta in [-1, 1]:
            check_col = i_col + delta
            while 0 <= check_col < config.n_colors:
                if self.right.state[i_row, check_col] != -1:
                    h_count += 1
                    check_col += delta
                else:
                    break

        # Count vertical adjacency
        v_count = 0
        for delta in [-1, 1]:
            check_row = i_row + delta
            while 0 <= check_row < config.n_colors:
                if self.right.state[check_row, i_col] != -1:
                    v_count += 1
                    check_row += delta
                else:
                    break

        # Adjacency bonus
        if h_count > 0:
            score += h_count + 1
        if v_count > 0:
            score += v_count + 1

        # Bonus for completing row/column
        if h_count == 4:  # Would complete row
            score += config.complete_row_score
        if v_count == 4:  # Would complete column
            score += config.complete_column_score

        # Check if this completes a color
        color_count = np.sum(self.right.state == color)
        if color_count == 4:  # This would be the 5th
            score += config.complete_color_score

        return score
