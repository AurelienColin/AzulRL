from dataclasses import dataclass
import typing
import numpy as np
from src.obj.container import Container
from rignak.src.lazy_property import LazyProperty
from src.config import config


@dataclass
class Left(Container):
    _state: typing.Optional[typing.List[np.ndarray]] = None
    n_states = config.n_colors * 2

    @LazyProperty
    def state(self) -> typing.List[np.ndarray]:
        return [np.full((i, 2), -1) for i in range(1, 1 + config.n_colors)]

    def get_state(self) -> np.ndarray:
        state = np.zeros(self.n_states)
        for i in range(1, config.n_colors):
            state[i] = self.state[i][0]
            state[i + self.n_colors] = self.state[i][1]
        return state

    def __repr__(self) -> str:
        description = "\n".join((
            f"Left part",
            *(
                f"{self.state[i_row]}/{i_row} - {config.get_color(self.state[i_row + config.n_colors])}"
                for i_row in range(config.n_colors)
            )
        ))
        return description


@dataclass
class Right(Container):
    is_complete: bool = False

    _state: typing.Optional[np.ndarray] = None

    def __repr__(self) -> str:
        description = "\n".join((
            f"Right part",
            *(
                '\t'.join((config.get_color(self.state[i_row, i_col])) for i_col in range(config.n_colors))
                for i_row in range(config.n_colors)
            )
        ))
        return description

    @LazyProperty
    def state(self) -> np.ndarray:
        return np.full((5, 5), -1)

    def get_column_score(self, i_row: int, i_col: int) -> int:
        col_score = 0

        for before in range(1, config.n_colors - 1):
            j_col = i_col - before
            if j_col >= 0 and self.state[i_row, j_col] == -1:
                col_score += 1
            else:
                break

        for after in range(1, config.n_colors):
            j_col = i_col + after
            if j_col < config.n_colors and self.state[i_row, j_col] == -1:
                col_score += 1
            else:
                break

        if col_score:
            col_score += 1

        if col_score == 5:
            col_score += config.complete_column_score
            self.is_complete = True
        return col_score

    def get_row_score(self, i_row: int, i_col: int) -> int:
        row_score = 0

        for before in range(1, config.n_colors - 1):
            j_row = i_row - before
            if i_col >= 0 and self.state[j_row, i_col] == -1:
                row_score += 1
            else:
                break

        for after in range(1, config.n_colors):
            j_row = i_row + after
            if j_row < config.n_colors and self.state[j_row, i_col] == -1:
                row_score += 1
            else:
                break

        if row_score:
            row_score += 1

        if row_score == 5:
            row_score += config.complete_row_score
        return row_score

    def get_color_score(self, color: int) -> int:
        if (self.state == color) == config.n_colors:
            color_score = config.complete_color_score
        else:
            color_score = 0
        return color_score

    def count_score(self, i_row: int, i_col: int, color: int) -> int:
        column_score = self.get_column_score(i_row, i_col)
        row_score = self.get_row_score(i_row, i_col)
        color_score = self.get_color_score(color)
        return color_score + row_score + column_score


@dataclass
class Penalties:
    n = 0

    def get_score(self) -> int:
        score = 0
        if self.n > 0:
            score += 1 + (self.n > 1)
        if self.n > 2:
            score += 2 * (1 + (self.n > 3))
        if self.n > 3:
            score += 3 * (self.n - 3)
        return score

    @property
    def state(self) -> np.ndarray:
        return np.full(1, self.get_score())


@dataclass
class Player:
    index:int=0
    score: int = 0
    is_first: bool = False
    _left: typing.Optional[Left] = None
    _right: typing.Optional[Right] = None
    _penalties: typing.Optional[Penalties] = None

    n_states: int = 2 * config.n_colors + config.n_colors ** 2 + 1

    @property
    def n_actions(self) -> int:
        return config.n_colors * (config.n_plates + 1)

    @property
    def displayed_name(self) -> str:
        return f"Player {self.index}"

    @LazyProperty
    def left(self) -> Left:
        return Left()

    @LazyProperty
    def right(self) -> Right:
        return Right()

    def get_state(self):
        return np.concatenate((self.left.state.flatten(), self.right.state.flatten()), self.penalties.state)

    @LazyProperty
    def penalties(self) -> Penalties:
        return Penalties()

    def end_of_round(self, game_state: np.ndarray) -> None:
        """
        For the bot, each decision should be composed of 5 parts: one for each raw, indicating the col in which
        setting the tile.
        """
        for i_row in config.n_colors:
            if i_row == self.left.state[i_row]:
                question = f"{self.prefix}Row {i_row}: Choose which col. to fill: "
                i_col = get_input(question, lambda x: isinstance(x, int))

                color = self.left.state[i_row + config.n_colors]
                if (self.right.state[i_row, i_col] == -1 and
                        color not in self.right.state[:, i_col]
                        and color not in self.right.state[i_row, :]):
                    self.score = self.right.count_score(i_row, i_col, color)
                    self.right.state[i_row, i_col] = color
                else:
                    self.penalties += i_row
                self.left.state[i_row] = 0
                self.left.state[i_row + config.n_colors] = -1

        self.score -= self.penalties.get_score()
        self.penalties.n = 0

    @property
    def prefix(self) -> str:
        return f"Player {self.index} - "

    def choose(self, game_state: np.ndarray) -> typing.Tuple[int, int, int]:
        """
        For the bot, each decision should be composed of three parts:
        - the index of the plate into which we take tiles;
        - the index of the color we choose;
        - the index of the row of the Left container into which we put the tile.
        """
        n_tiles_retrieved = i_row = None
        while True:
            question = f"{self.prefix}Choose a plate: "
            i_plate = get_input(question, lambda x: isinstance(x, int))
            i_plate = int(i_plate)

            question = f"{self.prefix}Choose a color index: "
            i_color = get_input(question, lambda x: isinstance(x, int))
            i_color = int(i_color)

            suffix = " (central part)" if i_plate == config.n_plates else ""
            print(f"{self.prefix}Plate #{i_plate}{suffix}, color: {i_color} ({config.colors[i_color]})")

            confirmation = get_input(f"{prefix}Confirm? (y/n): ")
            if confirmation.lower() != "y":
                continue

            n_tiles_retrieved = config.get_tile_retrieved(i_plate, i_color, game_state)
            if n_tiles_retrieved == 0:
                print(f"{self.prefix}. No tile of that color. "
                      f"Don't try to cheat. Getting {config.taboo_penalty} penalties.")
                self.score -= config.taboo_penalty
                continue

            question = f"{prefix}Chose with line to put in (integer between 0 and {config.n_colors}): "
            i_row = get_input(question, lambda x: isinstance(x, int))
            i_row = int(i_row)

        total = n_tiles_retrieved + self.left.state[i_row]
        if total > i_row:  # Each row of the left part as the same number of places as its index.
            self.penalties.n += (total - i_row)
            total = i_row
        self.left.state[i_row] = total
        return i_plate, i_color, i_row

    def __repr__(self) -> str:
        description = "\n".join((
            f'-- {self.displayed_name} --',
            f"Score: {self.score}",
            f"Penalties: {self.penalties.n}",
            repr(self.left),
            repr(self.right)
        ))
        return description


def get_input(question: str, condition: typing.Optional[typing.Callable[[str], bool]] = None) -> str:
    while True:
        action = input(question)
        if condition is not None:
            try:
                assert condition(action)
            except AssertionError:
                print("AssertionError. Retry.")
                continue
        break
    return action
