from dataclasses import dataclass
import typing
import numpy as np
import os

@dataclass
class Config:
    n_tile_per_color: int = 20
    n_tile_per_plate: int = 4
    taboo_penalty: int = 1000

    colors: typing.Tuple[str, ...] = ('black', 'white', 'red', 'blue', 'yellow')
    n_colors: int = len(colors)

    n_plates: typing.Optional[int] = None

    complete_column_score = 7
    complete_row_score = 2
    complete_color_score = 10

    weights_dir = 'res/models/'
    model_weights_path_1 = os.path.join(weights_dir, 'model_weights_1.h5')
    model_weights_path_2 = os.path.join(weights_dir, 'model_weights_2.h5')
    history_path =  os.path.join(weights_dir, 'history.png')

    def __post_init__(self):
        os.makedirs(self.weights_dir, exist_ok=True)


    def get_plate_number(self, n_players: int):
        if n_players == 3:
            self.n_plates = 7
        elif n_players == 4:
            self.n_plates = 9
        else:
            raise NotImplementedError(f"Only 3 or 4 players supported, not {n_players}")
        return self.n_plates

    def get_tile_retrieved(self, i_plate: int, i_color: int, game_state: np.ndarray) -> int:
        start_index = 2 * self.n_colors
        index = start_index + self.n_colors * i_plate + i_color
        return game_state[index]

    def get_color(self, color_index: int) -> str:
        if color_index == -1:
            return "None"
        return self.colors[color_index]


config = Config()
