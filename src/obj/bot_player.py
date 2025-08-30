import random
import typing
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from src.obj.player import Player
from src.config import config
from functools import cache

from rignak.src.lazy_property import LazyProperty
import os


@cache
def build_choose_model(input_length: int, n_plates: int) -> tf.keras.Model:
    input_layer = tf.keras.layers.Input(shape=(input_length,), dtype=tf.float32)

    current = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    current = tf.keras.layers.Dense(32, activation='relu')(current)

    plate_head = tf.keras.layers.Dense(n_plates + 1, name='plate')(current)

    current = tf.keras.layers.concatenate((plate_head, current))
    current = tf.keras.layers.Dense(16, activation='relu')(current)
    color_head = tf.keras.layers.Dense(config.n_colors, name='color')(current)

    current = tf.keras.layers.concatenate((color_head, current))
    current = tf.keras.layers.Dense(8, activation='relu')(current)
    row_head = tf.keras.layers.Dense(config.n_colors, name='row')(current)

    model = tf.keras.Model(inputs=input_layer, outputs=[plate_head, color_head, row_head])

    # if os.path.exists(config.model_weights_path_1):
    #     model.load_weights(config.model_weights_path_1)

    return model


@cache
def build_end_of_round_model(input_length: int) -> tf.keras.Model:
    input_layer = tf.keras.layers.Input(shape=(input_length,), dtype=tf.float32)

    current = tf.keras.layers.Dense(32, activation='relu')(input_layer)
    current = tf.keras.layers.Dense(16, activation='relu')(current)

    output_heads = []
    for i in range(config.n_colors):
        head = tf.keras.layers.Dense(config.n_colors, name=f'row_{i}_col')(current)
        output_heads.append(head)

    model = tf.keras.Model(inputs=input_layer, outputs=output_heads)
    # if os.path.exists(config.model_weights_path_2):
    #     model.load_weights(config.model_weights_path_2)
    return model


@dataclass
class BotPlayer(Player):
    n_plates: int = 0
    input_length: int = 0
    start_input_index: int = 0

    exploration = 0.1

    _choose_model: typing.Optional[tf.keras.models.Model] = None
    _end_of_round_model: typing.Optional[tf.keras.models.Model] = None

    @LazyProperty
    def choose_model(self) -> tf.keras.models.Model:
        return build_choose_model(self.input_length - self.start_input_index, self.n_plates)

    @LazyProperty
    def end_of_round_model(self) -> tf.keras.models.Model:
        return build_end_of_round_model(self.input_length - self.start_input_index)

    def internal_choice(self, game_state: np.ndarray) -> typing.Tuple[int, int, int, int]:
        if random.random() < self.exploration:
            i_plate = random.randint(0, self.n_plates)
            i_color = random.randint(0, config.n_colors - 1)
            i_row = random.randint(0, config.n_colors - 1)
            # print(f"Random choice: {i_plate=}, {i_color=}, {i_row=}")
        else:
            short_game_state=game_state[self.start_input_index:]
            reshaped_state = tf.convert_to_tensor(np.expand_dims(short_game_state, axis=0), dtype=tf.float32)
            plate_scores, color_scores, row_scores = self.choose_model(reshaped_state)

            i_plate = tf.math.argmax(plate_scores[0]).numpy()
            i_color = tf.math.argmax(color_scores[0]).numpy()
            i_row = tf.math.argmax(row_scores[0]).numpy()
            # print(f"RL choice: {i_plate=}, {i_color=}, {i_row=}")

        n_tiles_retrieved = config.get_tile_retrieved(i_plate, i_color, game_state)
        if n_tiles_retrieved == 0:
            self.score -= config.taboo_penalty

        return i_plate, i_color, i_row, n_tiles_retrieved

    def end_of_round(self, game_state: np.ndarray) -> None:
        short_game_state=game_state[self.start_input_index:]
        reshaped_state = tf.convert_to_tensor(np.expand_dims(short_game_state, axis=0), dtype=tf.float32)
        estimated_scores = self.end_of_round_model(reshaped_state)

        choices = []
        print(f"{self.prefix} - BEFORE EOR:\n{self.penalties.n=}\n{self.left}\n{self.right}")
        for i_row, col_scores in enumerate(estimated_scores):
            col_scores = col_scores[0]
            if i_row + 1 == self.left.state[i_row, 0]:
                if random.random() < self.exploration:
                    i_col = random.randint(0, config.n_colors - 1)
                else:
                    i_col = tf.math.argmax(col_scores).numpy()

                choices.append(i_col)

                color = self.left.state[i_row, 1]
                if (self.right.state[i_row, i_col] == -1 and
                        color not in self.right.state[:, i_col]
                        and color not in self.right.state[i_row, :]):
                    self.score += self.right.count_score(i_row, i_col, color)
                    self.right.state[i_row, i_col] = color
                else:
                    self.penalties.n += 1
                self.left.state[i_row, 0] = 0
                self.left.state[i_row, 1] = -1
            else:
                choices.append(-1)
        print(f"{self.prefix} - AFTER EOR:\n{self.penalties.n=}\n{self.left}\n{self.right}")

        self.score -= self.penalties.get_score()
        return choices

    def __repr__(self) -> str:
        return super().__repr__()
