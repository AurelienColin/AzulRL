"""Configuration settings for Azul RL training and game rules."""
from dataclasses import dataclass
import typing
import numpy as np
import os


@dataclass
class Config:
    """Global configuration for Azul game rules and RL hyperparameters.

    Contains tile counts, scoring bonuses, neural network architecture,
    training parameters, and file paths for model persistence.
    """
    # Tile and game constants
    n_tile_per_color: int = 20
    n_tile_per_plate: int = 4
    taboo_penalty: int = 1000
    NO_COLOR: int = -1  # Sentinel value for empty cell / no color selected

    # RL hyperparameters
    gamma: float = 0.99  # Discount factor for temporal credit assignment
    epsilon_start: float = 1.0      # Initial exploration rate (100%)
    epsilon_end: float = 0.01       # Final exploration rate (1%)
    epsilon_decay: float = 0.995    # Decay multiplier per episode

    # Experience replay parameters
    replay_buffer_size: int = 10000  # Maximum experiences to store in buffer
    batch_size: int = 64             # Batch size for training from replay buffer
    min_buffer_size: int = 128       # Minimum experiences before training starts
    train_every_n_rounds: int = 4    # Train after every N rounds of gameplay

    # Learning rate schedule parameters
    lr_initial: float = 0.001        # Initial learning rate
    lr_min: float = 0.00001          # Minimum learning rate (1% of initial)
    lr_decay_steps: int = 10000      # Steps for full cosine decay cycle
    lr_warmup_steps: int = 500       # Linear warmup steps before decay

    # Evaluation parameters
    eval_every_n_games: int = 50     # Evaluate win rate every N games
    eval_n_games: int = 20           # Number of evaluation games to play

    # Network architecture
    hidden_layers: typing.Tuple[int, ...] = (128, 64, 32)  # Layer sizes (wide to narrow)
    dropout_rate: float = 0.1  # Regularization between dense layers

    # State representation
    # start_input_index: Slice offset to exclude early state features from model input.
    # Setting to 2 * n_colors removes bag and graveyard counts (first 10 features).
    # Setting to 0 includes all state features including bag/graveyard info.
    # Bag/graveyard info could be useful for strategic decisions (knowing remaining tiles).
    start_input_index: int = 0  # Include all features (bag, graveyard, plates, central, etc.)

    colors: typing.Tuple[str, ...] = ('black', 'white', 'red', 'blue', 'yellow')
    n_colors: int = len(colors)

    n_plates: typing.Optional[int] = None

    complete_column_score: int = 7
    complete_row_score: int = 2
    complete_color_score: int = 10

    weights_dir: str = 'res/models/'
    model_weights_path_1: str = os.path.join(weights_dir, 'model_weights_1.weights.h5')
    model_weights_path_2: str = os.path.join(weights_dir, 'model_weights_2.weights.h5')
    history_path: str = os.path.join(weights_dir, 'history.png')
    metrics_path: str = os.path.join(weights_dir, 'metrics.json')

    def __post_init__(self):
        os.makedirs(self.weights_dir, exist_ok=True)


    def get_plate_number(self, n_players: int) -> int:
        """Get the number of factory plates based on player count.

        Args:
            n_players: Number of players (must be 3 or 4).

        Returns:
            Number of factory plates for the game.

        Raises:
            NotImplementedError: If player count is not 3 or 4.
        """
        if n_players == 3:
            self.n_plates = 7
        elif n_players == 4:
            self.n_plates = 9
        else:
            raise NotImplementedError(f"Only 3 or 4 players supported, not {n_players}")
        return self.n_plates

    def get_tile_retrieved(self, i_plate: int, i_color: int, game_state: np.ndarray) -> int:
        """Get the number of tiles of a color on a plate from game state.

        Args:
            i_plate: Index of the plate (0 to n_plates-1, or n_plates for central).
            i_color: Index of the color (0 to n_colors-1).
            game_state: Flattened numpy array of game state.

        Returns:
            Number of tiles of the specified color on the specified plate.
        """
        start_index = 2 * self.n_colors
        index = start_index + self.n_colors * i_plate + i_color
        return game_state[index]

    def get_color(self, color_index: int) -> str:
        """Get the color name from its index.

        Args:
            color_index: Index of the color (0-4), or NO_COLOR for no color.

        Returns:
            Name of the color, or "None" if index is NO_COLOR.
        """
        if color_index == self.NO_COLOR:
            return "None"
        return self.colors[color_index]


config = Config()
