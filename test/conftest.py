"""
Shared fixtures for AzulRL test suite.

This module provides reusable fixtures for game, player, and model testing.
"""
import pytest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config, Config
from src.obj.bag import Bag
from src.obj.plate import Plate
from src.obj.container import Container
from src.obj.central import Central
from src.obj.player import Player, Left, Right, Penalties
from src.obj.game import Game
try:
    from src.obj.bot_player import BotPlayer
    _HAS_TF = True
except ImportError:
    BotPlayer = None  # type: ignore[assignment,misc]
    _HAS_TF = False
from src.obj.replay_buffer import ReplayBuffer


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> Config:
    """Provide access to the default configuration."""
    return config


@pytest.fixture
def n_players() -> int:
    """Default number of players for testing."""
    return 4


@pytest.fixture
def n_plates(n_players: int) -> int:
    """Number of plates based on player count."""
    return config.get_plate_number(n_players)


# =============================================================================
# Container Fixtures
# =============================================================================

@pytest.fixture
def empty_container() -> Container:
    """Create an empty container."""
    return Container(displayed_name="Test Container")


@pytest.fixture
def filled_container() -> Container:
    """Create a container with some tiles."""
    container = Container(displayed_name="Test Container")
    container.n_blacks = 3
    container.n_whites = 2
    container.n_reds = 1
    container.n_blues = 4
    container.n_yellows = 0
    return container


@pytest.fixture
def fresh_bag() -> Bag:
    """Create a fresh bag with default tile counts."""
    return Bag()


@pytest.fixture
def empty_plate() -> Plate:
    """Create an empty plate."""
    return Plate(index=0)


@pytest.fixture
def filled_plate() -> Plate:
    """Create a plate with tiles."""
    plate = Plate(index=0)
    plate.n_blacks = 2
    plate.n_whites = 1
    plate.n_reds = 1
    return plate


@pytest.fixture
def empty_central() -> Central:
    """Create an empty central area."""
    return Central()


# =============================================================================
# Player Fixtures
# =============================================================================

@pytest.fixture
def empty_left() -> Left:
    """Create an empty left panel."""
    return Left()


@pytest.fixture
def empty_right() -> Right:
    """Create an empty right panel."""
    return Right()


@pytest.fixture
def penalties() -> Penalties:
    """Create a penalties counter."""
    return Penalties()


@pytest.fixture
def player() -> Player:
    """Create a basic player."""
    return Player(index=0)


@pytest.fixture
def player_with_tiles() -> Player:
    """Create a player with some tiles on left panel."""
    p = Player(index=0)
    # Fill row 0 with 1 black tile
    p.left.state[0, 0] = 1
    p.left.state[0, 1] = 0  # black color
    # Fill row 2 partially with 2 red tiles
    p.left.state[2, 0] = 2
    p.left.state[2, 1] = 2  # red color
    return p


# =============================================================================
# Game Fixtures
# =============================================================================

@pytest.fixture
def game(n_players: int) -> Game:
    """Create a game with default settings."""
    return Game(n_players=n_players)


@pytest.fixture
def game_3_players() -> Game:
    """Create a 3-player game."""
    return Game(n_players=3)


@pytest.fixture
def game_4_players() -> Game:
    """Create a 4-player game."""
    return Game(n_players=4)


@pytest.fixture
def game_state(game: Game) -> np.ndarray:
    """Get initial game state."""
    return game.get_state()


# =============================================================================
# Bot Player Fixtures
# =============================================================================

@pytest.fixture
def bot_player(game: Game) -> "BotPlayer":
    if not _HAS_TF:
        pytest.skip("tensorflow not available")
    assert BotPlayer is not None
    """Create a bot player configured for a game."""
    bot = BotPlayer(
        index=0,
        n_plates=game.n_plates,
        n_players=game.n_players,
        input_length=game.n_states,
        start_input_index=config.start_input_index
    )
    return bot


@pytest.fixture
def bot_player_high_epsilon(game: Game) -> "BotPlayer":
    if not _HAS_TF:
        pytest.skip("tensorflow not available")
    """Create a bot player with high exploration rate."""
    bot = BotPlayer(
        index=0,
        n_plates=game.n_plates,
        n_players=game.n_players,
        input_length=game.n_states,
        start_input_index=config.start_input_index,
        epsilon=1.0  # Full exploration
    )
    return bot


@pytest.fixture
def bot_player_no_exploration(game: Game) -> "BotPlayer":
    if not _HAS_TF:
        pytest.skip("tensorflow not available")
    """Create a bot player with no exploration (greedy)."""
    bot = BotPlayer(
        index=0,
        n_plates=game.n_plates,
        n_players=game.n_players,
        input_length=game.n_states,
        start_input_index=config.start_input_index,
        epsilon=0.0  # No exploration
    )
    return bot


# =============================================================================
# Replay Buffer Fixtures
# =============================================================================

@pytest.fixture
def empty_replay_buffer() -> ReplayBuffer:
    """Create an empty replay buffer."""
    return ReplayBuffer(capacity=100)


@pytest.fixture
def filled_replay_buffer() -> ReplayBuffer:
    """Create a replay buffer with sample experiences."""
    buffer = ReplayBuffer(capacity=100)
    # Add some sample experiences
    for _ in range(50):
        state = np.random.randn(100).astype(np.float32)
        targets = [
            np.random.randn(10).astype(np.float32),
            np.random.randn(5).astype(np.float32),
            np.random.randn(5).astype(np.float32)
        ]
        buffer.buffer.append((state, targets))
    return buffer


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def sample_state_vector(game: Game) -> np.ndarray:
    """Create a sample state vector for testing."""
    return game.get_state()


@pytest.fixture
def normalized_state(sample_state_vector: np.ndarray, n_plates: int, n_players: int) -> np.ndarray:
    """Create a normalized state vector."""
    from src.utils import normalize_state
    return normalize_state(sample_state_vector, config.n_colors, n_plates, n_players)


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility."""
    np.random.seed(42)
    yield
    # Reset to random state after test
    np.random.seed()


@pytest.fixture
def deterministic_game(random_seed, n_players: int) -> Game:
    """Create a deterministic game with fixed random seed."""
    return Game(n_players=n_players)
