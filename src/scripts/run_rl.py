import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import typing
from dataclasses import dataclass, field, asdict
from rignak.src.logging_utils import logger
from src.obj.game import Game
from src.obj.bot_player import BotPlayer
from src.obj.replay_buffer import ReplayBuffer
from src.obj.random_agent import RandomAgent
from src.obj.greedy_agent import GreedyAgent
from src.config import config
from src.utils import to_hot_encoded
from rignak.src.custom_display import Display


@dataclass
class TrainingMetrics:
    """
    Comprehensive metrics tracking for RL training.

    Tracks loss, win rates, scores, exploration, and training dynamics.
    All lists are indexed by episode number.
    """
    # Episode tracking
    episode: typing.List[int] = field(default_factory=list)

    # Loss metrics (per model)
    loss_choose: typing.List[float] = field(default_factory=list)
    loss_eor: typing.List[float] = field(default_factory=list)

    # Performance metrics
    win_rate: typing.List[float] = field(default_factory=list)
    avg_score: typing.List[float] = field(default_factory=list)
    avg_reward: typing.List[float] = field(default_factory=list)

    # Game dynamics
    game_length: typing.List[int] = field(default_factory=list)  # Number of rounds
    n_turns: typing.List[int] = field(default_factory=list)      # Number of player turns

    # Invalid action tracking
    invalid_row_rate: typing.List[float] = field(default_factory=list)  # Tiles to penalties
    invalid_col_rate: typing.List[float] = field(default_factory=list)  # EOR placement failures

    # Training hyperparameters over time
    exploration_rate: typing.List[float] = field(default_factory=list)
    learning_rate: typing.List[float] = field(default_factory=list)

    # Buffer statistics
    buffer_size_choose: typing.List[int] = field(default_factory=list)
    buffer_size_eor: typing.List[int] = field(default_factory=list)

    # Baseline comparison metrics
    win_rate_vs_random: typing.List[float] = field(default_factory=list)
    win_rate_vs_greedy: typing.List[float] = field(default_factory=list)

    def record_episode(
            self,
            episode_num: int,
            loss_choose: float,
            loss_eor: float,
            avg_score: float,
            game_length: int,
            n_turns: int,
            exploration_rate: float,
            learning_rate: float,
            buffer_size_choose: int,
            buffer_size_eor: int,
            win_rate: typing.Optional[float] = None,
            invalid_row_rate: float = 0.0,
            invalid_col_rate: float = 0.0,
            avg_reward: float = 0.0,
            win_rate_vs_random: typing.Optional[float] = None,
            win_rate_vs_greedy: typing.Optional[float] = None
    ) -> None:
        """Record metrics for a single episode."""
        self.episode.append(episode_num)
        self.loss_choose.append(loss_choose)
        self.loss_eor.append(loss_eor)
        self.avg_score.append(avg_score)
        self.game_length.append(game_length)
        self.n_turns.append(n_turns)
        self.exploration_rate.append(exploration_rate)
        self.learning_rate.append(learning_rate)
        self.buffer_size_choose.append(buffer_size_choose)
        self.buffer_size_eor.append(buffer_size_eor)
        self.win_rate.append(win_rate if win_rate is not None else float('nan'))
        self.invalid_row_rate.append(invalid_row_rate)
        self.invalid_col_rate.append(invalid_col_rate)
        self.avg_reward.append(avg_reward)
        self.win_rate_vs_random.append(
            win_rate_vs_random if win_rate_vs_random is not None else float('nan')
        )
        self.win_rate_vs_greedy.append(
            win_rate_vs_greedy if win_rate_vs_greedy is not None else float('nan')
        )

    def save_to_json(self, filepath: str) -> None:
        """Save metrics to JSON file for later analysis."""
        # Convert to dict, handling NaN values for JSON compatibility
        metrics_dict = asdict(self)
        for key, values in metrics_dict.items():
            metrics_dict[key] = [
                None if (isinstance(v, float) and np.isnan(v)) else v
                for v in values
            ]
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger(f"Metrics saved to {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)
        # Convert None back to NaN
        for key, values in metrics_dict.items():
            metrics_dict[key] = [
                float('nan') if v is None else v
                for v in values
            ]
        metrics = cls()
        for key, values in metrics_dict.items():
            setattr(metrics, key, values)
        return metrics


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.

    Schedule:
        - For step < warmup_steps: lr = lr_initial * (step / warmup_steps)
        - For step >= warmup_steps: lr = lr_min + 0.5 * (lr_initial - lr_min) *
                                         (1 + cos(pi * (step - warmup_steps) / decay_steps))

    Reference: Loshchilov & Hutter (2017) - SGDR: Stochastic Gradient Descent with Warm Restarts
    """

    def __init__(
            self,
            lr_initial: float,
            lr_min: float,
            decay_steps: int,
            warmup_steps: int = 0,
            name: str = "WarmupCosineDecay"
    ):
        """
        Initialize warmup cosine decay schedule.

        Args:
            lr_initial: Initial (maximum) learning rate after warmup.
            lr_min: Minimum learning rate at end of decay.
            decay_steps: Number of steps for cosine decay (after warmup).
            warmup_steps: Number of steps for linear warmup (default: 0, no warmup).
            name: Name of the schedule.
        """
        super().__init__()
        self.lr_initial = lr_initial
        self.lr_min = lr_min
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.name = name

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Compute learning rate at given step."""
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Linear warmup phase
        warmup_lr = self.lr_initial * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay phase (step adjusted for warmup)
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * decay_step / decay_steps))
        decay_lr = self.lr_min + (self.lr_initial - self.lr_min) * cosine_decay

        # Use warmup LR if in warmup phase, else use decay LR
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

    def get_config(self) -> typing.Dict[str, typing.Any]:
        """Return config for serialization."""
        return {
            "lr_initial": self.lr_initial,
            "lr_min": self.lr_min,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name
        }


def policy_gradient_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Policy gradient loss for REINFORCE algorithm.

    Computes: -sum(advantage * log(pi(a|s))) where pi(a|s) is the action probability.

    Args:
        y_true: Advantage-weighted one-hot actions of shape [batch, n_actions].
                Non-zero values represent advantages for the taken actions.
        y_pred: Predicted action probabilities (softmax output) of shape [batch, n_actions].

    Returns:
        Scalar mean loss across the batch.

    Mathematical formulation:
        L = -E[A(s,a) * log(pi(a|s))]
        where A(s,a) is the advantage and pi(a|s) is the policy probability.
    """
    # Clip predictions to avoid log(0) numerical instability
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)

    # Compute log probabilities
    log_probs = tf.math.log(y_pred_clipped)

    # Policy gradient: -advantage * log(prob_of_action_taken)
    # y_true contains (advantage * one_hot), so element-wise multiply selects the right action
    weighted_log_probs = y_true * log_probs

    # Sum over actions (only the taken action contributes due to one-hot structure)
    # Negate because we want to maximize expected return (minimize negative)
    loss_per_sample = -tf.reduce_sum(weighted_log_probs, axis=-1)

    # Return scalar mean loss for gradient computation
    return tf.reduce_mean(loss_per_sample)


def compute_discounted_returns(
        rewards: np.ndarray,
        gamma: float = 0.99,
        normalize: bool = True
) -> np.ndarray:
    """
    Compute discounted returns for policy gradient.

    For a sequence of T actions leading to a final reward R:
    G_T = R, G_{T-1} = gamma * R, G_{T-2} = gamma^2 * R, etc.

    Args:
        rewards: Array of shape (n_timesteps,) or (n_timesteps, n_actions) with
                reward assigned to each action. For 2D arrays, each row is a timestep.
        gamma: Discount factor for temporal credit assignment
        normalize: Whether to apply advantage normalization

    Returns:
        Discounted and optionally normalized returns with same shape as input.
    """
    n_timesteps = rewards.shape[0]
    if n_timesteps == 0:
        return rewards

    # Apply discount factors: later actions (higher index) get full reward,
    # earlier actions get discounted reward
    discount_factors = np.array([gamma ** (n_timesteps - 1 - i) for i in range(n_timesteps)])

    # Handle both 1D and 2D reward arrays
    if rewards.ndim == 1:
        discounted_returns = rewards * discount_factors
    else:
        # For 2D arrays, broadcast discount over columns
        discounted_returns = rewards * discount_factors[:, np.newaxis]

    # Advantage normalization for stable training
    if normalize and discounted_returns.size > 1:
        std = discounted_returns.std()
        if std > 1e-8:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (std + 1e-8)

    return discounted_returns


def collect_round_experiences(
        game: Game
) -> typing.Tuple[
    typing.List[int],
    int,
    typing.Optional[np.ndarray],
    typing.Optional[typing.List[np.ndarray]],
    typing.Optional[np.ndarray],
    typing.Optional[typing.List[np.ndarray]]
]:
    """
    Play a round and collect experiences for the replay buffer.

    Args:
        game: The game instance to play.

    Returns:
        Tuple of (after_score, n_turns, states_1, rewards_1, states_2, rewards_2).
        States and rewards are None if no turns were played.
    """
    previous_scores = [player.score for player in game.players]
    choices_1, states_1 = game.round(printing=False)
    choices_1 = to_hot_encoded(choices_1, game.n_plates + 1, config.n_colors, config.n_colors)

    choices_2, states_2 = game.end_of_round(printing=False)
    choices_2 = np.expand_dims(np.array(choices_2), axis=1)

    choices_2 = to_hot_encoded(choices_2, *(config.n_colors for i in range(config.n_colors)))

    after_score = [player.score for player in game.players]

    score_increase = [
        after_score - previous_score
        for previous_score, after_score in zip(previous_scores, after_score)
    ]

    start_index = game.players[0].start_input_index
    states_1 = np.concatenate(states_1)[:, start_index:]
    states_2 = np.array(states_2)[:, start_index:]
    n_turns = states_1.shape[0]

    if n_turns == 0:
        return after_score, n_turns, None, None, None, None

    rewards_1 = [[], [], []]
    rewards_2 = [[], [], [], [], []]
    for i, reward in enumerate(score_increase):
        for j, subchoices in enumerate(choices_1):
            rewards_1[j].append(subchoices[i] * reward)
        for j, subchoices in enumerate(choices_2):
            rewards_2[j].append(subchoices[i] * reward)

    # Concatenate rewards and apply discounted returns with advantage normalization
    for j in range(len(choices_1)):
        raw_rewards = np.concatenate(rewards_1[j])
        rewards_1[j] = compute_discounted_returns(raw_rewards, gamma=config.gamma, normalize=True)
    for j in range(len(choices_2)):
        raw_rewards = np.concatenate(rewards_2[j])
        rewards_2[j] = compute_discounted_returns(raw_rewards, gamma=config.gamma, normalize=True)

    return after_score, n_turns, states_1, rewards_1, states_2, rewards_2


def train_from_buffer(
        model: tf.keras.models.Model,
        buffer: ReplayBuffer,
        batch_size: int
) -> float:
    """
    Sample a batch from the replay buffer and train the model.

    Args:
        model: The Keras model to train.
        buffer: The replay buffer to sample from.
        batch_size: Number of samples to draw.

    Returns:
        Mean loss across outputs.
    """
    states, targets = buffer.sample(batch_size)
    loss = np.mean(model.train_on_batch(states, targets))
    return loss


def evaluate_win_rate(
        player_kwargs: typing.Dict[str, typing.Any],
        n_games: int = 20
) -> typing.Tuple[float, float]:
    """
    Evaluate win rate against random opponents.

    Runs evaluation games where player 0 uses the trained policy (epsilon=0)
    and other players use pure random actions (epsilon=1).

    Args:
        player_kwargs: Keyword arguments for BotPlayer construction.
        n_games: Number of evaluation games to play.

    Returns:
        Tuple of (win_rate, avg_score) where win_rate is fraction of games won
        and avg_score is average final score of player 0.
    """
    wins = 0
    total_score = 0.0
    n_players = player_kwargs.get('n_players', 4)

    for _ in range(n_games):
        game = Game(n_players=n_players)

        # Player 0 uses trained policy (epsilon=0), others use random (epsilon=1)
        for i in range(n_players):
            epsilon = 0.0 if i == 0 else 1.0
            game.players[i] = BotPlayer(index=i, epsilon=epsilon, **player_kwargs)
        game.players[0].is_first = True

        # Play full game
        i_round = 0
        while not game.has_ended() and i_round < 15:
            i_round += 1
            game.round(printing=False)
            game.end_of_round(printing=False)

        # Determine winner
        scores = [p.score for p in game.players]
        max_score = max(scores)
        total_score += scores[0]

        # Win if player 0 has the highest score (ties count as win)
        if scores[0] == max_score:
            wins += 1

    win_rate = wins / n_games
    avg_score = total_score / n_games

    return win_rate, avg_score


def evaluate_vs_baseline(
        bot_player: BotPlayer,
        baseline_type: str,
        n_games: int = 20,
        n_opponents: int = 3
) -> float:
    """
    Evaluate a trained bot against baseline agents (RandomAgent or GreedyAgent).

    The bot plays as player 0 against n_opponents baseline agents.

    Args:
        bot_player: The trained BotPlayer to evaluate (provides model weights).
        baseline_type: Type of baseline opponent ('random' or 'greedy').
        n_games: Number of evaluation games to play.
        n_opponents: Number of opponent agents (total players = n_opponents + 1).

    Returns:
        float: Win rate of the bot against the baseline agents.
    """
    wins = 0
    n_players = n_opponents + 1

    for _ in range(n_games):
        game = Game(n_players=n_players)

        # Create a fresh bot player for evaluation with epsilon=0 (greedy policy)
        eval_bot = BotPlayer(
            index=0,
            n_plates=game.n_plates,
            n_players=n_players,
            input_length=len(game.get_state()),
            start_input_index=config.start_input_index,
            epsilon=0.0  # Use greedy policy for evaluation
        )
        # Share model weights with the trained bot
        eval_bot._choose_model = bot_player.choose_model
        eval_bot._end_of_round_model = bot_player.end_of_round_model
        game.players[0] = eval_bot
        eval_bot.is_first = True

        # Create baseline opponents
        for i in range(1, n_players):
            if baseline_type.lower() == 'random':
                game.players[i] = RandomAgent(index=i)
            elif baseline_type.lower() == 'greedy':
                game.players[i] = GreedyAgent(index=i)
            else:
                raise ValueError(f"Unknown baseline type: {baseline_type}")

        # Play full game
        i_round = 0
        while not game.has_ended() and i_round < 15:
            i_round += 1
            game.round(printing=False)
            game.end_of_round(printing=False)

        # Determine winner
        scores = [p.score for p in game.players]
        max_score = max(scores)

        # Win if bot has the highest score (ties count as win)
        if scores[0] == max_score:
            wins += 1

    return wins / n_games


def train(
        n_games: int = 1000,
        plot_every: int = 1,
        n_players: int = 4
) -> None:
    # Setup
    dummy_game = Game(n_players=n_players)

    player_kwargs = {
        "n_plates": dummy_game.n_plates,
        "n_players": n_players,
        "input_length": len(dummy_game.get_state()),
        "start_input_index": config.start_input_index
    }

    # Create models and optimizers
    dummy_player = BotPlayer(index=-1, **player_kwargs)
    model_1 = dummy_player.choose_model
    model_2 = dummy_player.end_of_round_model

    model_1.summary()
    model_2.summary()

    # Create learning rate schedule with warmup and cosine decay
    lr_schedule = WarmupCosineDecay(
        lr_initial=config.lr_initial,
        lr_min=config.lr_min,
        decay_steps=config.lr_decay_steps,
        warmup_steps=config.lr_warmup_steps
    )
    logger(f"LR schedule: warmup={config.lr_warmup_steps} steps, "
           f"decay={config.lr_decay_steps} steps, "
           f"lr_initial={config.lr_initial}, lr_min={config.lr_min}")

    # Create optimizers with shared LR schedule
    optimizer_1 = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    optimizer_2 = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)

    model_1.compile(loss=policy_gradient_loss, optimizer=optimizer_1)
    model_2.compile(loss=policy_gradient_loss, optimizer=optimizer_2)

    # Initialize replay buffers for both models
    buffer_1 = ReplayBuffer(capacity=config.replay_buffer_size)
    buffer_2 = ReplayBuffer(capacity=config.replay_buffer_size)
    logger(f"Initialized replay buffers with capacity={config.replay_buffer_size}, "
           f"batch_size={config.batch_size}, min_buffer_size={config.min_buffer_size}")

    # Initialize comprehensive metrics tracking
    metrics = TrainingMetrics()

    # Track epsilon across games (shared state)
    current_epsilon = config.epsilon_start
    total_rounds = 0

    for game_index in range(n_games):
        logger(f"Starting game_index {game_index + 1}/{n_games} (epsilon={current_epsilon:.4f}, "
               f"buffer_1={len(buffer_1)}, buffer_2={len(buffer_2)})")
        i_turn = 0
        i_round = 0
        game = Game(n_players=n_players)
        for i in range(game.n_players):
            game.players[i] = BotPlayer(index=i, epsilon=current_epsilon, **player_kwargs)
            game.players[i].reset_metrics()  # Reset per-game metrics
        game.players[0].is_first = True

        game_loss1 = 0.0
        game_loss2 = 0.0
        train_count = 0
        game_reward_sum = 0.0

        while True:
            i_round += 1
            total_rounds += 1

            # Collect experiences from this round
            after_score, round_turns, states_1, rewards_1, states_2, rewards_2 = \
                collect_round_experiences(game)

            i_turn += round_turns

            # Track rewards for this round (average over all reward components)
            if rewards_1 is not None:
                for r in rewards_1:
                    game_reward_sum += np.sum(np.abs(r))

            # Store experiences in buffers
            if states_1 is not None and rewards_1 is not None:
                buffer_1.push(states_1, rewards_1)
            if states_2 is not None and rewards_2 is not None:
                buffer_2.push(states_2, rewards_2)

            # Train from buffer if ready and at training interval
            # Each model trains independently when its buffer has enough samples
            if total_rounds % config.train_every_n_rounds == 0:
                if buffer_1.is_ready(config.min_buffer_size):
                    loss1 = train_from_buffer(model_1, buffer_1, config.batch_size)
                    game_loss1 += loss1
                    train_count += 1
                    logger(f"  Model 1 trained from buffer (size={len(buffer_1)}): loss={loss1:.4f}")
                if buffer_2.is_ready(config.min_buffer_size):
                    loss2 = train_from_buffer(model_2, buffer_2, config.batch_size)
                    game_loss2 += loss2
                    logger(f"  Model 2 trained from buffer (size={len(buffer_2)}): loss={loss2:.4f}")

            if game.has_ended():
                break

            if i_round > 15:
                break

        # Collect invalid action rates from player 0 (the bot we're training)
        invalid_row_rate, invalid_col_rate = game.players[0].get_invalid_rates()

        # Get current learning rate from schedule (use optimizer_1's step count)
        current_lr = float(lr_schedule(optimizer_1.iterations))

        # Evaluate win rate periodically
        win_rate = None
        win_rate_vs_random = None
        win_rate_vs_greedy = None
        if (game_index + 1) % config.eval_every_n_games == 0:
            logger(f"Evaluating at game {game_index + 1}...")

            # Evaluate vs random BotPlayer opponents (legacy)
            win_rate, eval_score = evaluate_win_rate(player_kwargs, config.eval_n_games)
            logger(f"  vs Random BotPlayer: {win_rate:.2%}, Avg score: {eval_score:.1f}")

            # Evaluate vs baseline agents
            win_rate_vs_random = evaluate_vs_baseline(
                dummy_player, 'random', n_games=config.eval_n_games
            )
            logger(f"  vs RandomAgent: {win_rate_vs_random:.2%}")

            win_rate_vs_greedy = evaluate_vs_baseline(
                dummy_player, 'greedy', n_games=config.eval_n_games
            )
            logger(f"  vs GreedyAgent: {win_rate_vs_greedy:.2%}")

        # Record metrics for this episode
        metrics.record_episode(
            episode_num=game_index + 1,
            loss_choose=game_loss1 / max(train_count, 1),
            loss_eor=game_loss2 / max(train_count, 1),
            avg_score=float(np.nanmean(after_score)),
            game_length=i_round,
            n_turns=i_turn,
            exploration_rate=current_epsilon,
            learning_rate=current_lr,
            buffer_size_choose=len(buffer_1),
            buffer_size_eor=len(buffer_2),
            win_rate=win_rate,
            invalid_row_rate=invalid_row_rate,
            invalid_col_rate=invalid_col_rate,
            avg_reward=game_reward_sum / max(i_turn, 1),
            win_rate_vs_random=win_rate_vs_random,
            win_rate_vs_greedy=win_rate_vs_greedy
        )

        # Decay epsilon for the next game
        current_epsilon = max(config.epsilon_end, current_epsilon * config.epsilon_decay)

        if game_index % plot_every == 0 and game_index > 0:
            plot_metrics_dashboard(metrics)
            metrics.save_to_json(config.metrics_path)

    # Final save
    logger("Saving model weights and metrics...")
    model_1.save_weights(config.model_weights_path_1)
    model_2.save_weights(config.model_weights_path_2)
    metrics.save_to_json(config.metrics_path)
    plot_metrics_dashboard(metrics)
    logger(f"Final buffer sizes: buffer_1={len(buffer_1)}, buffer_2={len(buffer_2)}")


def plot_metrics_dashboard(metrics: TrainingMetrics) -> None:
    """
    Create a comprehensive visualization dashboard for training metrics.

    Displays a 3x4 grid of plots showing:
    - Performance: scores, win rate, rewards
    - Losses: choose model, EOR model
    - Training dynamics: exploration, learning rate
    - Invalid action rates and game dynamics

    Args:
        metrics: TrainingMetrics instance containing all tracked data.
    """
    n_episodes = len(metrics.episode)
    if n_episodes == 0:
        return

    display = Display(nrows=3, ncols=4, suptitle=f"Training Metrics - {n_episodes} episodes")

    kwargs = dict(xlabel="Episode")
    x = metrics.episode

    # Row 1: Performance metrics
    display[0].plot(x, metrics.avg_score, ylabel="Score", title="Average Score", **kwargs)

    # Win rate vs baselines (filter out NaN for plotting)
    valid_wr_random = [
        (ep, wr) for ep, wr in zip(x, metrics.win_rate_vs_random) if not np.isnan(wr)
    ]
    valid_wr_greedy = [
        (ep, wr) for ep, wr in zip(x, metrics.win_rate_vs_greedy) if not np.isnan(wr)
    ]
    if valid_wr_random or valid_wr_greedy:
        if valid_wr_random:
            wr_x, wr_y = zip(*valid_wr_random)
            display[1].ax.plot(wr_x, wr_y, 'b-', label='vs Random')
        if valid_wr_greedy:
            wr_x, wr_y = zip(*valid_wr_greedy)
            display[1].ax.plot(wr_x, wr_y, 'r-', label='vs Greedy')
        display[1].ax.set_ylim(0, 1)
        display[1].ax.set_xlabel("Episode")
        display[1].ax.set_ylabel("Win Rate")
        display[1].ax.set_title("Win Rate vs Baselines")
        display[1].ax.legend()
        display[1].ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    else:
        display[1].plot([], [], ylabel="Win Rate", title="Win Rate (pending)", **kwargs)

    display[2].plot(x, metrics.avg_reward, ylabel="Reward", title="Average Reward", **kwargs)
    display[3].plot(x, metrics.n_turns, ylabel="Turns", title="Turns per Game", **kwargs)

    # Row 2: Loss and training parameters
    display[4].plot(x, metrics.loss_choose, ylabel="Loss", title="Choose Model Loss", **kwargs)
    display[5].plot(x, metrics.loss_eor, ylabel="Loss", title="EOR Model Loss", **kwargs)
    display[6].plot(x, metrics.exploration_rate, ylabel="Epsilon", title="Exploration Rate", **kwargs)
    display[7].plot(x, metrics.learning_rate, ylabel="LR", title="Learning Rate", **kwargs)

    # Row 3: Invalid actions, game dynamics, buffer sizes
    display[8].plot(x, metrics.invalid_row_rate, ylabel="Rate", title="Invalid Row Rate", **kwargs)
    display[9].plot(x, metrics.invalid_col_rate, ylabel="Rate", title="Invalid Col Rate", **kwargs)
    display[10].plot(x, metrics.game_length, ylabel="Rounds", title="Game Length", **kwargs)
    display[11].plot(
        x, metrics.buffer_size_choose,
        ylabel="Size", title="Buffer Sizes", **kwargs
    )

    display.show(export_filename=config.history_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--plot_every', type=int, default=1)
    args = parser.parse_args()

    train(args.n_games, args.plot_every)
