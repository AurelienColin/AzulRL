import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import typing
from rignak.src.logging_utils import logger
from src.obj.game import Game
from src.obj.bot_player import BotPlayer
from src.obj.replay_buffer import ReplayBuffer
from src.config import config
from src.utils import to_hot_encoded
from rignak.src.custom_display import Display


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


def train(
        n_games: int = 1000,
        learning_rate: float = 0.1,
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

    model_1.compile(loss=policy_gradient_loss, optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate))
    model_2.compile(loss=policy_gradient_loss, optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate))

    # Initialize replay buffers for both models
    buffer_1 = ReplayBuffer(capacity=config.replay_buffer_size)
    buffer_2 = ReplayBuffer(capacity=config.replay_buffer_size)
    logger(f"Initialized replay buffers with capacity={config.replay_buffer_size}, "
           f"batch_size={config.batch_size}, min_buffer_size={config.min_buffer_size}")

    scores = []
    n_turns = []
    n_rounds = []
    losses1 = []
    losses2 = []
    epsilons = []
    buffer_sizes = []

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
        game.players[0].is_first = True

        game_loss1 = 0.0
        game_loss2 = 0.0
        train_count = 0

        while True:
            i_round += 1
            total_rounds += 1

            # Collect experiences from this round
            after_score, round_turns, states_1, rewards_1, states_2, rewards_2 = \
                collect_round_experiences(game)

            i_turn += round_turns

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

        scores.append(np.nanmean(after_score))
        n_turns.append(i_turn)
        n_rounds.append(i_round)
        # Average loss per training step, or 0 if no training occurred
        losses1.append(game_loss1 / max(train_count, 1))
        losses2.append(game_loss2 / max(train_count, 1))
        epsilons.append(current_epsilon)
        buffer_sizes.append((len(buffer_1), len(buffer_2)))

        # Decay epsilon for the next game
        current_epsilon = max(config.epsilon_end, current_epsilon * config.epsilon_decay)

        if game_index % plot_every == 0 and game_index > 0:
            plot_history(scores, n_turns, n_rounds, losses1, losses2, epsilons)

    # Save weights
    logger("Saving model weights...")
    model_1.save_weights(config.model_weights_path_1)
    model_2.save_weights(config.model_weights_path_2)
    logger(f"Final buffer sizes: buffer_1={len(buffer_1)}, buffer_2={len(buffer_2)}")


def plot_history(
        mean_scores: typing.List[float],
        n_turns: typing.List[int],
        n_rounds: typing.List[int],
        losses1: typing.List[float],
        losses2: typing.List[float],
        epsilons: typing.List[float]
) -> None:
    display = Display(ncols=6, suptitle=f"History after {len(mean_scores)} games.")

    kwargs = dict(xlabel="Game index")
    x = range(len(mean_scores))
    display[0].plot(x, mean_scores, ylabel="Score", title="Mean score after the game.", **kwargs)
    display[1].plot(x, n_turns, ylabel="n", title="Number of player turns.", **kwargs)
    display[2].plot(x, n_rounds, ylabel="n", title="Number of game rounds.", **kwargs)
    display[3].plot(x, losses1, ylabel="Score", title="In-turn loss", **kwargs)
    display[4].plot(x, losses2, ylabel="Score", title="End-turn loss", **kwargs)
    display[5].plot(x, epsilons, ylabel="Epsilon", title="Exploration rate decay", **kwargs)
    display.show(export_filename=config.history_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--plot_every', type=int, default=1)
    args = parser.parse_args()

    train(args.n_games, args.learning_rate, args.plot_every)
