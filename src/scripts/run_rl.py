import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import typing
from rignak.src.logging_utils import logger
from src.obj.game import Game
from src.obj.bot_player import BotPlayer
from src.config import config
from src.utils import to_hot_encoded
from rignak.src.custom_display import Display


def masked_mae(y_true: tf.Tensor, y_pred: tf.Tensor, espilon: float = 1E-8) -> tf.Tensor:
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32) + espilon
    squared_error = tf.abs(y_true - y_pred)
    masked_squared_error = squared_error * mask
    return masked_squared_error


def train_on_round(
        game: Game,
        model_1: tf.keras.models.Model,
        model_2: tf.keras.models.Model
):
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

    with tf.GradientTape(persistent=True) as tape:
        rewards_1 = [[], [], []]
        rewards_2 = [[], [], [], [], []]
        for i in range(game.n_players):
            reward = score_increase[i]
            for j, subchoices in enumerate(choices_1):
                rewards_1[j].append(subchoices[i] * reward)
            for j, subchoices in enumerate(choices_2):
                rewards_2[j].append(subchoices[i] * reward)

        for j in range(len(choices_1)):
            rewards_1[j] = np.concatenate(rewards_1[j])
        for j in range(len(choices_2)):
            rewards_2[j] = np.concatenate(rewards_2[j])

        start_index = game.players[0].start_input_index
        states_1 = np.concatenate(states_1)[:, start_index:]
        states_2 = np.array(states_2)[:, start_index:]

        logger(f"Train with {states_1.shape[0]} samples.")
        loss1 = loss2 = None
        if states_1.shape[0]:
            loss1 = np.mean(model_1.train_on_batch(states_1, rewards_1))
            loss2 = np.mean(model_2.train_on_batch(states_2, rewards_2))
    return after_score, rewards_1[0].shape[0], loss1, loss2


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
        "input_length": len(dummy_game.get_state()),
        "start_input_index": 2 * config.n_colors
    }

    # Create models and optimizers
    dummy_player = BotPlayer(index=-1, **player_kwargs)
    model_1 = dummy_player.choose_model
    model_2 = dummy_player.end_of_round_model

    model_1.summary()
    model_2.summary()

    model_1.compile(loss=masked_mae, optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate))
    model_2.compile(loss=masked_mae, optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate))

    scores = []
    n_turns = []
    n_rounds = []
    losses1 = []
    losses2 = []

    for game_index in range(n_games):
        logger(f"Starting game_index {game_index + 1}/{n_games}")
        i_turn = 0
        i_round = 0
        game = Game(n_players=n_players)
        for i in range(game.n_players):
            game.players[i] = BotPlayer(index=i, **player_kwargs)
        game.players[0].is_first = True

        total_loss1 = 0
        total_loss2 = 0
        while True:
            i_round += 1
            after_score, round_turns, loss1, loss2 = train_on_round(game, model_1, model_2)
            total_loss1 += loss1
            total_loss2 += loss2
            i_turn += round_turns
            if game.has_ended():
                break

            if i_round > 15:
                break

        scores.append(np.nanmean(after_score))
        n_turns.append(i_turn)
        n_rounds.append(i_round)
        losses1.append(total_loss1)
        losses2.append(total_loss1)

        if game_index % plot_every == 0 and game_index > 0:
            plot_history(scores, n_turns, n_rounds, losses1, losses2)

    # Save weights
    logger("Saving model weights...")
    model_1.save_weights(config.model_weights_path_1)
    model_2.save_weights(config.model_weights_path_2)


def plot_history(
        mean_scores: typing.List,
        n_turns: typing.List,
        n_rounds: typing.List,
        losses1: typing.List,
        losses2: typing.List
) -> None:
    display = Display(ncols=5, suptitle=f"History after {len(mean_scores)} games.")

    kwargs = dict(xlabel="Game index")
    x = range(len(mean_scores))
    display[0].plot(x, mean_scores, ylabel="Score", title="Mean score after the game.", **kwargs)
    display[1].plot(x, n_turns, ylabel="n", title="Number of player turns.", **kwargs)
    display[2].plot(x, n_rounds, ylabel="n", title="Number of game rounds.", **kwargs)
    display[3].plot(x, losses1, ylabel="Score", title="In-turn loss", **kwargs)
    display[4].plot(x, losses2, ylabel="Score", title="End-turn loss", **kwargs)
    display.show(export_filename=config.history_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--plot_every', type=int, default=1)
    args = parser.parse_args()

    train(args.n_games, args.learning_rate, args.plot_every)
