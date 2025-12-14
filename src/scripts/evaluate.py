"""Evaluation harness for comparing Azul agents.

Runs multiple games between agents and computes performance statistics.
Used for baseline comparison and tracking RL training progress.
"""
import typing
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy

from src.config import config
from src.obj.game import Game
from src.obj.player import Player
from src.obj.random_agent import RandomAgent
from src.obj.greedy_agent import GreedyAgent

from rignak.src.logging_utils import logger


@dataclass
class EvaluationResult:
    """Results from evaluating agents against each other."""
    n_games: int
    agent_names: typing.List[str]
    wins: typing.Dict[str, int] = field(default_factory=dict)
    win_rates: typing.Dict[str, float] = field(default_factory=dict)
    avg_scores: typing.Dict[str, float] = field(default_factory=dict)
    std_scores: typing.Dict[str, float] = field(default_factory=dict)
    score_margins: typing.List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            f"Evaluation Results ({self.n_games} games)",
            "-" * 40,
        ]
        for name in self.agent_names:
            lines.append(
                f"{name}: Win Rate={self.win_rates.get(name, 0):.1%}, "
                f"Avg Score={self.avg_scores.get(name, 0):.1f} "
                f"(+/- {self.std_scores.get(name, 0):.1f})"
            )
        return "\n".join(lines)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_games": self.n_games,
            "agent_names": self.agent_names,
            "wins": self.wins,
            "win_rates": self.win_rates,
            "avg_scores": self.avg_scores,
            "std_scores": self.std_scores,
            "avg_margin": float(np.mean(self.score_margins)) if self.score_margins else 0.0,
        }


def create_agent(agent_type: str, index: int) -> Player:
    """
    Factory function to create agents by type name.

    Args:
        agent_type: One of 'random', 'greedy', or 'human'.
        index: Player index (0-based).

    Returns:
        Player: An agent instance.

    Raises:
        ValueError: If agent_type is not recognized.
    """
    agent_type = agent_type.lower()
    if agent_type == "random":
        return RandomAgent(index=index)
    elif agent_type == "greedy":
        return GreedyAgent(index=index)
    elif agent_type == "human":
        return Player(index=index)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_game(
    players: typing.List[Player],
    printing: bool = False
) -> typing.Dict[int, int]:
    """
    Run a single game with the given players.

    Args:
        players: List of Player instances.
        printing: Whether to print game state during play.

    Returns:
        Dict mapping player index to final score.
    """
    n_players = len(players)

    # Create fresh game
    game = Game(n_players=n_players)

    # Replace default players with our agents
    game._players = players
    for i, player in enumerate(game.players):
        player.index = i
        if i == 0:
            player.is_first = True

    # Play rounds until game ends
    max_rounds = 20  # Safety limit
    for round_num in range(max_rounds):
        try:
            game.round(printing=printing)
            game.end_of_round(printing=printing)

            if game.has_ended():
                break
        except Exception as e:
            logger.warning(f"Game error in round {round_num}: {e}")
            break

    # Collect final scores
    scores = {player.index: player.score for player in game.players}
    return scores


def evaluate_agents(
    agent_types: typing.List[str],
    n_games: int = 100,
    printing: bool = False
) -> EvaluationResult:
    """
    Evaluate multiple agents against each other over many games.

    Args:
        agent_types: List of agent type names (e.g., ['random', 'greedy']).
        n_games: Number of games to play.
        printing: Whether to print game state during play.

    Returns:
        EvaluationResult: Statistics from the evaluation.
    """
    n_players = len(agent_types)
    agent_names = [f"{t.capitalize()}_{i}" for i, t in enumerate(agent_types)]

    # Initialize tracking
    wins = {name: 0 for name in agent_names}
    all_scores = {name: [] for name in agent_names}
    score_margins = []

    for game_idx in range(n_games):
        # Create fresh agents for each game
        players = [create_agent(t, i) for i, t in enumerate(agent_types)]

        # Run game
        scores = run_game(players, printing=printing)

        # Find winner
        max_score = max(scores.values())
        winners = [i for i, s in scores.items() if s == max_score]

        # Record results
        for i, name in enumerate(agent_names):
            all_scores[name].append(scores.get(i, 0))
            if i in winners:
                wins[name] += 1 / len(winners)  # Split ties

        # Calculate margin (first agent's perspective)
        if len(scores) >= 2:
            margin = scores.get(0, 0) - max(
                scores.get(i, 0) for i in range(1, n_players)
            )
            score_margins.append(margin)

    # Calculate statistics
    win_rates = {name: wins[name] / n_games for name in agent_names}
    avg_scores = {name: np.mean(all_scores[name]) for name in agent_names}
    std_scores = {name: np.std(all_scores[name]) for name in agent_names}

    return EvaluationResult(
        n_games=n_games,
        agent_names=agent_names,
        wins=wins,
        win_rates=win_rates,
        avg_scores=avg_scores,
        std_scores=std_scores,
        score_margins=score_margins,
    )


def evaluate_bot_vs_baseline(
    bot_player: Player,
    baseline_type: str = "random",
    n_games: int = 20,
    n_opponents: int = 3,
    printing: bool = False
) -> typing.Dict[str, float]:
    """
    Evaluate a trained bot against baseline agents.

    The bot plays as player 0 against n_opponents baseline agents.

    Args:
        bot_player: The trained BotPlayer to evaluate.
        baseline_type: Type of baseline opponent ('random' or 'greedy').
        n_games: Number of games to play.
        n_opponents: Number of opponent agents (total players = n_opponents + 1).
        printing: Whether to print game state.

    Returns:
        Dict with 'win_rate', 'avg_score', 'avg_margin'.
    """
    wins = 0
    scores = []
    margins = []

    for game_idx in range(n_games):
        # Create players: bot at index 0, baselines for the rest
        players = [_clone_bot(bot_player, 0)]
        for i in range(1, n_opponents + 1):
            players.append(create_agent(baseline_type, i))

        # Run game
        game_scores = run_game(players, printing=printing)

        # Track results
        bot_score = game_scores.get(0, 0)
        max_other = max(game_scores.get(i, 0) for i in range(1, n_opponents + 1))

        scores.append(bot_score)
        margins.append(bot_score - max_other)

        if bot_score > max_other:
            wins += 1
        elif bot_score == max_other:
            wins += 0.5  # Tie

    return {
        "win_rate": wins / n_games,
        "avg_score": float(np.mean(scores)),
        "avg_margin": float(np.mean(margins)),
        "n_games": n_games,
        "baseline": baseline_type,
    }


def _clone_bot(bot_player: Player, new_index: int) -> Player:
    """
    Create a fresh copy of a bot player for evaluation.

    This resets the player state while preserving the model weights.

    Args:
        bot_player: The bot to clone.
        new_index: New player index.

    Returns:
        Player: A fresh bot instance with the same model.
    """
    # Import BotPlayer here to avoid circular imports
    from src.obj.bot_player import BotPlayer

    if isinstance(bot_player, BotPlayer):
        # Create new BotPlayer with same models
        new_bot = BotPlayer(index=new_index)
        # Copy model weights (models are built lazily, so copy references)
        if hasattr(bot_player, '_choose_model') and bot_player._choose_model is not None:
            new_bot._choose_model = bot_player._choose_model
        if hasattr(bot_player, '_end_of_round_model') and bot_player._end_of_round_model is not None:
            new_bot._end_of_round_model = bot_player._end_of_round_model
        # Set epsilon to 0 for evaluation (greedy action selection)
        new_bot.epsilon = 0.0
        return new_bot
    else:
        # For non-BotPlayers, create a fresh instance of the same type
        return type(bot_player)(index=new_index)


def compare_baselines(n_games: int = 100) -> typing.Dict[str, EvaluationResult]:
    """
    Compare baseline agents against each other.

    Runs pairwise comparisons between random, greedy, and random agents
    to establish baseline performance expectations.

    Args:
        n_games: Number of games per comparison.

    Returns:
        Dict mapping comparison name to EvaluationResult.
    """
    results = {}

    # Random vs Random (should be ~50/50)
    logger.info("Evaluating: Random vs Random")
    results["random_vs_random"] = evaluate_agents(
        ["random", "random", "random", "random"],
        n_games=n_games
    )

    # Greedy vs Random
    logger.info("Evaluating: Greedy vs Random")
    results["greedy_vs_random"] = evaluate_agents(
        ["greedy", "random", "random", "random"],
        n_games=n_games
    )

    # Greedy vs Greedy
    logger.info("Evaluating: Greedy vs Greedy")
    results["greedy_vs_greedy"] = evaluate_agents(
        ["greedy", "greedy", "greedy", "greedy"],
        n_games=n_games
    )

    return results


if __name__ == "__main__":
    # Run baseline comparisons
    print("Running baseline agent comparisons...")
    print("=" * 50)

    results = compare_baselines(n_games=50)

    for name, result in results.items():
        print(f"\n{name}:")
        print(result)
