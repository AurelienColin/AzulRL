"""Generate baseline agent performance report.

Runs comprehensive evaluation of baseline agents against each other
and produces a summary report for documentation.
"""
import typing
import numpy as np
import sys
import io
from contextlib import redirect_stdout

from src.scripts.evaluate import (
    evaluate_agents,
    compare_baselines,
    EvaluationResult
)
from rignak.src.logging_utils import logger


def generate_baseline_report(n_games: int = 100) -> str:
    """
    Generate a comprehensive report comparing baseline agents.

    Args:
        n_games: Number of games per comparison.

    Returns:
        str: Formatted report text.
    """
    lines = [
        "=" * 60,
        "BASELINE AGENT PERFORMANCE REPORT",
        "=" * 60,
        "",
        f"Evaluation: {n_games} games per matchup",
        "4-player games, First player advantage normalized",
        "",
    ]

    # Suppress game output during evaluation
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Run evaluations
        logger("Running Random vs Random evaluation...")
        random_vs_random = evaluate_agents(
            ["random", "random", "random", "random"],
            n_games=n_games
        )

        logger("Running Greedy vs Random evaluation...")
        greedy_vs_random = evaluate_agents(
            ["greedy", "random", "random", "random"],
            n_games=n_games
        )

        logger("Running Greedy vs Greedy evaluation...")
        greedy_vs_greedy = evaluate_agents(
            ["greedy", "greedy", "greedy", "greedy"],
            n_games=n_games
        )
    finally:
        sys.stdout = old_stdout

    # Format results
    lines.extend([
        "-" * 60,
        "RANDOM vs RANDOM (4 players)",
        "-" * 60,
        "",
        "Expected: ~25% win rate for each player (uniform random)",
        "",
    ])
    for name in random_vs_random.agent_names:
        wr = random_vs_random.win_rates[name]
        avg = random_vs_random.avg_scores[name]
        std = random_vs_random.std_scores[name]
        lines.append(f"  {name}: Win Rate={wr:.1%}, Avg Score={avg:.1f} (+/- {std:.1f})")
    lines.append("")

    lines.extend([
        "-" * 60,
        "GREEDY vs RANDOM (1 Greedy + 3 Random)",
        "-" * 60,
        "",
        "Expected: Greedy should win significantly more than 25%",
        "",
    ])
    for name in greedy_vs_random.agent_names:
        wr = greedy_vs_random.win_rates[name]
        avg = greedy_vs_random.avg_scores[name]
        std = greedy_vs_random.std_scores[name]
        lines.append(f"  {name}: Win Rate={wr:.1%}, Avg Score={avg:.1f} (+/- {std:.1f})")
    lines.append("")

    lines.extend([
        "-" * 60,
        "GREEDY vs GREEDY (4 players)",
        "-" * 60,
        "",
        "Expected: ~25% win rate for each player (all same strategy)",
        "",
    ])
    for name in greedy_vs_greedy.agent_names:
        wr = greedy_vs_greedy.win_rates[name]
        avg = greedy_vs_greedy.avg_scores[name]
        std = greedy_vs_greedy.std_scores[name]
        lines.append(f"  {name}: Win Rate={wr:.1%}, Avg Score={avg:.1f} (+/- {std:.1f})")
    lines.append("")

    # Summary section
    lines.extend([
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        "",
        "Performance Hierarchy (expected after RL training):",
        "  1. Trained RL Agent",
        "  2. Greedy Agent (heuristic-based)",
        "  3. Random Agent (baseline)",
        "",
        "Target Win Rates for Trained RL Agent:",
        "  - vs Random: > 80%",
        "  - vs Greedy: > 60%",
        "",
        "Greedy Agent Analysis:",
        f"  - Win Rate vs Random: {greedy_vs_random.win_rates['Greedy_0']:.1%}",
        "  - This establishes the ceiling a simple heuristic can achieve",
        "",
        "Key Metrics for Training Success:",
        "  1. RL win rate vs Random increases over training",
        "  2. RL win rate vs Greedy increases over training",
        "  3. RL eventually surpasses Greedy performance",
        "",
    ])

    return "\n".join(lines)


def save_report(filepath: str, n_games: int = 100) -> None:
    """
    Generate and save baseline report to file.

    Args:
        filepath: Path to save the report.
        n_games: Number of games per comparison.
    """
    report = generate_baseline_report(n_games)
    with open(filepath, 'w') as f:
        f.write(report)
    logger(f"Report saved to {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate baseline agent performance report")
    parser.add_argument('--n_games', type=int, default=100, help='Games per comparison')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    report = generate_baseline_report(args.n_games)
    print(report)

    if args.output:
        save_report(args.output, args.n_games)
