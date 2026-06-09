"""
Sample cl input:
python testing/evaluate_agents.py --agents random checkerboard heatmap bayesian_mc --games 10 --seed 42

Use --fresh to clear and overwrite old csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List

TESTING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTING_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_RESULTS_DIR = TESTING_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battleship.engine import run_single_game
from battleship.types import GameConfig
from battleship.agent_random import RandomAgent
from battleship.agent_checkerboard import CheckerboardAgent
from battleship.agent_heatmap import ProbabilityHeatmapAgent

try:
    from battleship.agent_bayesianMC import BayesianMCAgent
except ImportError:
    BayesianMCAgent = None

def make_random_agent(seed: int):
    return RandomAgent(seed=seed)


def make_checkerboard_agent(seed: int):
    return CheckerboardAgent()


def make_heatmap_agent(seed: int):
    return ProbabilityHeatmapAgent()


def make_bayesian_mc_agent(seed: int):
    return BayesianMCAgent(seed=seed)

AGENT_FACTORIES: Dict[str, Callable[[int], object]] = {
    "random": make_random_agent,
    "checkerboard": make_checkerboard_agent,
    "heatmap": make_heatmap_agent,
    "bayesian_mc": make_bayesian_mc_agent,
}

PER_GAME_FIELDS = [
    "run_id",
    "agent_key",
    "agent_name",
    "game_number",
    "board_seed",
    "agent_seed",
    "total_attacks",
    "board_size",
    "ship_sizes",
]

SUMMARY_FIELDS = [
    "agent_key",
    "agent_name",
    "num_games",
    "mean_attacks",
    "median_attacks",
    "std_attacks",
    "min_attacks",
    "max_attacks",
]


def normalize_key(key: str) -> str:
    return key.strip().lstrip("\ufeff").lower()


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: List[dict] = []

        for row in reader:
            normalized = {
                normalize_key(k): v
                for k, v in row.items()
                if k is not None
            }
            rows.append(normalized)

    return rows


def append_csv_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    file_has_content = file_exists and path.stat().st_size > 0

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_has_content:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def get_existing_game_counts(per_game_rows: Iterable[dict]) -> Dict[str, int]:
    """
    Return how many games already exist for each agent.

    This lets incremental runs continue game numbering.
    """
    counts: Dict[str, int] = defaultdict(int)

    for row in per_game_rows:
        agent_key = row.get("agent_key") or row.get("agent") or row.get("agent_name")
        if agent_key is None:
            continue

        counts[agent_key] += 1

    return dict(counts)


def get_next_run_id(per_game_rows: Iterable[dict]) -> int:
    """
    Determine the next run_id.

    If no old rows exist, start at 1. Otherwise, use max existing run_id + 1.
    """
    run_ids: List[int] = []

    for row in per_game_rows:
        raw = row.get("run_id")
        if raw is None or raw == "":
            continue

        try:
            run_ids.append(int(raw))
        except ValueError:
            continue

    if not run_ids:
        return 1

    return max(run_ids) + 1


def run_one_game(*, agent_key: str, agent_seed: int, board_seed: int, board_size: int,
        ship_sizes: tuple[int, ...]) -> dict:

    if agent_key not in AGENT_FACTORIES:
        valid = ", ".join(sorted(AGENT_FACTORIES))
        raise ValueError(f"Unknown agent '{agent_key}'. Valid agents: {valid}")

    agent = AGENT_FACTORIES[agent_key](agent_seed)

    config = GameConfig(
        board_size=board_size,
        ship_sizes=ship_sizes,
    )

    result = run_single_game(agent,config=config,seed=board_seed)

    if isinstance(result, int):
        total_attacks = result
    elif hasattr(result, "turns"):
        total_attacks = int(result.turns)
    elif hasattr(result, "num_turns"):
        total_attacks = int(result.num_turns)
    elif hasattr(result, "total_attacks"):
        total_attacks = int(result.total_attacks)
    else:
        raise TypeError(
            "Could not determine total attacks from run_single_game result. "
            f"Got object of type {type(result)}: {result!r}"
        )

    agent_name = getattr(agent, "name", agent_key)

    return {
        "agent_key": agent_key,
        "agent_name": agent_name,
        "board_seed": board_seed,
        "agent_seed": agent_seed,
        "total_attacks": total_attacks,
        "board_size": board_size,
        "ship_sizes": "-".join(str(size) for size in ship_sizes),
    }


def run_evaluation(*, agents: List[str], games: int, seed: int, board_size: int,
    ship_sizes: tuple[int, ...], results_dir: Path, fresh: bool) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    per_game_path = results_dir / "per_game_results.csv"
    summary_path = results_dir / "summary_results.csv"
    summary_json_path = results_dir / "summary_results.json"

    if fresh:
        for path in [
            per_game_path,
            summary_path,
            summary_json_path,
        ]:
            if path.exists():
                path.unlink()

    old_rows = read_csv_rows(per_game_path)
    existing_counts = get_existing_game_counts(old_rows)
    run_id = get_next_run_id(old_rows)

    pending_rows: List[dict] = []

    for agent_key in agents:
        if agent_key not in AGENT_FACTORIES:
            valid = ", ".join(sorted(AGENT_FACTORIES))
            raise ValueError(f"Unknown agent '{agent_key}'. Valid agents: {valid}")

        already_done = existing_counts.get(agent_key, 0)

        print(
            f"Running {games} new games for agent '{agent_key}' "
            f"(already has {already_done} recorded games)."
        )

        for local_index in range(games):
            game_number = already_done + local_index + 1

            board_seed = seed + game_number
            agent_seed = seed + 1_000_000 + game_number

            result_row = run_one_game(
                agent_key=agent_key,
                agent_seed=agent_seed,
                board_seed=board_seed,
                board_size=board_size,
                ship_sizes=ship_sizes,
            )

            result_row = {
                "run_id": run_id,
                "agent_key": result_row["agent_key"],
                "agent_name": result_row["agent_name"],
                "game_number": game_number,
                "board_seed": result_row["board_seed"],
                "agent_seed": result_row["agent_seed"],
                "total_attacks": result_row["total_attacks"],
                "board_size": result_row["board_size"],
                "ship_sizes": result_row["ship_sizes"],
            }

            pending_rows.append(result_row)

            if game_number % 100 == 0:
                append_csv_rows(per_game_path, PER_GAME_FIELDS, pending_rows)
                pending_rows.clear()
                print(f"  {agent_key}: completed game {game_number}")

        if pending_rows:
            append_csv_rows(per_game_path, PER_GAME_FIELDS, pending_rows)
            pending_rows.clear()

    all_rows = read_csv_rows(per_game_path)

    summary_rows = build_summary_rows(all_rows)

    write_csv_rows(summary_path, SUMMARY_FIELDS, summary_rows)

    with summary_json_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)

    print()
    print("Evaluation complete.")
    print(f"Per-game results:   {per_game_path}")
    print(f"Summary results:    {summary_path}")
    print(f"Summary JSON:       {summary_json_path}")



def build_summary_rows(per_game_rows: List[dict]) -> List[dict]:
    """
    Build summary statistics from all accumulated per-game results.
    """
    by_agent: Dict[str, List[dict]] = defaultdict(list)

    for row in per_game_rows:
        agent_key = row["agent_key"]
        by_agent[agent_key].append(row)

    summary_rows: List[dict] = []

    for agent_key, rows in sorted(by_agent.items()):
        attacks = [int(row["total_attacks"]) for row in rows]

        if len(attacks) >= 2:
            std_attacks = statistics.stdev(attacks)
        else:
            std_attacks = 0.0

        summary_rows.append(
            {
                "agent_key": agent_key,
                "agent_name": rows[0]["agent_name"],
                "num_games": len(attacks),
                "mean_attacks": f"{statistics.mean(attacks):.6f}",
                "median_attacks": f"{statistics.median(attacks):.6f}",
                "std_attacks": f"{std_attacks:.6f}",
                "min_attacks": min(attacks),
                "max_attacks": max(attacks),
            }
        )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Battleship agents and save CSV results."
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=["random", "checkerboard", "heatmap", "bayesian_mc"],
        help=f"Agents to evaluate. Valid options: {', '.join(sorted(AGENT_FACTORIES))}",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of new games to run for each selected agent.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Incremental runs use continuing game numbers to avoid repeating boards.",
    )

    parser.add_argument(
        "--board-size",
        type=int,
        default=10,
        help="Battleship board size.",
    )

    parser.add_argument(
        "--ships",
        nargs="+",
        type=int,
        default=[5, 4, 3, 3, 2],
        help="Ship sizes.",
    )

    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where result CSV/JSON files are saved.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete old result files before running.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_evaluation(
        agents=args.agents,
        games=args.games,
        seed=args.seed,
        board_size=args.board_size,
        ship_sizes=tuple(args.ships),
        results_dir=Path(args.results),
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
