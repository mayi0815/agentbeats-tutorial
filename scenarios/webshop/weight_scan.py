#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = ROOT / "scenarios" / "webshop" / "webshop-agent" / "src"
sys.path.insert(0, str(AGENT_SRC))

from agent import Agent  # noqa: E402


def _sample_weights(base: dict[str, float], rng: random.Random, scale: float) -> dict[str, float]:
    sampled: dict[str, float] = {}
    for key, value in base.items():
        if value == 0:
            sampled[key] = 0.0
            continue
        jitter = rng.uniform(1.0 - scale, 1.0 + scale)
        sampled[key] = round(value * jitter, 3)
    return sampled


def _run_trial(
    scenario: Path,
    weights: dict[str, float],
    result_path: Path,
    show_logs: bool,
) -> dict[str, object]:
    if result_path.exists():
        result_path.unlink()

    env = os.environ.copy()
    env["WEBSHOP_SCORING_WEIGHTS"] = json.dumps(weights)
    env["WEBSHOP_RESULT_PATH"] = str(result_path)

    cmd = ["uv", "run", "agentbeats-run", str(scenario)]
    if show_logs:
        cmd.append("--show-logs")

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=not show_logs,
    )
    if proc.returncode != 0:
        if not show_logs:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"agentbeats-run failed (exit={proc.returncode})")

    if not result_path.exists():
        raise RuntimeError(f"Missing result file at {result_path}")

    return json.loads(result_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Random search over WebShop scoring weights.")
    parser.add_argument("--scenario", type=Path, default=ROOT / "scenarios" / "webshop" / "scenario.toml")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("/tmp/webshop_weight_scan.jsonl"))
    parser.add_argument("--show-logs", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = Agent()._default_scoring_weights()

    results: list[dict[str, object]] = []
    for idx in range(args.trials):
        weights = _sample_weights(base, rng, args.scale)
        result_path = Path(f"/tmp/webshop_scan_result_{idx}.json")
        result = _run_trial(args.scenario, weights, result_path, args.show_logs)
        episodes = result.get("episodes", [])
        episode_count = len(episodes) if isinstance(episodes, list) else 0
        total_reward = float(result.get("total_reward", 0.0))
        avg_reward = total_reward / episode_count if episode_count else 0.0
        success_count = sum(1 for ep in episodes if ep.get("success")) if episode_count else 0

        record = {
            "trial": idx,
            "weights": weights,
            "avg_reward": round(avg_reward, 4),
            "total_reward": round(total_reward, 4),
            "episodes": episode_count,
            "successes": success_count,
        }
        results.append(record)

        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

    results.sort(key=lambda item: item["avg_reward"], reverse=True)
    top = results[: min(3, len(results))]
    print("Top weight sets:")
    for entry in top:
        print(
            f"trial={entry['trial']} avg_reward={entry['avg_reward']} "
            f"successes={entry['successes']}/{entry['episodes']}"
        )
        print(json.dumps(entry["weights"], indent=2))
    print(f"Full scan saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
