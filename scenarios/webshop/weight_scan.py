#!/usr/bin/env python3
"""
Weight optimization for WebShop agent using gradient descent and random search.

Ce script optimise les poids de scoring de l'agent en utilisant:
1. Random search initial pour explorer l'espace
2. Gradient descent pour affiner les meilleurs poids trouvés
3. Historique des résultats pour apprentissage continu
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = ROOT / "scenarios" / "webshop" / "webshop-agent" / "src"
sys.path.insert(0, str(AGENT_SRC))

from agent import Agent  # noqa: E402


# ============================================================================
# HISTORIQUE DES RÉSULTATS POUR APPRENTISSAGE
# ============================================================================

@dataclass
class TrialResult:
    """Résultat d'un essai d'optimisation."""
    weights: dict[str, float]
    avg_reward: float
    total_reward: float
    episodes: int
    successes: int
    
    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "episodes": self.episodes,
            "successes": self.successes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrialResult":
        return cls(
            weights=data["weights"],
            avg_reward=data["avg_reward"],
            total_reward=data["total_reward"],
            episodes=data["episodes"],
            successes=data["successes"],
        )


@dataclass
class OptimizationHistory:
    """Historique des optimisations pour apprentissage continu."""
    results: list[TrialResult] = field(default_factory=list)
    best_weights: Optional[dict[str, float]] = None
    best_reward: float = 0.0
    
    def add_result(self, result: TrialResult) -> None:
        self.results.append(result)
        if result.avg_reward > self.best_reward:
            self.best_reward = result.avg_reward
            self.best_weights = result.weights.copy()
    
    def save(self, path: Path) -> None:
        data = {
            "results": [r.to_dict() for r in self.results],
            "best_weights": self.best_weights,
            "best_reward": self.best_reward,
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "OptimizationHistory":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        history = cls(
            best_weights=data.get("best_weights"),
            best_reward=data.get("best_reward", 0.0),
        )
        for r in data.get("results", []):
            history.results.append(TrialResult.from_dict(r))
        return history
    
    def get_top_n(self, n: int = 5) -> list[TrialResult]:
        """Retourne les n meilleurs résultats."""
        sorted_results = sorted(self.results, key=lambda r: r.avg_reward, reverse=True)
        return sorted_results[:n]


# ============================================================================
# GRADIENT DESCENT OPTIMIZER
# ============================================================================

class GradientDescentOptimizer:
    """Optimise les poids par gradient descent approximatif."""
    
    def __init__(
        self,
        base_weights: dict[str, float],
        learning_rate: float = 0.1,
        epsilon: float = 0.05,
    ):
        self.base_weights = base_weights.copy()
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Pour estimation du gradient
    
    def estimate_gradient(
        self,
        weights: dict[str, float],
        run_fn,
        key: str,
    ) -> float:
        """Estime le gradient pour un poids spécifique."""
        # f(x + epsilon)
        weights_plus = weights.copy()
        weights_plus[key] = weights[key] + self.epsilon
        reward_plus = run_fn(weights_plus)
        
        # f(x - epsilon)
        weights_minus = weights.copy()
        weights_minus[key] = weights[key] - self.epsilon
        reward_minus = run_fn(weights_minus)
        
        # Gradient approximatif
        gradient = (reward_plus - reward_minus) / (2 * self.epsilon)
        return gradient
    
    def step(
        self,
        weights: dict[str, float],
        gradients: dict[str, float],
    ) -> dict[str, float]:
        """Effectue un pas de gradient descent."""
        new_weights = weights.copy()
        for key, grad in gradients.items():
            new_weights[key] = weights[key] + self.learning_rate * grad
        return new_weights
    
    def optimize(
        self,
        initial_weights: dict[str, float],
        run_fn,
        keys_to_optimize: list[str],
        max_iterations: int = 10,
        verbose: bool = True,
    ) -> tuple[dict[str, float], float]:
        """Optimise les poids par gradient descent.
        
        Args:
            initial_weights: Poids de départ
            run_fn: Fonction qui prend des poids et retourne la reward moyenne
            keys_to_optimize: Liste des clés de poids à optimiser
            max_iterations: Nombre maximum d'itérations
            verbose: Afficher les logs
        
        Returns:
            (meilleurs poids, meilleure reward)
        """
        current_weights = initial_weights.copy()
        best_weights = current_weights.copy()
        best_reward = run_fn(current_weights)
        
        if verbose:
            print(f"[GD] Initial reward: {best_reward:.4f}")
        
        for iteration in range(max_iterations):
            # Estimer les gradients pour chaque poids
            gradients = {}
            for key in keys_to_optimize:
                grad = self.estimate_gradient(current_weights, run_fn, key)
                gradients[key] = grad
            
            # Effectuer un pas
            new_weights = self.step(current_weights, gradients)
            new_reward = run_fn(new_weights)
            
            if verbose:
                print(f"[GD] Iteration {iteration + 1}: reward={new_reward:.4f}")
            
            # Mise à jour si amélioration
            if new_reward > best_reward:
                best_reward = new_reward
                best_weights = new_weights.copy()
            
            current_weights = new_weights
            
            # Réduction du learning rate
            self.learning_rate *= 0.95
        
        return best_weights, best_reward


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

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


def _parse_result(result: dict) -> TrialResult:
    """Parse un résultat en TrialResult."""
    episodes = result.get("episodes", [])
    episode_count = len(episodes) if isinstance(episodes, list) else 0
    total_reward = float(result.get("total_reward", 0.0))
    avg_reward = total_reward / episode_count if episode_count else 0.0
    success_count = sum(1 for ep in episodes if ep.get("success")) if episode_count else 0
    
    return TrialResult(
        weights={},  # Sera rempli par l'appelant
        avg_reward=round(avg_reward, 4),
        total_reward=round(total_reward, 4),
        episodes=episode_count,
        successes=success_count,
    )


def _load_fixed_weights(args: argparse.Namespace) -> dict[str, float] | None:
    if not args.fixed_weights_json and not args.fixed_weights_file:
        return None
    if args.fixed_weights_json and args.fixed_weights_file:
        raise ValueError("Use only one of --fixed-weights-json or --fixed-weights-file")
    raw = args.fixed_weights_json
    if args.fixed_weights_file:
        raw = Path(args.fixed_weights_file).read_text(encoding="utf-8")
    try:
        weights = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for fixed weights: {exc}") from exc
    if not isinstance(weights, dict):
        raise ValueError("Fixed weights must be a JSON object")
    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        if isinstance(value, (int, float)):
            cleaned[key] = float(value)
    if not cleaned:
        raise ValueError("Fixed weights JSON did not contain numeric values")
    return cleaned


def _prepare_scenario(path: Path, episodes: int | None, max_steps: int | None) -> Path:
    if episodes is None and max_steps is None:
        return path
    text = path.read_text(encoding="utf-8")
    updated = text
    if episodes is not None:
        updated = re.sub(r"^num_episodes\s*=\s*\d+", f"num_episodes = {episodes}", updated, flags=re.MULTILINE)
        if updated == text:
            updated = updated + f"\nnum_episodes = {episodes}\n"
    if max_steps is not None:
        updated = re.sub(r"^max_steps\s*=\s*\d+", f"max_steps = {max_steps}", updated, flags=re.MULTILINE)
        if updated == text:
            updated = updated + f"\nmax_steps = {max_steps}\n"
    suffix = f"episodes_{episodes or 'default'}_steps_{max_steps or 'default'}"
    tmp_path = Path(f"/tmp/webshop_scenario_{suffix}.toml")
    tmp_path.write_text(updated, encoding="utf-8")
    return tmp_path


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize WebShop scoring weights using gradient descent and random search."
    )
    parser.add_argument("--scenario", type=Path, default=ROOT / "scenarios" / "webshop" / "scenario.toml")
    parser.add_argument("--trials", type=int, default=10, help="Number of random search trials")
    parser.add_argument("--scale", type=float, default=0.35, help="Jitter scale for random sampling")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--episodes", type=int, help="Override num_episodes in scenario")
    parser.add_argument("--max-steps", type=int, help="Override max_steps in scenario")
    parser.add_argument("--resample-top-k", type=int, default=0, help="Resample around top-k weights")
    parser.add_argument("--resample-trials", type=int, default=10, help="Trials per top-k base")
    parser.add_argument("--resample-scale", type=float, default=0.15, help="Jitter scale for top-k resampling")
    parser.add_argument("--out", type=Path, default=Path("/tmp/webshop_weight_scan.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("/tmp/webshop_optimization_history.json"))
    parser.add_argument("--show-logs", action="store_true")
    parser.add_argument("--gradient-descent", action="store_true", help="Enable gradient descent refinement")
    parser.add_argument("--gd-iterations", type=int, default=5, help="Gradient descent iterations")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Gradient descent learning rate")
    parser.add_argument("--use-history", action="store_true", help="Use historical best weights as starting point")
    parser.add_argument("--fixed-weights-json", type=str, help="Fixed weights as JSON string")
    parser.add_argument("--fixed-weights-file", type=str, help="Fixed weights JSON file")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    
    # Charger l'historique
    history = OptimizationHistory.load(args.history)

    # Option: overrides du scenario (episodes)
    scenario = _prepare_scenario(args.scenario, args.episodes, args.max_steps)

    # Option: poids fixes
    fixed_weights = _load_fixed_weights(args)
    
    # Déterminer les poids de base
    if fixed_weights is not None:
        base = fixed_weights
        print("Using fixed weights")
    elif args.use_history and history.best_weights:
        base = history.best_weights
        print(f"Using historical best weights (reward={history.best_reward:.4f})")
    else:
        base = Agent()._default_scoring_weights()
        print("Using default weights")

    results: list[dict[str, object]] = []
    
    # ========================================================================
    # PHASE 1: Random Search
    # ========================================================================
    print(f"\n=== PHASE 1: Random Search ({args.trials} trials) ===")
    
    for idx in range(args.trials):
        weights = base if fixed_weights is not None else _sample_weights(base, rng, args.scale)
        result_path = Path(f"/tmp/webshop_scan_result_{idx}.json")
        
        try:
            result = _run_trial(scenario, weights, result_path, args.show_logs)
            trial_result = _parse_result(result)
            trial_result.weights = weights
            
            history.add_result(trial_result)
            
            record = {
                "trial": idx,
                "phase": "random_search",
                **trial_result.to_dict(),
            }
            results.append(record)

            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")
            
            print(f"  Trial {idx}: avg_reward={trial_result.avg_reward:.4f}, successes={trial_result.successes}/{trial_result.episodes}")
        
        except Exception as e:
            print(f"  Trial {idx}: FAILED - {e}")

    # ========================================================================
    # PHASE 1B: Top-k Resample (optionnel)
    # ========================================================================
    if args.resample_top_k > 0:
        top_bases = history.get_top_n(args.resample_top_k)
        if top_bases:
            print(f"\n=== PHASE 1B: Top-k Resample (k={args.resample_top_k}, trials={args.resample_trials}) ===")
        for base_idx, base_result in enumerate(top_bases):
            for trial_idx in range(args.resample_trials):
                weights = _sample_weights(base_result.weights, rng, args.resample_scale)
                result_path = Path(f"/tmp/webshop_resample_result_{base_idx}_{trial_idx}.json")

                try:
                    result = _run_trial(scenario, weights, result_path, args.show_logs)
                    trial_result = _parse_result(result)
                    trial_result.weights = weights

                    history.add_result(trial_result)

                    record = {
                        "trial": f"resample_{base_idx}_{trial_idx}",
                        "phase": "topk_resample",
                        **trial_result.to_dict(),
                    }
                    results.append(record)

                    with args.out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record))
                        handle.write("\n")

                    print(
                        f"  Resample {base_idx}:{trial_idx} avg_reward={trial_result.avg_reward:.4f}, "
                        f"successes={trial_result.successes}/{trial_result.episodes}"
                    )

                except Exception as e:
                    print(f"  Resample {base_idx}:{trial_idx} FAILED - {e}")

    # ========================================================================
    # PHASE 2: Gradient Descent (optionnel)
    # ========================================================================
    if args.gradient_descent and len(history.results) > 0:
        print(f"\n=== PHASE 2: Gradient Descent Refinement ===")
        
        top_results = history.get_top_n(3)
        
        for i, top_result in enumerate(top_results):
            print(f"\nRefining top result #{i+1} (reward={top_result.avg_reward:.4f})")
            
            optimizer = GradientDescentOptimizer(
                base_weights=top_result.weights,
                learning_rate=args.learning_rate,
                epsilon=0.05,
            )
            
            # Fonction pour évaluer les poids
            def evaluate_weights(weights: dict[str, float]) -> float:
                result_path = Path(f"/tmp/webshop_gd_result_{i}.json")
                try:
                    result = _run_trial(scenario, weights, result_path, args.show_logs)
                    trial_result = _parse_result(result)
                    return trial_result.avg_reward
                except Exception:
                    return 0.0
            
            # Optimiser les poids les plus importants
            keys_to_optimize = [
                "type_match", "type_missing", "gender_match", "gender_mismatch",
                "price_match", "price_mismatch", "brand_match", "full_match_bonus",
            ]
            
            best_weights, best_reward = optimizer.optimize(
                initial_weights=top_result.weights,
                run_fn=evaluate_weights,
                keys_to_optimize=keys_to_optimize,
                max_iterations=args.gd_iterations,
                verbose=True,
            )
            
            # Ajouter le résultat final
            gd_result = TrialResult(
                weights=best_weights,
                avg_reward=best_reward,
                total_reward=0.0,  # Non calculé ici
                episodes=0,
                successes=0,
            )
            history.add_result(gd_result)
            
            record = {
                "trial": f"gd_{i}",
                "phase": "gradient_descent",
                **gd_result.to_dict(),
            }
            results.append(record)
            
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")

    # ========================================================================
    # RÉSULTATS FINAUX
    # ========================================================================
    print("\n=== FINAL RESULTS ===")
    
    # Sauvegarder l'historique
    history.save(args.history)
    print(f"History saved to {args.history}")
    
    # Afficher les meilleurs résultats
    results.sort(key=lambda item: item["avg_reward"], reverse=True)
    top = results[: min(3, len(results))]
    print("\nTop weight sets:")
    for entry in top:
        print(
            f"trial={entry['trial']} phase={entry.get('phase', 'unknown')} "
            f"avg_reward={entry['avg_reward']} successes={entry.get('successes', '?')}/{entry.get('episodes', '?')}"
        )
        print(json.dumps(entry["weights"], indent=2))
    
    print(f"\nFull scan saved to {args.out}")
    
    # Afficher le meilleur global de l'historique
    if history.best_weights:
        print(f"\n=== BEST WEIGHTS (Historical) ===")
        print(f"Best reward: {history.best_reward:.4f}")
        print(json.dumps(history.best_weights, indent=2))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
