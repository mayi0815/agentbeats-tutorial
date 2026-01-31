import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


ROOT_DIR = Path(__file__).resolve().parents[4]
WEB_AGENT_ROOT = ROOT_DIR / "third_party" / "webshop"
if str(WEB_AGENT_ROOT) not in sys.path:
    sys.path.append(str(WEB_AGENT_ROOT))

from web_agent_site.envs import WebAgentTextEnv  # noqa: E402


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webshop_evaluator")


class WebshopConfig(BaseModel):
    max_steps: int = Field(20, ge=1)
    observation_mode: str = Field("text")
    num_products: int = Field(1000, ge=1)
    session_prefix: str | None = None
    limit_goals: int | None = None
    human_goals: bool = False
    num_episodes: int = Field(1, ge=1)
    seed: int | None = None
    task_ids: list[int] | None = None
    split: str | None = None
    result_path: str | None = None


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


@dataclass
class ActionTrace:
    episode: int
    step: int
    action: str
    reward: float
    observation: str
    done: bool


@dataclass
class EpisodeResult:
    episode: int
    goal_idx: int
    goal: dict[str, Any]
    trace: list[ActionTrace]
    total_reward: float
    steps: int
    success: bool


class Agent:
    required_roles: list[str] = ["webshop_agent"]

    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
        except ValidationError as exc:
            await updater.reject(new_agent_text_message(f"Invalid request: {exc}"))
            return

        config = self._parse_config(request.config)
        participant_url = self._resolve_participant(request.participants)
        context_id = message.context_id or message.message_id or "webshop-session"

        env = self._create_env(config)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("WebShop assessment started."),
        )

        try:
            goal_indices = self._prepare_goal_indices(env, config)
            episode_results: list[EpisodeResult] = []

            for episode_idx, goal_idx in enumerate(goal_indices):
                episode_result = await self._run_episode(
                    env,
                    participant_url,
                    updater,
                    config,
                    context_id,
                    episode_idx,
                    goal_idx,
                )
                episode_results.append(episode_result)
        finally:
            self.messenger.reset()
            env.close()

        total_reward = sum(ep.total_reward for ep in episode_results)
        total_steps = sum(ep.steps for ep in episode_results)
        success = any(ep.success for ep in episode_results)
        result_data = {
            "success": success,
            "total_reward": total_reward,
            "steps": total_steps,
            "episodes": [
                {
                    "episode": ep.episode,
                    "goal_idx": ep.goal_idx,
                    "goal": ep.goal,
                    "success": ep.success,
                    "total_reward": ep.total_reward,
                    "steps": ep.steps,
                    "trace": [asdict(entry) for entry in ep.trace],
                }
                for ep in episode_results
            ],
        }
        result_path = config.result_path or os.getenv("WEBSHOP_RESULT_PATH")
        if result_path:
            try:
                path = Path(result_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
            except OSError as exc:
                logger.warning("Failed to write result to %s: %s", result_path, exc)
        summary_text = (
            f"WebShop run complete (episodes={len(episode_results)}, "
            f"success={success}, reward={total_reward:.2f}, steps={total_steps})"
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=result_data)),
            ],
            name="Result",
        )
        await updater.complete(new_agent_text_message(summary_text))

    def _parse_config(self, raw_config: dict[str, Any]) -> WebshopConfig:
        config = WebshopConfig.model_validate(raw_config or {})
        return config

    def _resolve_participant(self, participants: dict[str, HttpUrl]) -> str:
        if "webshop_agent" in participants:
            return str(participants["webshop_agent"])
        if participants:
            return str(next(iter(participants.values())))
        raise ValueError("No participant available for WebShop assessment")

    def _create_env(self, config: WebshopConfig) -> WebAgentTextEnv:
        kwargs: dict[str, Any] = {
            "observation_mode": config.observation_mode,
            "num_products": config.num_products,
            "session_prefix": config.session_prefix,
            "limit_goals": config.limit_goals if config.limit_goals is not None else -1,
            "human_goals": config.human_goals,
        }
        return WebAgentTextEnv(**kwargs)

    async def _run_episode(
        self,
        env: WebAgentTextEnv,
        participant_url: str,
        updater: TaskUpdater,
        config: WebshopConfig,
        context_id: str,
        episode_idx: int,
        goal_idx: int,
    ) -> EpisodeResult:
        session_id = f"{context_id}-ep{episode_idx}"
        env.server.user_sessions.pop(session_id, None)
        goal = env.server.goals[goal_idx]
        env.server.user_sessions[session_id] = {"goal": goal, "done": False}
        self.messenger.reset()

        obs, _ = env.reset(session=session_id)
        trace: list[ActionTrace] = []
        total_reward = 0.0
        steps = 0
        done = False

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Episode {episode_idx + 1} start (goal idx {goal_idx})."
            ),
        )

        try:
            while steps < config.max_steps and not done:
                available_actions = env.get_available_actions()
                prompt = {
                    "observation": obs,
                    "available_actions": available_actions,
                    "step": steps,
                    "max_steps": config.max_steps,
                    "instruction": env.instruction_text,
                    "episode": episode_idx,
                    "goal_idx": goal_idx,
                }
                response = await self.messenger.talk_to_agent(
                    json.dumps(prompt),
                    participant_url,
                    new_conversation=True,
                )

                action = self._parse_action(response, env.instruction_text)
                action_str = self._format_action(action, env.instruction_text)

                obs, reward, done, info = env.step(action_str)
                total_reward += reward
                steps += 1
                trace.append(
                    ActionTrace(
                        episode=episode_idx,
                        step=steps,
                        action=action_str,
                        reward=reward,
                        observation=obs,
                        done=done,
                    )
                )

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Episode {episode_idx + 1} Step {steps}: "
                        f"{action_str} (reward={reward}, done={done})"
                    ),
                )
        finally:
            env.server.user_sessions.pop(session_id, None)

        success = any(entry.reward >= 1 for entry in trace)
        return EpisodeResult(
            episode=episode_idx,
            goal_idx=goal_idx,
            goal=goal,
            trace=trace,
            total_reward=total_reward,
            steps=steps,
            success=success,
        )

    def _prepare_goal_indices(
        self, env: WebAgentTextEnv, config: WebshopConfig
    ) -> list[int]:
        total = len(env.server.goals)
        if total == 0:
            raise RuntimeError("WebShop environment loaded zero goals.")

        if config.split:
            candidate_indices = self._split_indices(total, config.split)
            if not candidate_indices:
                logger.warning(
                    "Split %s produced no goals; falling back to entire goal set.",
                    config.split,
                )
                candidate_indices = list(range(total))
        else:
            candidate_indices = list(range(total))

        if config.task_ids:
            filtered = []
            candidate_set = set(candidate_indices)
            for idx in config.task_ids:
                if 0 <= idx < total and idx in candidate_set:
                    filtered.append(idx)
            if filtered:
                candidate_indices = filtered
            else:
                logger.warning(
                    "Provided task_ids %s do not intersect with available goals; "
                    "using default sampling.",
                    config.task_ids,
                )
        if not candidate_indices:
            candidate_indices = list(range(total))

        if config.task_ids:
            limited = candidate_indices[: min(config.num_episodes, len(candidate_indices))]
            return limited

        rng = random.Random(config.seed if config.seed is not None else 0)
        rng.shuffle(candidate_indices)
        count = min(config.num_episodes, len(candidate_indices))
        if count == 0:
            count = max(1, len(candidate_indices))
        return candidate_indices[:count]

    def _split_indices(self, total: int, split: str) -> list[int]:
        split = split.lower() if split else ""
        if split == "test":
            return list(range(min(500, total)))
        if split == "eval":
            start = min(500, total)
            end = min(1500, total)
            return list(range(start, end))
        if split == "train":
            start = min(1500, total)
            return list(range(start, total))
        return list(range(total))

    def _parse_action(self, response: str, instruction: str) -> dict[str, Any]:
        try:
            payload = json.loads(response)
            if isinstance(payload, dict) and "type" in payload:
                return payload
        except json.JSONDecodeError:
            pass
        return {"type": "search", "query": self._default_query(instruction)}

    def _format_action(self, action: dict[str, Any], instruction: str) -> str:
        kind = action.get("type", "search")
        if kind == "search":
            query = (action.get("query") or "").strip()
            if not query:
                query = self._default_query(instruction)
            return f"search[{query}]"
        if kind in {"click", "choose", "buy"}:
            text = (action.get("text") or "").strip()
            if not text:
                text = "Buy Now"
            return f"click[{text}]"
        return f"search[{self._default_query(instruction)}]"

    def _default_query(self, instruction: str) -> str:
        tokens = [token for token in instruction.split() if token.isalnum()]
        if tokens:
            return " ".join(tokens[:2])
        return "best sellers"
