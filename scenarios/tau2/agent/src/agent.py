import json
import os
import re
from typing import Any

from dotenv import load_dotenv
import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part
from a2a.utils import get_message_text


load_dotenv()


SYSTEM_PROMPT = (
    "You are a helpful customer service agent.\n"
    "Follow the policy and tool instructions provided by the evaluator.\n"
    "Always return a single JSON object in this exact shape:\n"
    '{"name":"<tool_or_respond>","arguments":{...}}\n'
    "Do not include markdown fences or extra commentary."
)

RESPOND_ACTION_NAME = "respond"


class Agent:
    def __init__(self):
        self.model = os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1")
        self.max_turns = int(os.getenv("TAU2_AGENT_MAX_TURNS", "60"))
        self.allowed_actions: set[str] = {RESPOND_ACTION_NAME}
        self.messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        discovered_actions = self._extract_allowed_actions(input_text)
        if discovered_actions:
            self.allowed_actions = discovered_actions

        self.messages.append({"role": "user", "content": input_text})
        action = self._plan_action(self.allowed_actions)
        action_json = json.dumps(action, ensure_ascii=True)
        self.messages.append({"role": "assistant", "content": action_json})
        self._trim_history()

        await updater.add_artifact(parts=[Part(root=DataPart(data=action))], name="Action")

    def _plan_action(self, allowed_actions: set[str]) -> dict[str, Any]:
        try:
            completion = litellm.completion(
                model=self.model,
                messages=self.messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_output = self._extract_raw_output(completion)
            parsed = self._parse_json_object(raw_output)
            return self._sanitize_action(parsed, allowed_actions)
        except Exception:
            return self._fallback_action(
                allowed_actions,
                "I ran into an error processing your request.",
            )

    def _extract_raw_output(self, completion: Any) -> str:
        message = completion.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            first_call = tool_calls[0]
            function = getattr(first_call, "function", None)
            name = getattr(function, "name", None) if function else None
            args_text = getattr(function, "arguments", "{}") if function else "{}"
            try:
                arguments = json.loads(args_text) if isinstance(args_text, str) else {}
            except Exception:
                arguments = {}
            return json.dumps({"name": name, "arguments": arguments})
        return "{}"

    def _extract_allowed_actions(self, prompt: str) -> set[str]:
        names = {
            match.group(1).strip()
            for match in re.finditer(r'"name"\s*:\s*"([^"]+)"', prompt)
            if match.group(1).strip()
        }
        if names:
            names.add(RESPOND_ACTION_NAME)
        return names

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()

        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass

        candidate = self._find_first_json_object(stripped)
        if candidate is not None:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded

        raise ValueError("Model response did not contain a JSON object")

    def _find_first_json_object(self, text: str) -> str | None:
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            start = text.find("{", start + 1)
        return None

    def _sanitize_action(self, action: dict[str, Any], allowed_actions: set[str]) -> dict[str, Any]:
        name = action.get("name")
        arguments = action.get("arguments")

        if not isinstance(name, str) or not name.strip():
            return self._fallback_action(allowed_actions, "Could you clarify your request?")
        name = name.strip()

        if not isinstance(arguments, dict):
            arguments = {}

        if name not in allowed_actions:
            return self._fallback_action(
                allowed_actions,
                f"I cannot call '{name}'. Please provide the next step.",
            )

        if name == RESPOND_ACTION_NAME:
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                arguments = {"content": "Could you share more details so I can help?"}

        return {"name": name, "arguments": arguments}

    def _fallback_action(self, allowed_actions: set[str], content: str) -> dict[str, Any]:
        if RESPOND_ACTION_NAME in allowed_actions:
            return {"name": RESPOND_ACTION_NAME, "arguments": {"content": content}}
        fallback_name = next(iter(sorted(allowed_actions)), RESPOND_ACTION_NAME)
        return {"name": fallback_name, "arguments": {}}

    def _trim_history(self) -> None:
        max_messages = 1 + self.max_turns * 2
        if len(self.messages) <= max_messages:
            return
        self.messages = [self.messages[0]] + self.messages[-(max_messages - 1) :]
