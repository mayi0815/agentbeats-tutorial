import json
import os
import random
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable

from dotenv import load_dotenv
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()

PRODUCT_TYPE_SYNONYMS: dict[str, list[str]] = {
    "tshirt": ["t shirt", "t-shirt", "tshirt", "tee"],
    "sweater": ["sweater", "pullover", "jumper"],
    "hoodie": ["hoodie", "hooded"],
    "tank": ["tank", "tank top"],
    "polo": ["polo"],
    "dress": ["dress", "tunic"],
    "shirt": ["shirt", "shirts"],
    "pants": ["pants", "trousers"],
    "shorts": ["shorts"],
    "shoes": ["shoes", "sneakers", "boots", "sandals"],
    "jacket": ["jacket", "coat"],
    "skirt": ["skirt"],
    "leggings": ["leggings"],
}


class Agent:
    def __init__(self):
        self._rng = random.Random(42)
        self._base_weights = self._load_scoring_weights()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = self._parse_payload(get_message_text(message))
        action = self._select_action(payload)
        action_str = f"{action['type']}:{action.get('query') or action.get('text')}"

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Selected action: {action_str}"),
        )

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=action))],
            name="Action",
        )

    def _parse_payload(self, text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"observation": text}

    def _select_action(self, payload: dict[str, Any]) -> Dict[str, str]:
        available = payload.get("available_actions", {})
        clickables = self._clean_clickables(available.get("clickables", []))
        instruction = payload.get("instruction", "")
        observation = payload.get("observation", "")
        weights = self._merge_scoring_weights(payload.get("scoring"))

        step = int(payload.get("step", 0) or 0)

        if available.get("has_search_bar"):
            return {"type": "search", "query": self._build_search_query(instruction)}

        results = self._extract_search_results(observation)
        if results:
            best_asin = self._pick_best_result(results, instruction, clickables, weights)
            if best_asin:
                return {"type": "click", "text": best_asin}

        desired_values = self._desired_values(instruction)
        option_groups = self._extract_option_groups(observation)
        option_pick = self._pick_page_option(clickables, instruction, option_groups, step)
        if option_pick:
            return {"type": "click", "text": option_pick}

        buy_button = self._pick_buy(clickables)
        has_options = bool(option_groups.get("color") or option_groups.get("size"))
        if buy_button and (step >= 3 or not desired_values or not has_options):
            return {"type": "click", "text": buy_button}

        matched_option = self._pick_matching_option(clickables, desired_values)
        if matched_option:
            return {"type": "click", "text": matched_option}

        asin_click = self._pick_asin(clickables)
        if asin_click:
            return {"type": "click", "text": asin_click}

        fallback = self._pick_fallback(clickables)
        if fallback:
            return {"type": "click", "text": fallback}

        return {"type": "search", "query": self._build_search_query(instruction)}

    def _default_query(self, instruction: str) -> str:
        instruction = instruction.strip()
        match = re.search(r"find me\s+(.+?)(?:\s+with|\s+that|,|$)", instruction, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
        tokens = [chunk for chunk in re.split(r"\s+", instruction) if chunk.isalnum()]
        if tokens:
            return " ".join(tokens[:4])
        return "best sellers"

    def _build_search_query(self, instruction: str) -> str:
        base = self._query_from_instruction(instruction)
        base_clean = self._clean_query_text(base)
        base_lower = base_clean.lower()
        gender = self._instruction_gender(instruction)
        attributes = self._extract_attribute_phrases(instruction)
        type_tokens = []
        if base_clean:
            type_tokens = [base_clean]
        else:
            for type_key in self._instruction_product_types(instruction):
                phrases = PRODUCT_TYPE_SYNONYMS.get(type_key, [])
                if phrases:
                    type_tokens.append(phrases[0])
        tokens: list[str] = []
        if gender:
            tokens.append(gender)
        tokens.extend(type_tokens)
        desired_color = self._desired_option_value(instruction, "color")
        desired_size = self._desired_option_value(instruction, "size")
        desired_color = desired_color.lower() if desired_color else None
        desired_size = desired_size.lower() if desired_size else None
        for phrase in self._filter_query_attributes(
            attributes,
            desired_color=desired_color,
            desired_size=desired_size,
        ):
            if phrase in base_lower:
                continue
            if any(phrase in token for token in type_tokens):
                continue
            tokens.append(phrase)
        query = self._dedupe_and_join_tokens(tokens, max_tokens=10)
        if query:
            return query
        return self._default_query(instruction)

    def _clean_clickables(self, clickables: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        for candidate in clickables:
            if not candidate:
                continue
            value = candidate.strip()
            if value:
                cleaned.append(value)
        return cleaned

    def _desired_values(self, instruction: str) -> list[str]:
        values: list[str] = []
        for key in ("color", "size", "brand", "material"):
            pattern = rf"{key}\s*:\s*([^,]+)"
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                values.append(match.group(1).strip())
        for phrase in ("long sleeve", "short sleeve", "cotton", "polyester", "spandex", "leather"):
            if phrase in instruction.lower():
                values.append(phrase)
        return [value for value in values if value]

    def _pick_matching_option(self, clickables: list[str], desired_values: list[str]) -> str | None:
        if not desired_values:
            return None
        for desired in desired_values:
            desired_lower = desired.lower()
            for candidate in clickables:
                if desired_lower in candidate.lower():
                    return candidate
        return None

    def _pick_buy(self, clickables: list[str]) -> str | None:
        for candidate in clickables:
            if "buy" in candidate.lower():
                return candidate
        return None

    def _pick_asin(self, clickables: list[str]) -> str | None:
        for candidate in clickables:
            if re.fullmatch(r"[A-Z0-9]{10}", candidate):
                return candidate
        return None

    def _pick_fallback(self, clickables: list[str]) -> str | None:
        ignore = {
            "search",
            "next >",
            "previous",
            "< prev",
            "back to search",
            "description",
            "features",
            "reviews",
        }
        for candidate in clickables:
            if candidate.lower() in ignore:
                continue
            return candidate
        if clickables:
            return clickables[0]
        return None

    def _extract_search_results(self, observation: str) -> list[dict[str, str]]:
        if not observation:
            return []
        parts = [chunk.strip() for chunk in observation.split("[SEP]") if chunk.strip()]
        results: list[dict[str, str]] = []
        for idx, chunk in enumerate(parts):
            if not self._is_asin(chunk):
                continue
            title = parts[idx + 1] if idx + 1 < len(parts) else ""
            price = parts[idx + 2] if idx + 2 < len(parts) else ""
            if not title:
                continue
            results.append({"asin": chunk, "title": title, "price": price})
        return results

    def _pick_best_result(
        self,
        results: list[dict[str, str]],
        instruction: str,
        clickables: list[str],
        weights: dict[str, float],
    ) -> str | None:
        desired_values = self._desired_values(instruction)
        desired_phrases = self._desired_phrases_for_scoring(instruction)
        price_upper = self._parse_price_upper(instruction)
        keywords = self._instruction_keywords(instruction)
        gender = self._instruction_gender(instruction)
        query = self._query_from_instruction(instruction)
        query_norm = self._normalize(query) if query else ""
        target_types = self._instruction_product_types(instruction)

        best_score = None
        best_asin = None

        for result in results:
            asin = result["asin"]
            clickable = self._match_clickable(clickables, asin)
            if not clickable:
                continue
            title_raw = result["title"]
            title = title_raw.lower()
            title_norm = self._normalize(title)
            price = self._parse_price(result.get("price", ""))
            score = 0.0
            if query_norm:
                score += SequenceMatcher(None, query_norm, title_norm).ratio() * weights["query_similarity"]
            for token in keywords:
                if token in title:
                    score += weights["keyword"]
            for desired in desired_values:
                if desired.lower() in title:
                    score += weights["desired_value"]
            for phrase in desired_phrases:
                phrase_norm = self._normalize(phrase)
                if phrase_norm and phrase_norm in title_norm:
                    score += weights["desired_phrase"]
            if target_types:
                if any(self._type_presence(title_norm, type_key) for type_key in target_types):
                    score += weights["type_match"]
                else:
                    score += weights["type_missing"]
                for type_key in PRODUCT_TYPE_SYNONYMS:
                    if type_key in target_types:
                        continue
                    if self._type_presence(title_norm, type_key):
                        score += weights["non_target_type"]
            for type_key in target_types:
                score += self._mismatch_type_penalty(type_key, title_norm, weights["mismatch_type"])
            if gender:
                if gender in title:
                    score += weights["gender_match"]
                elif self._gender_mismatch(title, gender):
                    score += weights["gender_mismatch"]
            if price_upper is not None and price is not None:
                if price <= price_upper:
                    score += weights["price_match"]
                else:
                    score += weights["price_mismatch"]
            if best_score is None or score > best_score:
                best_score = score
                best_asin = clickable

        return best_asin

    def _match_clickable(self, clickables: list[str], asin: str) -> str | None:
        asin_lower = asin.lower()
        for candidate in clickables:
            if candidate.lower() == asin_lower:
                return candidate
        return None

    def _is_asin(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9]{10}", value.strip(), re.IGNORECASE))

    def _parse_price(self, text: str) -> float | None:
        matches = re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", text)
        if not matches:
            return None
        prices = [float(price) for price in matches]
        return min(prices) if prices else None

    def _parse_price_upper(self, instruction: str) -> float | None:
        match = re.search(r"price lower than\s+([0-9]+(?:\.[0-9]+)?)", instruction, re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1))

    def _instruction_keywords(self, instruction: str) -> list[str]:
        instruction = instruction.lower()
        match = re.search(r"find me\s+(.+?)(?:\s+with|,|$)", instruction)
        if match:
            query = match.group(1)
        else:
            query = instruction
        tokens = [token for token in re.split(r"\W+", query) if token]
        stopwords = {"find", "me", "with", "and", "for", "the", "a", "an", "price", "lower"}
        return [token for token in tokens if token not in stopwords]

    def _desired_phrases_for_scoring(self, instruction: str) -> list[str]:
        instruction_lower = instruction.lower()
        phrases = set(self._desired_values(instruction))
        color = self._desired_option_value(instruction, "color")
        size = self._desired_option_value(instruction, "size")
        if color:
            phrases.add(color)
        if size:
            phrases.update(self._expand_size_tokens(size))
        for phrase in self._extract_attribute_phrases(instruction):
            phrases.add(phrase)

        extra_phrases = [
            "machine wash",
            "hand wash",
            "long sleeve",
            "short sleeve",
            "crew neck",
            "v neck",
            "round neck",
            "hoodie",
            "sweater",
            "t-shirt",
            "t shirt",
            "tee",
            "tank",
            "polo",
            "tunic",
            "dress",
            "stretch fabric",
            "cotton",
            "polyester",
            "spandex",
            "leather",
            "fleece",
        ]
        for phrase in extra_phrases:
            if phrase in instruction_lower:
                phrases.add(phrase)
        return [phrase for phrase in phrases if phrase]

    def _extract_attribute_phrases(self, instruction: str) -> list[str]:
        instruction_lower = instruction.lower()
        if "with" not in instruction_lower:
            return []
        segment = instruction_lower.split("with", 1)[1]
        segment = segment.split("price lower than", 1)[0]
        segment = segment.replace("with color", "")
        segment = segment.replace("with size", "")
        raw_parts = re.split(r",| and ", segment)
        phrases = []
        for part in raw_parts:
            part = part.strip(" .:-")
            if not part:
                continue
            if "fit type" in part:
                continue
            part = re.sub(r"color\s*:\s*", "", part)
            part = re.sub(r"size\s*:\s*", "", part)
            if not part:
                continue
            phrases.append(part)
        return phrases

    def _query_from_instruction(self, instruction: str) -> str:
        instruction = instruction.strip()
        match = re.search(r"find me\s+(.+?)(?:\s+with|,|$)", instruction, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return instruction

    def _instruction_product_types(self, instruction: str) -> list[str]:
        base = self._query_from_instruction(instruction).lower()
        types: set[str] = set()
        for type_key, phrases in PRODUCT_TYPE_SYNONYMS.items():
            for phrase in phrases:
                if phrase in base:
                    types.add(type_key)
                    break
        return list(types)

    def _type_presence(self, title_norm: str, type_key: str) -> bool:
        for phrase in PRODUCT_TYPE_SYNONYMS.get(type_key, []):
            if self._normalize(phrase) in title_norm:
                return True
        return False

    def _mismatch_type_penalty(self, target_type: str, title_norm: str, weight: float) -> float:
        mismatch_map = {
            "sweater": ["tshirt", "tank", "polo"],
            "tshirt": ["sweater", "hoodie", "dress"],
            "dress": ["tshirt", "tank", "polo"],
            "hoodie": ["tshirt", "tank"],
            "tank": ["sweater", "hoodie", "dress"],
            "polo": ["tank", "hoodie"],
            "shoes": ["shirt", "tshirt", "sweater"],
        }
        penalty = 0.0
        for other in mismatch_map.get(target_type, []):
            if self._type_presence(title_norm, other):
                penalty += weight
        return penalty

    def _load_scoring_weights(self) -> dict[str, float]:
        weights = self._default_scoring_weights()
        raw = os.getenv("WEBSHOP_SCORING_WEIGHTS")
        if raw:
            try:
                overrides = json.loads(raw)
                if isinstance(overrides, dict):
                    self._apply_weight_overrides(weights, overrides)
            except json.JSONDecodeError:
                pass
        return weights

    def _merge_scoring_weights(self, override: object) -> dict[str, float]:
        weights = dict(self._base_weights)
        if isinstance(override, dict):
            self._apply_weight_overrides(weights, override)
        return weights

    def _apply_weight_overrides(self, weights: dict[str, float], overrides: dict[str, Any]) -> None:
        for key, value in overrides.items():
            if key in weights and isinstance(value, (int, float)):
                weights[key] = float(value)

    def _default_scoring_weights(self) -> dict[str, float]:
        return {
            "query_similarity": 2.0,
            "keyword": 1.2,
            "desired_value": 2.0,
            "desired_phrase": 2.5,
            "type_match": 5.0,
            "type_missing": -2.5,
            "non_target_type": -2.0,
            "mismatch_type": -4.0,
            "gender_match": 2.0,
            "gender_mismatch": -3.5,
            "price_match": 1.5,
            "price_mismatch": -1.0,
        }

    def _filter_query_attributes(
        self,
        phrases: list[str],
        *,
        desired_color: str | None,
        desired_size: str | None,
    ) -> list[str]:
        filtered: list[str] = []
        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            if "fit type" in phrase:
                continue
            if phrase.startswith("color") or phrase.startswith("size"):
                continue
            if desired_color and phrase == desired_color.lower():
                continue
            if desired_size and phrase == desired_size.lower():
                continue
            if len(phrase) > 40:
                continue
            filtered.append(phrase)
            if len(filtered) >= 2:
                break
        return filtered

    def _dedupe_and_join_tokens(self, tokens: list[str], max_tokens: int = 10) -> str:
        seen = set()
        cleaned: list[str] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            norm = self._normalize(token)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            cleaned.append(token)
            if len(cleaned) >= max_tokens:
                break
        return " ".join(cleaned).strip()

    def _clean_query_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s\-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _instruction_gender(self, instruction: str) -> str | None:
        instruction = instruction.lower()
        if re.search(r"\bwomen's\b|\bwomen\b", instruction):
            return "women"
        if re.search(r"\bmen's\b|\bmen\b", instruction):
            return "men"
        if re.search(r"\bgirls\b|\bgirl's\b", instruction):
            return "girls"
        if re.search(r"\bboys\b|\bboy's\b", instruction):
            return "boys"
        return None

    def _gender_mismatch(self, title: str, gender: str) -> bool:
        opposite = {
            "men": ["women", "girls"],
            "women": ["men", "boys"],
            "girls": ["men", "boys"],
            "boys": ["women", "girls"],
        }
        for other in opposite.get(gender, []):
            if other in title:
                return True
        return False

    def _expand_size_tokens(self, size: str) -> list[str]:
        size_lower = size.lower().strip()
        variants = {size_lower}
        size_map = {
            "xx-large": "xxl",
            "x-large": "xl",
            "large": "l",
            "medium": "m",
            "small": "s",
            "x-small": "xs",
            "xx-small": "xxs",
            "3x-large": "3xl",
            "4x-large": "4xl",
        }
        compact = size_lower.replace(" ", "").replace("-", "")
        variants.add(compact)
        if size_lower in size_map:
            variants.add(size_map[size_lower])
        return list(variants)

    def _pick_page_option(
        self,
        clickables: list[str],
        instruction: str,
        option_groups: dict[str, list[str]],
        step: int,
    ) -> str | None:
        color = self._desired_option_value(instruction, "color")
        size = self._desired_option_value(instruction, "size")
        color_option = None
        size_option = None
        if color:
            color_option = self._match_option_value(color, option_groups.get("color", []), clickables)
        if size:
            size_option = self._match_option_value(size, option_groups.get("size", []), clickables)

        if color_option and not size_option:
            return color_option if step <= 2 else None
        if size_option and not color_option:
            return size_option if step <= 2 else None
        if color_option and size_option:
            if step <= 3:
                if step % 2 == 0:
                    return size_option
                return color_option
        return None

    def _desired_option_value(self, instruction: str, key: str) -> str | None:
        pattern = rf"{key}\s*:\s*([^,]+)"
        match = re.search(pattern, instruction, re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def _extract_option_groups(self, observation: str) -> dict[str, list[str]]:
        if not observation:
            return {}
        parts = [chunk.strip() for chunk in observation.split("[SEP]") if chunk.strip()]
        groups: dict[str, list[str]] = {}
        current_key = None
        stop_tokens = {"price", "rating", "description", "features", "reviews", "buy now", "back to search", "< prev"}
        for part in parts:
            lower = part.lower()
            if lower in {"color", "size"}:
                current_key = lower
                groups.setdefault(current_key, [])
                continue
            if lower in stop_tokens or lower.startswith("price"):
                current_key = None
                continue
            if current_key:
                groups[current_key].append(part)
        return groups

    def _match_option_value(
        self,
        desired: str,
        options: list[str],
        clickables: list[str],
    ) -> str | None:
        desired_norm = self._normalize(desired)
        for option in options:
            if self._normalize(option) == desired_norm:
                return self._match_clickable(clickables, option) or option
        for option in options:
            if desired_norm in self._normalize(option):
                return self._match_clickable(clickables, option) or option
        for candidate in clickables:
            if desired_norm in self._normalize(candidate):
                return candidate
        return None

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())
