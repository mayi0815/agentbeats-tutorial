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

# ============================================================================
# FUZZY MATCHING AVEC LEVENSHTEIN
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaînes."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Retourne un score de similarité entre 0 et 1 basé sur Levenshtein."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (distance / max_len)


def fuzzy_match_color(target: str, candidates: list[str], threshold: float = 0.75) -> str | None:
    """Trouve la meilleure correspondance fuzzy pour une couleur.
    
    Args:
        target: La couleur recherchée (potentiellement mal orthographiée)
        candidates: Liste des couleurs disponibles
        threshold: Seuil minimum de similarité (0-1)
    
    Returns:
        La meilleure correspondance ou None si aucune ne dépasse le seuil
    """
    target_norm = target.lower().strip()
    best_match = None
    best_score = threshold
    
    for candidate in candidates:
        candidate_norm = candidate.lower().strip()
        
        # Vérifier chaque partie si couleur composée
        parts = re.split(r"\s*[|/]\s*", candidate_norm)
        for part in parts:
            score = levenshtein_similarity(target_norm, part)
            if score > best_score:
                best_score = score
                best_match = candidate
    
    return best_match


# Couleurs communes pour le fuzzy matching
COMMON_COLORS = [
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "navy", "beige", "tan", "cream",
    "maroon", "olive", "teal", "coral", "salmon", "turquoise", "aqua",
    "gold", "silver", "bronze", "ivory", "lavender", "magenta", "cyan",
    "charcoal", "burgundy", "khaki", "indigo", "violet", "rose", "peach",
    "mint", "forest", "royal", "heather", "dark", "light", "bright"
]


def extract_readable_color(cryptic_color: str) -> str:
    """Extrait une couleur lisible depuis un pattern cryptique.
    
    Ex: "xnj-tshirt348-black" -> "black"
        "b-gray" -> "gray"
        "paris eiffel tower1goo8867" -> None (pas de couleur connue)
        "navy blue" -> "navy blue"
    
    Returns:
        La couleur lisible extraite ou le texte original si pas de pattern cryptique détecté
    """
    original = cryptic_color.lower().strip()
    
    # Si c'est déjà une couleur connue, retourner tel quel
    for color in COMMON_COLORS:
        if original == color:
            return original
    
    # Chercher des couleurs connues à la fin du pattern (après un tiret, underscore ou chiffres)
    # Pattern: quelque chose-couleur / quelque chose_couleur / quelque chose + chiffres + couleur
    for color in COMMON_COLORS:
        # Pattern: "-color" at end (ex: "xnj-tshirt348-black")
        if original.endswith(f"-{color}"):
            return color
        # Pattern: "_color" at end (ex: "xnj_tshirt348_black")
        if original.endswith(f"_{color}"):
            return color
        # Pattern: "X-color" where X is a letter (ex: "b-gray")
        if re.match(rf"^[a-z]-{color}$", original):
            return color
        # Pattern: "X_color" where X is a letter (ex: "b_gray")
        if re.match(rf"^[a-z]_{color}$", original):
            return color
        # Pattern: word boundary color (ex: "dark heather blue")
        if re.search(rf"\b{color}\b", original):
            # Si c'est une couleur composée comme "navy blue", retourner les deux
            match = re.search(rf"(\w+\s+)?{color}(\s+\w+)?", original)
            if match:
                result = match.group(0).strip()
                # Vérifier que les mots supplémentaires sont aussi des couleurs
                words = result.split()
                if len(words) <= 2 and all(w in COMMON_COLORS or w in ["dark", "light", "bright"] for w in words):
                    return result
                return color
    
    # Aucune couleur connue trouvée - retourner l'original
    return original

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

# NOUVEAU: Mapping des tailles avec leurs variations
SIZE_ALIASES: dict[str, list[str]] = {
    "xx-small": ["xx-small", "xxs", "xxsmall", "2xs"],
    "x-small": ["x-small", "xs", "xsmall", "extra small", "extra-small"],
    "small": ["small", "s", "sm"],
    "medium": ["medium", "m", "med"],
    "large": ["large", "l", "lg"],
    "x-large": ["x-large", "xl", "xlarge", "extra large", "extra-large"],
    "xx-large": ["xx-large", "xxl", "xxlarge", "2xl"],
    "3x-large": ["3x-large", "3xl", "xxxl", "3xlarge"],
    "4x-large": ["4x-large", "4xl", "xxxxl", "4xlarge"],
}


def fuzzy_match_size(target: str, candidates: list[str]) -> str | None:
    """Trouve la meilleure correspondance pour une taille.
    
    Args:
        target: La taille recherchée (ex: "large", "L", "xl")
        candidates: Liste des tailles disponibles
    
    Returns:
        La meilleure correspondance ou None
    """
    target_norm = target.lower().strip().replace(" ", "").replace("-", "")
    
    # Trouver la taille canonique correspondant à la cible
    target_canonical = None
    for canonical, aliases in SIZE_ALIASES.items():
        for alias in aliases:
            alias_norm = alias.lower().replace(" ", "").replace("-", "")
            if target_norm == alias_norm:
                target_canonical = canonical
                break
        if target_canonical:
            break
    
    if not target_canonical:
        # Pas de correspondance trouvée, essayer une correspondance partielle
        for canonical, aliases in SIZE_ALIASES.items():
            for alias in aliases:
                alias_norm = alias.lower().replace(" ", "").replace("-", "")
                if target_norm in alias_norm or alias_norm in target_norm:
                    target_canonical = canonical
                    break
            if target_canonical:
                break
    
    if not target_canonical:
        return None
    
    # Chercher dans les candidats
    target_aliases = [a.lower().replace(" ", "").replace("-", "") for a in SIZE_ALIASES.get(target_canonical, [])]
    
    for candidate in candidates:
        candidate_norm = candidate.lower().strip().replace(" ", "").replace("-", "")
        if candidate_norm in target_aliases:
            return candidate
        # Vérifier si le candidat contient une des variations
        for alias in target_aliases:
            if alias in candidate_norm or candidate_norm in alias:
                return candidate
    
    return None


# ============================================================================
# SESSION CACHE - Éviter les boucles en mémorisant les produits vus
# ============================================================================

class SessionCache:
    """Cache de session pour mémoriser les produits déjà visités et les options sélectionnées."""
    
    def __init__(self, max_size: int = 1000):
        self._seen_products: dict[str, int] = {}  # asin -> visit_count
        self._seen_actions: list[str] = []  # Historique des actions
        self._max_size = max_size
        self._failed_products: set[str] = set()  # Produits où l'achat a échoué
        self._option_retry_counts: dict[str, int] = {}  # asin -> retry_count
        self._secondary_search_query: str | None = None
        self._top_asin_index: int = 0
        self._consecutive_option_failures: int = 0
        # NOUVEAU: Tracking des options sélectionnées
        self._selected_color: str | None = None
        self._selected_size: str | None = None
        self._on_product_page: bool = False
        self._current_asin: str | None = None
        self._required_color: str | None = None  # Couleur requise par l'instruction
        self._required_size: str | None = None   # Taille requise par l'instruction
    
    def mark_seen(self, asin: str) -> None:
        """Marque un produit comme vu."""
        self._seen_products[asin] = self._seen_products.get(asin, 0) + 1
        if len(self._seen_products) > self._max_size:
            oldest = list(self._seen_products.keys())[:100]
            for key in oldest:
                del self._seen_products[key]
    
    def is_seen(self, asin: str) -> bool:
        return asin in self._seen_products
    
    def visit_count(self, asin: str) -> int:
        return self._seen_products.get(asin, 0)
    
    def mark_failed(self, asin: str) -> None:
        self._failed_products.add(asin)
    
    def is_failed(self, asin: str) -> bool:
        return asin in self._failed_products
    
    def add_action(self, action: str) -> None:
        self._seen_actions.append(action)
        if len(self._seen_actions) > 100:
            self._seen_actions = self._seen_actions[-50:]
    
    def is_loop_detected(self, action: str, window: int = 5) -> bool:
        if len(self._seen_actions) < window:
            return False
        recent = self._seen_actions[-window:]
        return recent.count(action) >= 2
    
    def reset(self) -> None:
        """Réinitialise le cache de session."""
        self._seen_products.clear()
        self._seen_actions.clear()
        self._failed_products.clear()
        self._option_retry_counts.clear()
        self._secondary_search_query = None
        self._top_asin_index = 0
        self._consecutive_option_failures = 0
        self._selected_color = None
        self._selected_size = None
        self._on_product_page = False
        self._current_asin = None
        self._required_color = None
        self._required_size = None

    def increment_option_retry(self, asin: str) -> int:
        """Incrémente le compteur de retry pour un produit et retourne le total."""
        self._option_retry_counts[asin] = self._option_retry_counts.get(asin, 0) + 1
        return self._option_retry_counts[asin]

    def get_option_retry_count(self, asin: str) -> int:
        return self._option_retry_counts.get(asin, 0)

    def increment_option_failure(self) -> int:
        self._consecutive_option_failures += 1
        return self._consecutive_option_failures

    def reset_option_failures(self) -> None:
        self._consecutive_option_failures = 0

    def get_option_failures(self) -> int:
        return self._consecutive_option_failures

    def set_secondary_search_query(self, query: str | None) -> None:
        self._secondary_search_query = query

    def pop_secondary_search_query(self) -> str | None:
        query = self._secondary_search_query
        self._secondary_search_query = None
        return query

    def pick_from_top_asins(self, asins: list[str]) -> str | None:
        if not asins:
            return None
        start_index = self._top_asin_index % len(asins)
        for offset in range(len(asins)):
            idx = (start_index + offset) % len(asins)
            asin = asins[idx]
            if not self.is_failed(asin) and not self.is_seen(asin):
                self._top_asin_index = idx + 1
                return asin
        self._top_asin_index = start_index + 1
        return None
    
    def get_penalty(self, asin: str) -> float:
        visits = self.visit_count(asin)
        if visits == 0:
            return 0.0
        if self.is_failed(asin):
            return -10.0
        return -1.0 * visits
    
    # NOUVEAU: Methods pour le tracking d'options
    def set_required_options(self, color: str | None, size: str | None) -> None:
        """Définit les options requises par l'instruction."""
        self._required_color = color
        self._required_size = size
    
    def enter_product_page(self, asin: str) -> None:
        """Appelé quand on entre sur une page produit."""
        if self._current_asin != asin:
            # Nouveau produit: reset les options
            self._current_asin = asin
            self._selected_color = None
            self._selected_size = None
            self._on_product_page = True
    
    def leave_product_page(self) -> None:
        """Appelé quand on quitte une page produit (retour en arrière)."""
        self._on_product_page = False
        self._current_asin = None
        self._selected_color = None
        self._selected_size = None
    
    def select_color(self, color: str) -> None:
        """Marque une couleur comme sélectionnée."""
        self._selected_color = color.lower()
        self.reset_option_failures()
    
    def select_size(self, size: str) -> None:
        """Marque une taille comme sélectionnée."""
        self._selected_size = size.lower()
        self.reset_option_failures()
    
    def is_color_selected(self) -> bool:
        """Vérifie si une couleur a été sélectionnée."""
        if not self._required_color:
            return True  # Pas de couleur requise
        return self._selected_color is not None
    
    def is_size_selected(self) -> bool:
        """Vérifie si une taille a été sélectionnée."""
        if not self._required_size:
            return True  # Pas de taille requise
        return self._selected_size is not None
    
    def are_all_options_selected(self) -> bool:
        """Vérifie si toutes les options requises sont sélectionnées."""
        return self.is_color_selected() and self.is_size_selected()
    
    def get_missing_option(self) -> str | None:
        """Retourne le type d'option manquante (color ou size)."""
        if self._required_color and not self._selected_color:
            return "color"
        if self._required_size and not self._selected_size:
            return "size"
        return None


# Cache global de session (partagé entre les appels)
_session_cache = SessionCache()


class Agent:
    def __init__(self):
        self._rng = random.Random(42)
        self._base_weights = self._load_scoring_weights()
        self._session_cache = _session_cache  # Utiliser le cache global

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
        
        # Reset cache au début d'une nouvelle session (step 0 avec search bar)
        if step == 0 and available.get("has_search_bar"):
            self._session_cache.reset()
            # Définir les options requises dès le début
            required_color = self._desired_option_value(instruction, "color")
            required_size = self._desired_option_value(instruction, "size")
            self._session_cache.set_required_options(required_color, required_size)

        if available.get("has_search_bar"):
            secondary_query = self._session_cache.pop_secondary_search_query()
            query = secondary_query or self._build_search_query(instruction)
            return {"type": "search", "query": query}

        results = self._extract_search_results(observation)
        if results:
            ranked_asins = self._rank_results(results, instruction, clickables, weights)
            top_asins = ranked_asins[:3]
            picked_asin = self._session_cache.pick_from_top_asins(top_asins) or (top_asins[0] if top_asins else None)
            if picked_asin:
                self._session_cache.mark_seen(picked_asin)
                action_str = f"click:{picked_asin}"
                
                if self._session_cache.is_loop_detected(action_str):
                    alt_asin = self._pick_alternative_result(results, picked_asin, clickables)
                    if alt_asin:
                        self._session_cache.mark_seen(alt_asin)
                        self._session_cache.add_action(f"click:{alt_asin}")
                        self._session_cache.enter_product_page(alt_asin)
                        return {"type": "click", "text": alt_asin}
                
                self._session_cache.add_action(action_str)
                self._session_cache.enter_product_page(picked_asin)
                return {"type": "click", "text": picked_asin}

        # Extraire les groupes d'options disponibles sur la page
        option_groups = self._extract_option_groups(observation)
        has_color_options = bool(option_groups.get("color"))
        has_size_options = bool(option_groups.get("size"))

        # NOUVEAU: Backtrack strict si les options requises ne sont pas disponibles ou matchables
        desired_color = self._desired_option_value(instruction, "color")
        desired_size = self._desired_option_value(instruction, "size")
        missing_option = self._session_cache.get_missing_option()
        back_to_search = self._pick_back_to_search(clickables)

        # NOUVEAU: Prioriser la taille quand les options sont rares
        if desired_color and desired_size:
            color_missing = self._session_cache._selected_color is None
            size_missing = self._session_cache._selected_size is None
            if color_missing and size_missing and has_color_options and has_size_options:
                color_count = len(option_groups.get("color", []))
                size_count = len(option_groups.get("size", []))
                if size_count <= 2 and color_count > 2:
                    missing_option = "size"
                elif color_count <= 2 and size_count > 2:
                    missing_option = "color"

        if missing_option == "color" and desired_color:
            can_match_color = has_color_options and self._can_match_option(desired_color, option_groups.get("color", []))
            if not can_match_color and back_to_search:
                if self._session_cache._current_asin:
                    self._session_cache.mark_failed(self._session_cache._current_asin)
                    self._session_cache.increment_option_retry(self._session_cache._current_asin)
                failures = self._session_cache.increment_option_failure()
                include_brand = failures < 2
                self._session_cache.set_secondary_search_query(
                    self._build_secondary_query(
                        instruction,
                        include_size=True,
                        include_brand=include_brand,
                    )
                )
                self._session_cache.leave_product_page()
                return {"type": "click", "text": back_to_search}

        if missing_option == "size" and desired_size:
            can_match_size = has_size_options and self._can_match_option(desired_size, option_groups.get("size", []))
            if not can_match_size and back_to_search:
                if self._session_cache._current_asin:
                    self._session_cache.mark_failed(self._session_cache._current_asin)
                    self._session_cache.increment_option_retry(self._session_cache._current_asin)
                failures = self._session_cache.increment_option_failure()
                include_brand = failures < 2
                self._session_cache.set_secondary_search_query(
                    self._build_secondary_query(
                        instruction,
                        include_size=True,
                        include_brand=include_brand,
                    )
                )
                self._session_cache.leave_product_page()
                return {"type": "click", "text": back_to_search}
        
        # Logique améliorée de sélection d'options
        # Priorité: sélectionner les options manquantes AVANT de cliquer sur Buy
        
        if missing_option == "color" and has_color_options:
            if desired_color:
                color_option = self._match_option_value(desired_color, option_groups.get("color", []), clickables)
                if color_option:
                    self._session_cache.select_color(color_option)
                    return {"type": "click", "text": color_option}
        
        if missing_option == "size" and has_size_options:
            if desired_size:
                size_option = self._match_option_value(desired_size, option_groups.get("size", []), clickables)
                if size_option:
                    self._session_cache.select_size(size_option)
                    return {"type": "click", "text": size_option}

        # NOUVEAU: Backtrack si une option reste manquante apres avoir tente les selections
        if missing_option and back_to_search:
            if self._session_cache._current_asin:
                self._session_cache.mark_failed(self._session_cache._current_asin)
                self._session_cache.increment_option_retry(self._session_cache._current_asin)
            include_size = missing_option == "size"
            failures = self._session_cache.increment_option_failure()
            include_brand = failures < 2
            self._session_cache.set_secondary_search_query(
                self._build_secondary_query(
                    instruction,
                    include_size=include_size,
                    include_brand=include_brand,
                )
            )
            self._session_cache.leave_product_page()
            return {"type": "click", "text": back_to_search}
        
        # Si on n'a pas réussi avec missing_option, essayer l'ancienne logique
        option_pick = self._pick_page_option(clickables, instruction, option_groups, step)
        if option_pick:
            # Tracker ce qui a été sélectionné
            option_lower = option_pick.lower()
            if option_groups.get("color") and any(self._normalize(c) == self._normalize(option_pick) for c in option_groups.get("color", [])):
                self._session_cache.select_color(option_pick)
            elif option_groups.get("size") and any(self._normalize(s) == self._normalize(option_pick) for s in option_groups.get("size", [])):
                self._session_cache.select_size(option_pick)
            return {"type": "click", "text": option_pick}

        # NOUVEAU: Logique Buy stricte - refuser l'achat si option requise manquante
        buy_button = self._pick_buy(clickables)
        if buy_button:
            should_buy = False
            
            # Si pas d'options sur la page, on peut acheter
            if not has_color_options and not has_size_options:
                should_buy = True
            # Si toutes les options requises sont sélectionnées
            elif self._session_cache.are_all_options_selected():
                should_buy = True
            # Pas d'options requises dans l'instruction
            elif not self._session_cache._required_color and not self._session_cache._required_size:
                should_buy = True
            
            if should_buy:
                return {"type": "click", "text": buy_button}

        # Fallback sur les options correspondantes
        desired_values = self._desired_values(instruction)
        matched_option = self._pick_matching_option(clickables, desired_values)
        if matched_option:
            return {"type": "click", "text": matched_option}

        asin_click = self._pick_asin(clickables)
        if asin_click:
            self._session_cache.enter_product_page(asin_click)
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
        """Construit une requête de recherche optimisée à partir de l'instruction.
        
        Priorité: genre > type de produit > marque > attributs clés
        Exclut: tailles, attributs de soin (machine wash, etc.)
        """
        # Extraire le type de produit principal
        product_type = self._extract_product_type(instruction)
        target_types = self._instruction_product_types(instruction)
        gender = self._instruction_gender(instruction)
        
        tokens: list[str] = []
        seen_normalized: set[str] = set()
        
        def add_token(token: str) -> None:
            """Ajoute un token s'il n'est pas déjà présent."""
            if not token:
                return
            norm = self._normalize(token)
            if norm and norm not in seen_normalized:
                seen_normalized.add(norm)
                tokens.append(token)
        
        # 1. Genre (très important pour le filtrage)
        if gender:
            add_token(gender)
        
        # 2. Type de produit (essentiel)
        if target_types:
            preferred_type = self._preferred_type_token(target_types, instruction)
            if preferred_type:
                add_token(preferred_type)
        elif product_type:
            add_token(product_type)
        
        # 3. Couleur (si spécifiée) - utile pour améliorer le top-1
        desired_color = self._desired_option_value(instruction, "color")
        if desired_color:
            readable_color = extract_readable_color(desired_color)
            add_token(readable_color)

        # 4. Marque (si spécifiée)
        for brand in self._desired_brand_tokens(instruction):
            add_token(brand)

        # 5. Materiau (si specifie)
        for material in self._desired_material_tokens(instruction):
            add_token(material)

        # 6. Attributs pertinents (max 2, exclure attributs de soin)
        key_attributes = self._extract_key_attributes(instruction)
        for attr in key_attributes[:2]:
            add_token(attr)
        
        query = " ".join(tokens[:6])  # Max 6 tokens
        if query:
            return query
        return self._default_query(instruction)

    def _build_secondary_query(
        self,
        instruction: str,
        *,
        include_size: bool = False,
        include_brand: bool = True,
    ) -> str:
        """Construit une requete secondaire (type + couleur + marque + taille)."""
        tokens: list[str] = []
        seen: set[str] = set()

        def add_token(token: str) -> None:
            if not token:
                return
            norm = self._normalize(token)
            if norm and norm not in seen:
                seen.add(norm)
                tokens.append(token)

        target_types = self._instruction_product_types(instruction)
        if target_types:
            preferred_type = self._preferred_type_token(target_types, instruction)
            if preferred_type:
                add_token(preferred_type)

        desired_color = self._desired_option_value(instruction, "color")
        if desired_color:
            add_token(extract_readable_color(desired_color))
        if include_brand:
            for brand in self._desired_brand_tokens(instruction):
                add_token(brand)
        if include_size:
            desired_size = self._desired_option_value(instruction, "size")
            if desired_size:
                add_token(self._preferred_size_token(desired_size))

        query = " ".join(tokens[:4]).strip()
        return query if query else self._build_search_query(instruction)

    def _preferred_type_token(self, target_types: list[str], instruction: str) -> str | None:
        instruction_lower = instruction.lower()
        for type_key in target_types:
            for phrase in PRODUCT_TYPE_SYNONYMS.get(type_key, []):
                if phrase in instruction_lower:
                    return phrase
        if target_types:
            type_key = target_types[0]
            return PRODUCT_TYPE_SYNONYMS.get(type_key, [type_key])[0]
        return None

    def _preferred_size_token(self, size: str) -> str:
        candidates = self._expand_size_tokens(size)
        for token in sorted(candidates, key=len):
            if len(token) <= 3:
                return token
        return size
    
    def _extract_product_type(self, instruction: str) -> str:
        """Extrait le type de produit de l'instruction."""
        # Pattern: "Find me [modifiers] [type de produit] with..."
        # Matches: "Find me machine wash, wash cold women's fashion hoodies..."
        match = re.search(
            r"find me\s+(?:machine wash|hand wash)?[\s,]*(?:wash cold|dry clean)?[\s,]*(.+?)(?:\s+with|\s+for|$)",
            instruction,
            re.IGNORECASE
        )
        if match:
            product = match.group(1).strip()
            # Nettoyer les virgules initiales/finales
            product = product.strip(',').strip()
            # Nettoyer les préfixes de genre déjà extraits
            product = re.sub(r"^(men's|women's|men|women)\s+", "", product, flags=re.IGNORECASE)
            # Nettoyer les attributs de soin
            product = re.sub(r"\b(machine wash|hand wash|wash cold|dry clean|tumble dry)\b", "", product, flags=re.IGNORECASE)
            # Nettoyer "for dry clean, tumble dry" patterns
            product = re.sub(r"\s+for\s*$", "", product, flags=re.IGNORECASE)
            product = re.sub(r"\s+", " ", product).strip()
            return product
        return ""
    
    def _extract_key_attributes(self, instruction: str) -> list[str]:
        """Extrait les attributs clés pour la recherche (exclut couleur, taille, soin)."""
        # Attributs à exclure de la requête de recherche
        exclude_patterns = [
            r"color", r"size", r"price", r"fit type",
            r"machine wash", r"hand wash", r"wash cold", 
            r"dry clean", r"tumble dry", r"iron",
            r"\$\d+", r"\d+\.\d+ dollars"
        ]
        exclude_regex = re.compile("|".join(exclude_patterns), re.IGNORECASE)
        
        # Attributs intéressants pour la recherche
        good_patterns = [
            r"long sleeve", r"short sleeve", r"sleeveless",
            r"cotton", r"polyester", r"leather", r"wool", r"silk",
            r"stretch", r"slim fit", r"regular fit", r"relaxed",
            r"button", r"zip", r"crew neck", r"v-neck",
            r"casual", r"formal", r"vintage", r"classic"
        ]
        
        results: list[str] = []
        instruction_lower = instruction.lower()
        
        for pattern in good_patterns:
            if re.search(pattern, instruction_lower):
                # Éviter les doublons
                if pattern not in [r.lower() for r in results]:
                    results.append(pattern.replace(r"", ""))
        
        return results[:3]  # Max 3 attributs

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
                values.extend(self._split_compound_values(match.group(1).strip()))
        for phrase in self._desired_material_tokens(instruction):
            values.append(phrase)
        for phrase in ("long sleeve", "short sleeve", "cotton", "polyester", "spandex", "leather"):
            if phrase in instruction.lower():
                values.append(phrase)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            norm = self._normalize(value)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(value)
        return deduped

    def _pick_matching_option(self, clickables: list[str], desired_values: list[str]) -> str | None:
        """Trouve une option clickable qui correspond aux valeurs désirées.
        
        Gère les couleurs composées comme "black | blue".
        """
        if not desired_values:
            return None
        
        for desired in desired_values:
            desired_lower = desired.lower()
            desired_parts = self._split_compound_color(desired)
            
            for candidate in clickables:
                candidate_lower = candidate.lower()
                candidate_parts = self._split_compound_color(candidate)
                
                # Correspondance directe
                if desired_lower in candidate_lower:
                    return candidate
                
                # Correspondance de couleur composée
                for desired_part in desired_parts:
                    for cand_part in candidate_parts:
                        if desired_part.lower() == cand_part.lower():
                            return candidate
                        if desired_part.lower() in cand_part.lower():
                            return candidate
        return None

    def _pick_buy(self, clickables: list[str]) -> str | None:
        for candidate in clickables:
            if "buy" in candidate.lower():
                return candidate
        return None

    def _pick_back_to_search(self, clickables: list[str]) -> str | None:
        back_tokens = {
            "back to search",
            "< prev",
            "previous",
        }
        for candidate in clickables:
            if candidate.lower() in back_tokens:
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
            results.append({"asin": chunk, "title": title, "price": price, "rank": len(results)})
        return results

    def _rank_results(
        self,
        results: list[dict[str, str]],
        instruction: str,
        clickables: list[str],
        weights: dict[str, float],
    ) -> list[str]:
        desired_values = self._desired_values(instruction)
        desired_phrases = self._desired_phrases_for_scoring(instruction)
        price_lower, price_upper = self._parse_price_bounds(instruction)
        keywords = self._instruction_keywords(instruction)
        gender = self._instruction_gender(instruction)
        brand_tokens = self._desired_brand_tokens(instruction)
        material_tokens = self._desired_material_tokens(instruction)
        query = self._query_from_instruction(instruction)
        query_norm = self._normalize(query) if query else ""
        target_types = self._instruction_product_types(instruction)
        scored: list[tuple[str, float]] = []

        for result in results:
            asin = result["asin"]
            clickable = self._match_clickable(clickables, asin)
            if not clickable:
                continue
            if self._session_cache.is_failed(asin) or self._session_cache.visit_count(asin) > 0:
                continue
            title_raw = result["title"]
            title = title_raw.lower()
            title_norm = self._normalize(title)
            price = self._parse_price(result.get("price", ""))
            rank = result.get("rank", 0)
            score = 0.0
            
            # Compteurs pour le bonus de correspondance complète
            matches_count = 0
            total_criteria = 0
            if target_types:
                total_criteria += 1
            if gender:
                total_criteria += 1
            if brand_tokens:
                total_criteria += 1
            if material_tokens:
                total_criteria += 1
            if price_lower is not None or price_upper is not None:
                total_criteria += 1
            
            if query_norm:
                score += SequenceMatcher(None, query_norm, title_norm).ratio() * weights["query_similarity"]

            # Bonus de rang: privilégier les premiers résultats
            score += weights.get("rank_bonus", 0.0) * (1.0 / (1.0 + float(rank)))
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
                target_present = any(self._type_presence(title_norm, type_key) for type_key in target_types)
                if not target_present:
                    non_target_present = any(
                        self._type_presence(title_norm, type_key)
                        for type_key in PRODUCT_TYPE_SYNONYMS
                        if type_key not in target_types
                    )
                    if non_target_present:
                        continue
                if target_present:
                    score += weights["type_match"]
                    matches_count += 1
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
                    matches_count += 1
                elif self._gender_mismatch(title, gender):
                    score += weights["gender_mismatch"]
            if brand_tokens:
                if self._tokens_in_text(brand_tokens, title_norm):
                    score += weights["brand_match"]
                    matches_count += 1
                else:
                    score += weights["brand_mismatch"]
            if material_tokens:
                if self._tokens_in_text(material_tokens, title_norm):
                    score += weights["material_match"]
                    matches_count += 1
                else:
                    score += weights["material_mismatch"]
            if price_lower is not None or price_upper is not None:
                if price is not None:
                    if self._price_in_bounds(price, price_lower, price_upper):
                        score += weights["price_match"]
                        matches_count += 1
                    else:
                        score += weights["price_mismatch"]
                else:
                    score += weights.get("price_missing", 0.0)
            
            # NOUVEAU: Appliquer bonus de correspondance complète
            if total_criteria > 0:
                match_ratio = matches_count / total_criteria
                if match_ratio >= 1.0:
                    score += weights.get("full_match_bonus", 0.0)
                elif match_ratio >= 0.75:
                    score += weights.get("partial_match_bonus", 0.0)
            
            # NOUVEAU: Pénalité pour les produits déjà vus (évite les boucles)
            cache_penalty = self._session_cache.get_penalty(asin)
            score += cache_penalty * weights.get("cache_penalty_weight", 1.0)
            
            scored.append((clickable, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [asin for asin, _ in scored]



    def _pick_best_result(
        self,
        results: list[dict[str, str]],
        instruction: str,
        clickables: list[str],
        weights: dict[str, float],
    ) -> str | None:
        ranked = self._rank_results(results, instruction, clickables, weights)
        return ranked[0] if ranked else None
    
    def _pick_alternative_result(
        self,
        results: list[dict[str, str]],
        exclude_asin: str,
        clickables: list[str],
    ) -> str | None:
        """Trouve un produit alternatif (pour éviter les boucles)."""
        for result in results:
            asin = result["asin"]
            if asin == exclude_asin:
                continue
            if self._session_cache.is_failed(asin) or self._session_cache.visit_count(asin) > 0:
                continue
            clickable = self._match_clickable(clickables, asin)
            if clickable:
                return clickable
        return None

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

    def _parse_price_bounds(self, instruction: str) -> tuple[float | None, float | None]:
        text = instruction.lower()
        match = re.search(
            r"(?:price\s+between|between)\s+\$?([0-9]+(?:\.[0-9]+)?)\s+(?:and|to)\s+\$?([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if match:
            return float(match.group(1)), float(match.group(2))
        match = re.search(
            r"(?:price\s+lower than|price\s+less than|price\s+under|price\s+below|lower than|less than|under|below)\s+\$?([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if match:
            return None, float(match.group(1))
        match = re.search(
            r"(?:price\s+greater than|price\s+over|price\s+above|greater than|over|above)\s+\$?([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if match:
            return float(match.group(1)), None
        return None, None

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
            part = re.sub(r"(?:color|size|brand|material)\s*:\s*", "", part)
            if not part:
                continue
            phrases.append(part)
        return phrases

    def _query_from_instruction(self, instruction: str) -> str:
        """Extrait la requête brute de l'instruction (legacy, utilisé par d'autres méthodes)."""
        instruction = instruction.strip()
        # Pattern amélioré: ignore les attributs de soin avant le type de produit
        match = re.search(
            r"find me\s+(?:machine wash|hand wash|wash cold)?\s*(.+?)(?:\s+with|\s+for|,|$)",
            instruction,
            re.IGNORECASE
        )
        if match:
            result = match.group(1).strip()
            # Nettoyer les attributs de soin résiduels
            result = re.sub(r"\b(machine wash|hand wash)\b,?\s*", "", result, flags=re.IGNORECASE)
            return result.strip()
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
        # Poids optimisés via weight_scan.py (trial 1 - avg_reward=0.6267)
        return {
            # Similarité et correspondance textuelle
            "query_similarity": 2.345,
            "keyword": 1.929,
            "desired_value": 1.31,
            "desired_phrase": 3.888,
            
            # Type de produit (le plus important)
            "type_match": 5.243,
            "type_missing": -3.146,
            "non_target_type": -2.312,
            "mismatch_type": -3.983,
            
            # Genre
            "gender_match": 2.617,
            "gender_mismatch": -4.042,
            
            # Prix
            "price_match": 2.098,
            "price_mismatch": -3.792,
            "price_missing": -0.462,
            
            # Marque
            "brand_match": 3.086,
            "brand_mismatch": -2.213,
            
            # Matériau
            "material_match": 2.575,
            "material_mismatch": -2.059,
            
            # Bonus de correspondance complète
            "full_match_bonus": 5.541,
            "partial_match_bonus": 1.913,
            
            # Cache de session (évite les boucles)
            "cache_penalty_weight": 2.5,

            # Bonus de rang (résultats en haut de liste)
            "rank_bonus": 4.0,
        }

    def _filter_query_attributes(
        self,
        phrases: list[str],
        *,
        desired_color: str | None,
        desired_size: str | None,
        desired_brand: str | None,
        desired_material: str | None,
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
            if desired_brand and phrase == desired_brand.lower():
                continue
            if desired_material and phrase == desired_material.lower():
                continue
            if len(phrase) > 40:
                continue
            filtered.append(phrase)
            if len(filtered) >= 2:
                break
        return filtered

    def _split_compound_values(self, value: str) -> list[str]:
        parts = re.split(r"/|,| and ", value)
        return [part.strip() for part in parts if part.strip()]

    def _desired_brand_tokens(self, instruction: str) -> list[str]:
        tokens: list[str] = []
        brand = self._desired_option_value(instruction, "brand")
        if brand:
            tokens.extend(self._split_compound_values(brand))
        match = re.search(r"brand\s+is\s+([^,]+)", instruction, re.IGNORECASE)
        if match:
            tokens.extend(self._split_compound_values(match.group(1).strip()))
        return [token.lower() for token in tokens if token]

    def _desired_material_tokens(self, instruction: str) -> list[str]:
        tokens: list[str] = []
        material = self._desired_option_value(instruction, "material")
        if material:
            tokens.extend(self._split_compound_values(material))
        for pattern in (r"made of\s+([^,]+)", r"made from\s+([^,]+)"):
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                tokens.extend(self._split_compound_values(match.group(1).strip()))
        return [token.lower() for token in tokens if token]

    def _tokens_in_text(self, tokens: list[str], text_norm: str) -> bool:
        for token in tokens:
            token_norm = self._normalize(token)
            if token_norm and token_norm in text_norm:
                return True
        return False

    def _price_in_bounds(
        self,
        price: float,
        lower: float | None,
        upper: float | None,
    ) -> bool:
        if lower is not None and price < lower:
            return False
        if upper is not None and price > upper:
            return False
        return True

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
        """Sélectionne une option de page (couleur ou taille) de manière intelligente.
        
        AMÉLIORATION: Utilise le cache de session pour savoir quelles options 
        ont déjà été sélectionnées, au lieu de se baser sur step % 2.
        """
        color = self._desired_option_value(instruction, "color")
        size = self._desired_option_value(instruction, "size")
        color_option = None
        size_option = None
        
        if color:
            color_option = self._match_option_value(color, option_groups.get("color", []), clickables)
        if size:
            size_option = self._match_option_value(size, option_groups.get("size", []), clickables)
        
        # NOUVEAU: Priorité aux options non sélectionnées
        color_already_selected = self._session_cache._selected_color is not None
        size_already_selected = self._session_cache._selected_size is not None

        # Si les deux sont encore manquantes, prioriser celle qui est plus rare
        if color_option and size_option and not color_already_selected and not size_already_selected:
            color_count = len(option_groups.get("color", []))
            size_count = len(option_groups.get("size", []))
            if size_count <= 2 and color_count > 2:
                return size_option
            if color_count <= 2 and size_count > 2:
                return color_option
        
        # Si couleur requise mais pas encore sélectionnée
        if color_option and not color_already_selected:
            return color_option
        
        # Si taille requise mais pas encore sélectionnée  
        if size_option and not size_already_selected:
            return size_option
        
        # Fallback: ancienne logique basée sur step (au cas où le tracking échoue)
        if color_option and not size_option:
            return color_option if step <= 2 else None
        if size_option and not color_option:
            return size_option if step <= 2 else None
        if color_option and size_option:
            if step <= 3:
                # Préférer couleur d'abord car elle est souvent plus discriminante
                if not color_already_selected:
                    return color_option
                return size_option
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
        """Match une valeur désirée avec les options disponibles.
        
        Gère les couleurs composées comme "black | blue" ou "red/blue".
        Gère les couleurs cryptiques comme "xnj-tshirt348-black" → "black".
        Utilise le fuzzy matching Levenshtein pour les couleurs mal orthographiées.
        """
        # NOUVEAU: Extraire la couleur lisible depuis un pattern cryptique
        readable_color = extract_readable_color(desired)
        
        desired_norm = self._normalize(desired)
        readable_norm = self._normalize(readable_color)
        desired_parts = self._split_compound_color(desired)
        readable_parts = self._split_compound_color(readable_color)
        # NEW: Split colors into words
        desired_words = [w.strip() for w in desired.lower().split() if w.strip()]
        readable_words = [w.strip() for w in readable_color.lower().split() if w.strip()]
        
        # 0. NOUVEAU: Si on a extrait une couleur lisible, l'utiliser d'abord
        if readable_color != desired.lower().strip():
            for option in options:
                opt_norm = self._normalize(option)
                # Correspondance exacte avec la couleur extraite
                if opt_norm == readable_norm:
                    return self._match_clickable(clickables, option) or option
                # Correspondance partielle avec la couleur extraite
                if readable_norm in opt_norm or opt_norm in readable_norm:
                    return self._match_clickable(clickables, option) or option
            # Chercher parmi les mots de la couleur lisible
            for word in readable_words:
                for option in options:
                    if self._normalize(option) == self._normalize(word):
                        return self._match_clickable(clickables, option) or option
        
        # 1. Correspondance exacte
        for option in options:
            if self._normalize(option) == desired_norm:
                return self._match_clickable(clickables, option) or option
        
        # 2. Correspondance de couleur composée (ex: "black" match "black | blue")
        for option in options:
            option_parts = self._split_compound_color(option)
            # Si la couleur désirée est dans les parties de l'option composée
            for desired_part in desired_parts + readable_parts:
                desired_part_norm = self._normalize(desired_part)
                for option_part in option_parts:
                    if self._normalize(option_part) == desired_part_norm:
                        return self._match_clickable(clickables, option) or option
        
        # 2b. Check if any word of desired matches exactly (e.g., "navy blue" -> match "navy")
        for desired_word in desired_words + readable_words:
            for option in options:
                if self._normalize(option) == self._normalize(desired_word):
                    return self._match_clickable(clickables, option) or option
        
        # 3. Correspondance partielle (substring) - both directions
        for option in options:
            opt_norm = self._normalize(option)
            # Check if desired is IN option OR option is IN desired
            if desired_norm in opt_norm or opt_norm in desired_norm:
                return self._match_clickable(clickables, option) or option
            # Also check readable color
            if readable_norm in opt_norm or opt_norm in readable_norm:
                return self._match_clickable(clickables, option) or option
        
        # 4. Recherche dans les clickables
        for candidate in clickables:
            candidate_parts = self._split_compound_color(candidate)
            for desired_part in desired_parts + readable_parts:
                desired_part_norm = self._normalize(desired_part)
                for cand_part in candidate_parts:
                    if self._normalize(cand_part) == desired_part_norm:
                        return candidate
            cand_norm = self._normalize(candidate)
            if desired_norm in cand_norm or cand_norm in desired_norm:
                return candidate
            if readable_norm in cand_norm or cand_norm in readable_norm:
                return candidate
        
        # 5. NOUVEAU: Fuzzy matching pour les tailles (large -> L, etc.)
        size_match = fuzzy_match_size(desired, options)
        if size_match:
            return self._match_clickable(clickables, size_match) or size_match
        
        # 6. Fuzzy matching Levenshtein pour les couleurs mal orthographiées
        fuzzy_match = fuzzy_match_color(readable_color, options, threshold=0.75)
        if fuzzy_match:
            return self._match_clickable(clickables, fuzzy_match) or fuzzy_match
        
        # 7. Fuzzy matching dans les clickables
        fuzzy_match = fuzzy_match_color(readable_color, clickables, threshold=0.75)
        if fuzzy_match:
            return fuzzy_match
        
        # 8. Si aucune correspondance, prendre la première option disponible
        #    (mieux vaut sélectionner une option que de ne rien faire)
        if options:
            first_option = options[0]
            return self._match_clickable(clickables, first_option) or first_option
        
        return None

    
    def _can_match_option(
        self,
        desired: str,
        options: list[str],
    ) -> bool:
        """Vérifie si la valeur désirée peut être matchée avec une option.
        
        Similaire à _match_option_value mais sans fallback - retourne True/False.
        Utilisé pour déterminer si on doit revenir en arrière.
        """
        if not options:
            return False
        
        # Extraire la couleur lisible depuis un pattern cryptique
        readable_color = extract_readable_color(desired)
        
        desired_norm = self._normalize(desired)
        readable_norm = self._normalize(readable_color)
        desired_parts = self._split_compound_color(desired)
        readable_parts = self._split_compound_color(readable_color)
        desired_words = [w.strip() for w in desired.lower().split() if w.strip()]
        readable_words = [w.strip() for w in readable_color.lower().split() if w.strip()]
        
        # Vérifier toutes les stratégies de matching
        for option in options:
            opt_norm = self._normalize(option)
            option_parts = self._split_compound_color(option)
            
            # Correspondance exacte
            if opt_norm == desired_norm or opt_norm == readable_norm:
                return True
            
            # Correspondance partielle (substring)
            if desired_norm in opt_norm or opt_norm in desired_norm:
                return True
            if readable_norm in opt_norm or opt_norm in readable_norm:
                return True
            
            # Correspondance de couleur composée
            for desired_part in desired_parts + readable_parts:
                desired_part_norm = self._normalize(desired_part)
                for option_part in option_parts:
                    if self._normalize(option_part) == desired_part_norm:
                        return True
            
            # Correspondance par mot
            for word in desired_words + readable_words:
                if self._normalize(option) == self._normalize(word):
                    return True
        
        # Fuzzy matching pour les tailles
        size_match = fuzzy_match_size(desired, options)
        if size_match:
            return True
        
        # Fuzzy matching Levenshtein pour les couleurs
        fuzzy_match = fuzzy_match_color(readable_color, options, threshold=0.75)
        if fuzzy_match:
            return True
        
        return False
    
    def _split_compound_color(self, value: str) -> list[str]:
        """Sépare une couleur composée en ses parties.
        
        Ex: "black | blue" -> ["black", "blue"]
            "red/white" -> ["red", "white"]
            "dark heather" -> ["dark heather"]
        """
        # Séparateurs de couleurs composées
        parts = re.split(r"\s*[|/]\s*", value)
        return [p.strip() for p in parts if p.strip()]

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())
