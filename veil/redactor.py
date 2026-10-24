"""
veil/redactor.py
----------------
VEIL's local privacy engine.

Uses Microsoft Presidio (+ spaCy en_core_web_lg) to detect PII, replaces
each detected entity with a typed, indexed placeholder, and stores the
original values in an in-memory VaultSnapshot that never leaves this
process unless explicitly restored.

No external LLM is involved in this layer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from veil.models import VaultEntry, VaultSnapshot


# ---------------------------------------------------------------------------
# Lazy import helpers — Presidio is optional at import time so the module
# can be loaded even in environments where the package isn't installed yet.
# ---------------------------------------------------------------------------

def _load_presidio():
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
        return AnalyzerEngine(), AnonymizerEngine(), OperatorConfig
    except ImportError as exc:
        raise RuntimeError(
            "Presidio is not installed. Run: pip install presidio-analyzer presidio-anonymizer"
        ) from exc


# Entity types we want to detect and redact
_SUPPORTED_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "ORGANIZATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "NRP",               # National Registration / ID numbers
    "MEDICAL_LICENSE",
]


class VeilRedactor:
    """
    Detects and redacts PII from a raw user prompt.

    Usage
    -----
    >>> redactor = VeilRedactor()
    >>> sanitised, vault = redactor.redact("Call me at 9876543210, Priya.")
    >>> sanitised
    'Call me at <PHONE_NUMBER_1>, <PERSON_1>.'
    >>> vault.restore(sanitised)
    'Call me at 9876543210, Priya.'
    """

    def __init__(self, language: str = "en", score_threshold: float = 0.4):
        self._language        = language
        self._score_threshold = score_threshold
        self._analyzer, self._anonymizer, self._OperatorConfig = _load_presidio()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> tuple[str, VaultSnapshot]:
        """
        Detect PII in *text*, replace each entity with a typed placeholder,
        and return (sanitised_text, vault).

        The vault maps every placeholder back to its original value.
        """
        results = self._analyzer.analyze(
            text=text,
            entities=_SUPPORTED_ENTITIES,
            language=self._language,
            score_threshold=self._score_threshold,
        )

        if not results:
            return text, VaultSnapshot()

        # Sort by start position so we can assign consistent indices
        results_sorted = sorted(results, key=lambda r: r.start)

        # Build a counter per entity type for deterministic numbering
        counters: dict[str, int] = defaultdict(int)

        vault_entries: list[VaultEntry] = []
        offset       = 0
        sanitised    = text

        for result in results_sorted:
            original_value = text[result.start : result.end]
            entity_type    = result.entity_type

            counters[entity_type] += 1
            placeholder = f"<{entity_type}_{counters[entity_type]}>"

            # Perform replacement in the running sanitised string,
            # adjusting for accumulated offset caused by prior replacements.
            adj_start = result.start + offset
            adj_end   = result.end   + offset

            sanitised = sanitised[:adj_start] + placeholder + sanitised[adj_end:]
            offset   += len(placeholder) - (result.end - result.start)

            vault_entries.append(
                VaultEntry(
                    placeholder    = placeholder,
                    original_value = original_value,
                    entity_type    = entity_type,
                )
            )

        return sanitised, VaultSnapshot(entries=vault_entries)

    def validate_placeholders(self, text: str, vault: VaultSnapshot) -> bool:
        """
        Return True if every placeholder that was registered for this
        request is still present in *text* and no foreign/altered
        placeholder tokens appear.
        """
        for placeholder in vault.placeholders:
            if placeholder not in text:
                return False

        # Also check that no raw placeholder-like tokens outside the vault exist
        all_tokens_in_text = set(re.findall(r"<[A-Z_]+_\d+>", text))
        registered_tokens  = set(vault.placeholders)
        unexpected = all_tokens_in_text - registered_tokens
        return len(unexpected) == 0
