"""
veil/polisher.py
----------------
Thin wrapper around the external polishing LLM (Gemini).

This is the component that VEIL GUARD evaluates.  It receives only
the sanitised (redacted) prompt — never raw PII.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted

logger = logging.getLogger(__name__)


_POLISH_SYSTEM_PROMPT = """\
You are a professional text editor. Your job is to polish and improve the \
clarity, grammar, and tone of messages while preserving their exact meaning.

Important rules:
1. NEVER alter or remove any placeholder tokens such as <PERSON_1>, <PHONE_NUMBER_1>, etc.
2. Preserve every placeholder exactly as it appears — same spelling, same angle brackets.
3. Do not invent new information or add details not present in the original.
4. Output only the polished message — no preamble, no explanation.
"""

# Fallback chain — tried in order if the primary model is unavailable
_MODEL_FALLBACK_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-1.5-flash",
]


class VeilPolisher:
    """
    Sends a redacted prompt to the external LLM and returns its polished text.
    Automatically falls back through the model chain on 503 / quota errors.

    Parameters
    ----------
    api_key
        Google AI API key.  Defaults to ``GEMINI_API_KEY`` env var.
    model_name
        Primary model for polishing. Falls back to _MODEL_FALLBACK_CHAIN on error.
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model_name:  str           = "gemini-3.8-flash",
        temperature: float         = 0.7,
    ):
        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=_key)

        self._api_key    = _key
        self._model_name = model_name
        self._temperature = temperature

        # Build fallback list: primary first, then rest of chain (deduped)
        self._model_chain = [model_name] + [
            m for m in _MODEL_FALLBACK_CHAIN if m != model_name
        ]

        self._model = self._build_model(model_name)

    def _build_model(self, model_name: str):
        return genai.GenerativeModel(
            model_name         = model_name,
            system_instruction = _POLISH_SYSTEM_PROMPT,
            generation_config  = genai.GenerationConfig(temperature=self._temperature),
        )

    def polish(self, redacted_text: str, extra_instruction: str = "") -> str:
        """
        Polish *redacted_text* and return the proposed response.
        Automatically retries through the model fallback chain on 503/quota errors.
        """
        prompt = redacted_text
        if extra_instruction:
            prompt += f"\n\nAdditional instruction: {extra_instruction}"

        last_error = None
        for model_name in self._model_chain:
            try:
                if model_name != self._model_name:
                    logger.warning("VEIL POLISHER: falling back to model %s", model_name)
                    model = self._build_model(model_name)
                else:
                    model = self._model

                response = model.generate_content(prompt)
                return response.text.strip()

            except (ServiceUnavailable, ResourceExhausted) as e:
                logger.warning(
                    "VEIL POLISHER: model %s unavailable (%s), trying next...",
                    model_name, e,
                )
                last_error = e
                continue

        raise RuntimeError(
            f"All models in fallback chain exhausted. Last error: {last_error}"
        )
