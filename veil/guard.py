"""
veil/guard.py
-------------
VEIL GUARD — the AI agent that evaluates the external LLM's response.

VEIL GUARD receives:
  • The original (redacted) prompt sent to the external LLM
  • The LLM's proposed response
  • The list of expected placeholders (from the vault)

It returns a GuardDecision with one of four verdicts:
  • APPROVE  — Case 1 (implemented here): response satisfactorily answers
               the prompt, placeholders intact, safe to restore and return.
  • RETRY    — Case 2: quality/tone problem, fixable with better instruction.
  • ASK_USER — Case 3: ambiguity that only the user can resolve.
  • REJECT   — Case 4: privacy violation, discard the response entirely.

The guard calls Gemini (via the google-generativeai SDK) and asks it to
return a strict JSON object matching the GuardDecision schema.  A
deterministic post-processing step validates placeholder integrity
independently of the LLM—so even a hallucinating guard cannot approve
a response that has dropped or mutated a placeholder.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted
from pydantic import ValidationError

from veil.models import DecisionCode, GuardDecision, VaultSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — baked into every VEIL GUARD call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are VEIL GUARD, a specialised AI evaluator embedded in a privacy-preserving \
text-polishing pipeline.

Your ONLY job is to assess whether the external LLM's response adequately and \
safely answers the user's (redacted) request. You must return a single, valid \
JSON object — no markdown fences, no extra text — that conforms exactly to \
this schema:

{
  "decision":              "APPROVE" | "RETRY" | "ASK_USER" | "REJECT",
  "reason_code":           "<SHORT_LABEL>",
  "feedback":              "<human-readable explanation or retry instruction>",
  "confidence":            <float 0.0–1.0>,
  "sensitive_data_needed": <true | false>
}

Evaluation criteria
-------------------
1. TASK COMPLETION   — Does the response fully address the user's request?
2. MEANING FIDELITY  — Is the core meaning preserved without distortion?
3. WRITING QUALITY   — Is it grammatical, clear, and appropriately toned?
4. PLACEHOLDER SAFETY — Are all <TYPE_N> placeholders retained unchanged?
                        (Do NOT check whether every placeholder is present;
                         that is handled by a separate deterministic check.)
5. PRIVACY GUARD     — Does the response avoid introducing any new personal
                       information or asking for sensitive data unnecessarily?

Decision rules
--------------
• Use APPROVE   when ALL five criteria pass.
• Use RETRY     when the response fails criteria 1–3 but NOT 4 or 5.
                Set sensitive_data_needed = false unless real PII is
                genuinely required (which is rare for text polishing).
• Use ASK_USER  when criteria 1 fails due to ambiguity that only the user
                can resolve.
• Use REJECT    when criteria 4 or 5 fails.

Never approve a response that has altered or dropped a placeholder.
Never set sensitive_data_needed = true for a writing-quality problem.
"""


# ---------------------------------------------------------------------------
# VEIL GUARD agent
# ---------------------------------------------------------------------------

class VeilGuard:
    """
    AI agent that evaluates the external LLM's response and returns a
    structured GuardDecision.

    Parameters
    ----------
    api_key
        Google AI API key.  Defaults to the ``GEMINI_API_KEY`` environment
        variable.
    model_name
        Gemini model to use for guard evaluation.  A capable but economical
        model is preferred because VEIL GUARD is called on every response.
    temperature
        Low temperature keeps the JSON output deterministic.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model_name: str           = "gemini-3.8-flash",
        temperature: float        = 0.1,
    ):
        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Export it as an environment variable or pass api_key=."
            )
        genai.configure(api_key=_key)

        # Fallback chain for guard evaluation
        self._model_chain = [model_name] + [
            m for m in [
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemini-1.5-flash",
            ] if m != model_name
        ]
        self._temperature = temperature
        self._model = self._build_model(model_name)

    def _build_model(self, model_name: str):
        return genai.GenerativeModel(
            model_name         = model_name,
            system_instruction = _SYSTEM_PROMPT,
            generation_config  = genai.GenerationConfig(
                temperature        = self._temperature,
                response_mime_type = "application/json",
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        original_prompt:       str,
        llm_response:          str,
        vault:                 VaultSnapshot,
        requested_tone:        str = "polite and professional",
    ) -> GuardDecision:
        """
        Evaluate *llm_response* against *original_prompt* and return a
        GuardDecision.

        A deterministic placeholder-integrity check is applied AFTER the
        agent returns its decision.  If the check fails, the decision is
        overridden to REJECT regardless of what the agent said.

        Parameters
        ----------
        original_prompt
            The redacted prompt that was sent to the external polishing LLM.
        llm_response
            The response returned by the external LLM.
        vault
            The VaultSnapshot for this request (used for placeholder checks).
        requested_tone
            Tone the user asked for (e.g. "friendly", "formal").
        """
        user_message = self._build_user_message(
            original_prompt  = original_prompt,
            llm_response     = llm_response,
            placeholders     = vault.placeholders,
            requested_tone   = requested_tone,
        )

        logger.debug("VEIL GUARD → sending evaluation request to Gemini")
        raw = self._call_gemini(user_message)
        logger.debug("VEIL GUARD ← raw response: %s", raw)

        decision = self._parse_decision(raw)

        # ------------------------------------------------------------------
        # Deterministic override: placeholder integrity check
        # This runs independently of the AI model and cannot be bypassed.
        # ------------------------------------------------------------------
        decision = self._enforce_placeholder_integrity(decision, llm_response, vault)

        return decision

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(
        *,
        original_prompt: str,
        llm_response:    str,
        placeholders:    list[str],
        requested_tone:  str,
    ) -> str:
        placeholder_list = ", ".join(placeholders) if placeholders else "(none)"
        return (
            f"## Evaluation request\n\n"
            f"**Requested tone:** {requested_tone}\n\n"
            f"**Redacted prompt sent to the LLM:**\n{original_prompt}\n\n"
            f"**LLM's proposed response:**\n{llm_response}\n\n"
            f"**Expected placeholders that must appear in the response:**\n"
            f"{placeholder_list}\n\n"
            f"Evaluate the response and return your JSON decision."
        )

    def _call_gemini(self, user_message: str) -> str:
        """Send the evaluation request to Gemini with model fallback."""
        last_error = None
        for model_name in self._model_chain:
            try:
                if model_name != self._model_chain[0]:
                    logger.warning("VEIL GUARD: falling back to model %s", model_name)
                    model = self._build_model(model_name)
                else:
                    model = self._model
                response = model.generate_content(user_message)
                return response.text.strip()
            except (ServiceUnavailable, ResourceExhausted) as e:
                logger.warning(
                    "VEIL GUARD: model %s unavailable (%s), trying next...",
                    model_name, e,
                )
                last_error = e
                continue

        raise RuntimeError(
            f"All guard models exhausted. Last error: {last_error}"
        )

    @staticmethod
    def _parse_decision(raw: str) -> GuardDecision:
        """
        Parse the raw JSON string returned by Gemini into a GuardDecision.
        Falls back to REJECT with a diagnostic reason if parsing fails.
        """
        # Strip accidental markdown fences if the model adds them
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
            return GuardDecision(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("VEIL GUARD could not parse response: %s | raw=%s", exc, raw)
            return GuardDecision(
                decision              = DecisionCode.REJECT,
                reason_code           = "GUARD_PARSE_ERROR",
                feedback              = f"VEIL GUARD returned an unparseable response: {exc}",
                confidence            = 0.0,
                sensitive_data_needed = False,
            )

    @staticmethod
    def _enforce_placeholder_integrity(
        decision:     GuardDecision,
        llm_response: str,
        vault:        VaultSnapshot,
    ) -> GuardDecision:
        """
        Deterministically verify that every placeholder registered in the
        vault is still present in the LLM response.

        If any placeholder is missing and the decision is APPROVE, override
        it to REJECT.  This check cannot be bypassed by the AI model.
        """
        if decision.decision != DecisionCode.APPROVE:
            return decision   # Only override APPROVE decisions

        missing = [p for p in vault.placeholders if p not in llm_response]
        if missing:
            logger.warning(
                "VEIL GUARD deterministic check: missing placeholders %s — "
                "overriding APPROVE to REJECT", missing
            )
            return GuardDecision(
                decision              = DecisionCode.REJECT,
                reason_code           = "PLACEHOLDER_MISSING",
                feedback              = (
                    f"The following placeholders were missing from the response "
                    f"and must be preserved: {', '.join(missing)}"
                ),
                confidence            = 1.0,
                sensitive_data_needed = False,
            )

        return decision
