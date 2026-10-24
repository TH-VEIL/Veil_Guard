"""
veil/controller.py
------------------
VEIL pipeline controller — wires VEIL, the polishing LLM, and VEIL GUARD
together into a single end-to-end request handler.

Current implementation covers:

    Case 1 — APPROVE
        VEIL GUARD is satisfied with the LLM's response.
        Placeholders are restored locally and the polished text is returned.

    (Cases 2–4 are stubbed and will be wired in subsequent iterations.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from veil.guard     import VeilGuard
from veil.models    import DecisionCode, GuardDecision, VaultSnapshot
from veil.polisher  import VeilPolisher
from veil.redactor  import VeilRedactor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result object returned to the caller
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    The complete result of one VEIL pipeline run.

    Attributes
    ----------
    success
        True when a polished response was safely returned to the user.
    polished_text
        The final message with placeholders restored.  None if not approved.
    redacted_input
        The sanitised prompt that was sent to the external LLM.
    raw_llm_response
        The raw response returned by the polishing LLM.
    guard_decision
        The GuardDecision produced by VEIL GUARD.
    attempts
        How many polishing attempts were made (always 1 in Case 1).
    error_message
        Set when success is False.
    """
    success:          bool
    polished_text:    Optional[str]
    redacted_input:   str
    raw_llm_response: Optional[str]
    guard_decision:   Optional[GuardDecision]
    attempts:         int              = 1
    error_message:    Optional[str]    = None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class VeilController:
    """
    Orchestrates the full VEIL pipeline for a single user request.

    Parameters
    ----------
    api_key
        Google AI API key shared by the polisher and guard.
    max_retries
        Maximum number of retry attempts (Case 2).  Currently unused.
    requested_tone
        Default tone to pass to the polishing LLM and VEIL GUARD.
    """

    def __init__(
        self,
        api_key:        Optional[str] = None,
        max_retries:    int           = 2,
        requested_tone: str           = "polite and professional",
    ):
        self._redactor      = VeilRedactor()
        self._polisher      = VeilPolisher(api_key=api_key)
        self._guard         = VeilGuard(api_key=api_key)
        self._max_retries   = max_retries
        self._requested_tone = requested_tone

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, user_text: str) -> PipelineResult:
        """
        Run the complete VEIL pipeline for *user_text*.

        Steps
        -----
        1. VEIL redacts PII → produces sanitised prompt + vault.
        2. Polishing LLM receives sanitised prompt → produces candidate.
        3. VEIL GUARD evaluates the candidate.
        4a. APPROVE (Case 1): restore placeholders locally and return.
        4b. RETRY   (Case 2): stub — will re-polish with guard feedback.
        4c. ASK_USER (Case 3): stub — will surface clarification request.
        4d. REJECT  (Case 4): stub — will discard and return safe error.
        """
        # ── Step 1: Redact ──────────────────────────────────────────────
        logger.info("VEIL: redacting user input")
        redacted_input, vault = self._redactor.redact(user_text)
        logger.info(
            "VEIL: redacted %d entity/entities → %s",
            len(vault.entries), redacted_input,
        )

        # ── Step 2: Polish ──────────────────────────────────────────────
        logger.info("VEIL POLISHER: sending redacted prompt to external LLM")
        llm_response = self._polisher.polish(redacted_input)
        logger.info("VEIL POLISHER: received response")

        # ── Step 3 & 4: Guard evaluates ─────────────────────────────────
        logger.info("VEIL GUARD: evaluating LLM response")
        decision = self._guard.evaluate(
            original_prompt = redacted_input,
            llm_response    = llm_response,
            vault           = vault,
            requested_tone  = self._requested_tone,
        )
        logger.info(
            "VEIL GUARD decision: %s | reason: %s | confidence: %.2f",
            decision.decision, decision.reason_code, decision.confidence,
        )

        # ── Case 1: APPROVE ─────────────────────────────────────────────
        if decision.decision == DecisionCode.APPROVE:
            polished = vault.restore(llm_response)
            logger.info("VEIL: placeholders restored → returning polished text")
            return PipelineResult(
                success          = True,
                polished_text    = polished,
                redacted_input   = redacted_input,
                raw_llm_response = llm_response,
                guard_decision   = decision,
                attempts         = 1,
            )

        # ── Case 2: RETRY (stub) ─────────────────────────────────────────
        if decision.decision == DecisionCode.RETRY:
            logger.warning(
                "VEIL GUARD: RETRY requested — Case 2 not yet implemented. "
                "Feedback: %s", decision.feedback,
            )
            return PipelineResult(
                success          = False,
                polished_text    = None,
                redacted_input   = redacted_input,
                raw_llm_response = llm_response,
                guard_decision   = decision,
                attempts         = 1,
                error_message    = (
                    f"[Case 2 — RETRY pending] Guard feedback: {decision.feedback}"
                ),
            )

        # ── Case 3: ASK_USER (stub) ──────────────────────────────────────
        if decision.decision == DecisionCode.ASK_USER:
            logger.warning(
                "VEIL GUARD: ASK_USER — Case 3 not yet implemented. "
                "Feedback: %s", decision.feedback,
            )
            return PipelineResult(
                success          = False,
                polished_text    = None,
                redacted_input   = redacted_input,
                raw_llm_response = llm_response,
                guard_decision   = decision,
                attempts         = 1,
                error_message    = (
                    f"[Case 3 — ASK_USER pending] Clarification needed: {decision.feedback}"
                ),
            )

        # ── Case 4: REJECT ───────────────────────────────────────────────
        logger.error(
            "VEIL GUARD: REJECT — %s | %s", decision.reason_code, decision.feedback
        )
        return PipelineResult(
            success          = False,
            polished_text    = None,
            redacted_input   = redacted_input,
            raw_llm_response = llm_response,
            guard_decision   = decision,
            attempts         = 1,
            error_message    = (
                f"[Case 4 — REJECT] {decision.reason_code}: {decision.feedback}"
            ),
        )
