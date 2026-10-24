"""
veil/models.py
--------------
Shared Pydantic models used across VEIL components.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# VEIL GUARD decision model
# ---------------------------------------------------------------------------

class DecisionCode(str, Enum):
    """
    All possible top-level decisions VEIL GUARD can return.

    APPROVE  — Case 1: The LLM response is satisfactory. Safe to restore
               placeholders and return to the user.
    RETRY    — Case 2 (future): Response is unsatisfactory but fixable.
               The controller will re-send with improved instructions.
    ASK_USER — Case 3 (future): Ambiguity cannot be resolved without
               clarification from the user.
    REJECT   — Case 4 (future): Response contains a privacy/policy
               violation and must be discarded.
    """
    APPROVE   = "APPROVE"
    RETRY     = "RETRY"
    ASK_USER  = "ASK_USER"
    REJECT    = "REJECT"


class GuardDecision(BaseModel):
    """
    Structured output returned by the VEIL GUARD agent.

    Fields
    ------
    decision
        One of the DecisionCode values above.
    reason_code
        A short, machine-readable label for the specific reason.
        Examples: "SATISFACTORY", "TONE_MISMATCH", "MEANING_CHANGED",
                  "PLACEHOLDER_VIOLATED", "AMBIGUOUS_INTENT".
    feedback
        Human-readable explanation. If decision == RETRY, this string is
        appended to the next polishing prompt as an improvement instruction.
    confidence
        Float in [0, 1]. How confident the guard is in its decision.
    sensitive_data_needed
        True only when the guard believes the task literally cannot be
        completed without a specific sensitive value. This does NOT
        authorise disclosure — a separate policy layer must approve.
    """
    decision:              DecisionCode = Field(..., description="Top-level verdict")
    reason_code:           str           = Field(..., description="Short machine-readable reason label")
    feedback:              str           = Field(..., description="Human-readable explanation / retry instruction")
    confidence:            float         = Field(ge=0.0, le=1.0, description="Guard confidence [0, 1]")
    sensitive_data_needed: bool          = Field(False, description="True only if PII is genuinely required")


# ---------------------------------------------------------------------------
# VEIL redaction / vault models
# ---------------------------------------------------------------------------

class VaultEntry(BaseModel):
    """A single placeholder ↔ original-value mapping."""
    placeholder:    str = Field(..., description="Token used in the sanitised prompt, e.g. <PERSON_1>")
    original_value: str = Field(..., description="The real sensitive value that was replaced")
    entity_type:    str = Field(..., description="PII category, e.g. PERSON, PHONE_NUMBER, EMAIL_ADDRESS")


class VaultSnapshot(BaseModel):
    """
    The complete set of placeholder mappings for one user request.
    Kept in memory only for the lifetime of a single request.
    """
    entries: list[VaultEntry] = Field(default_factory=list)

    @property
    def placeholders(self) -> list[str]:
        return [e.placeholder for e in self.entries]

    def restore(self, text: str) -> str:
        """Replace every placeholder in *text* with its original value."""
        for entry in self.entries:
            text = text.replace(entry.placeholder, entry.original_value)
        return text
