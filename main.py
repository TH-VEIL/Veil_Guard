"""
main.py
-------
Entry point for the VEIL demo.

Run:
    export GEMINI_API_KEY=your_key_here
    python main.py

You can also pass --text "your message here" on the command line.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap

from veil.controller import VeilController
from veil.models import DecisionCode

# ---------------------------------------------------------------------------
# Logging setup — set to DEBUG for full pipeline trace
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Colour helpers (optional, degrades gracefully if terminal doesn't support)
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner(title: str, colour: str = CYAN) -> str:
    line = "─" * 60
    return f"\n{colour}{BOLD}{line}\n  {title}\n{line}{RESET}"


def print_result(result) -> None:
    print(banner("VEIL PIPELINE RESULT"))

    print(f"\n{BOLD}📥  Redacted input sent to LLM:{RESET}")
    print(f"    {result.redacted_input}")

    print(f"\n{BOLD}🤖  Raw LLM response:{RESET}")
    print(f"    {result.raw_llm_response}")

    decision = result.guard_decision
    if decision:
        colour = GREEN if decision.decision == DecisionCode.APPROVE else YELLOW
        print(f"\n{BOLD}🛡️  VEIL GUARD decision:{RESET}")
        print(f"    {colour}{decision.decision.value}{RESET}"
              f"  (confidence: {decision.confidence:.0%})")
        print(f"    Reason : {decision.reason_code}")
        print(f"    Feedback: {decision.feedback}")
        print(f"    Sensitive data needed: {decision.sensitive_data_needed}")

    if result.success:
        print(f"\n{GREEN}{BOLD}✅  Case 1 — APPROVED{RESET}")
        print(f"\n{BOLD}📤  Final polished message:{RESET}")
        print(textwrap.indent(result.polished_text, "    "))
    else:
        print(f"\n{RED}{BOLD}❌  Not approved{RESET}")
        print(f"    {result.error_message}")

    print()


# ---------------------------------------------------------------------------
# Demo messages
# ---------------------------------------------------------------------------
DEMO_MESSAGES = [
    (
        "casual → professional",
        "hey priya, i cant come today call me on 9876543210 to reschedule. its urgent"
    ),
    (
        "formal request with email",
        "hi john, please send the report to sadhana@example.com before 5pm. thx"
    ),
    (
        "appointment reminder",
        "reminder for dr. mehta: your patient raj sharma has an appt on 15 nov at 3pm "
        "at 42 marine drive, mumbai"
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="VEIL — privacy-preserving text polisher")
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text to polish. Omit to run the built-in demo messages.",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="polite and professional",
        help='Desired tone, e.g. "friendly" or "formal". Default: polite and professional.',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging for full pipeline trace.",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY environment variable is not set.{RESET}")
        print("  Export it with:  export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    controller = VeilController(api_key=api_key, requested_tone=args.tone)

    if args.text:
        messages = [("Custom input", args.text)]
    else:
        messages = DEMO_MESSAGES

    for label, text in messages:
        print(banner(f"Demo: {label}"))
        print(f"\n{BOLD}📝  Original user message:{RESET}")
        print(f"    {text}")

        result = controller.process(text)
        print_result(result)

        if len(messages) > 1:
            input(f"{CYAN}Press Enter to continue to the next demo...{RESET}\n")


if __name__ == "__main__":
    main()
