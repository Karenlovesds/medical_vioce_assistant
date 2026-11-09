"""Deterministic safety layer — runs BEFORE intent routing and the LLM prompt.

Interview line: "Safety is enforced in code, not just in the prompt, so it
holds even when the model is uncertain or the caller is adversarial."

Each function is one check you can explain in one sentence.
"""

import re

EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bcan'?t breathe\b",
    r"\btrouble breathing\b",
    r"\bsevere bleeding\b",
    r"\b(?:having|think i'?m having) a stroke\b",
    r"\boverdose",
]

SELF_HARM_PATTERNS = [
    r"\bsuicid",
    r"\bkill(?:ing)? myself\b",
    r"\bhurt(?:ing)? myself\b",
    r"\bend my life\b",
]

# Requests the agent must NEVER fulfill: diagnosis or treatment changes.
# Education is allowed; clinical guidance is not.
DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b.{0,30}\b(?:diabetes|cancer|condition|disease|infection|flu|covid)\b",
    r"\bwhat(?:'s| is) wrong with me\b",
    r"\bdiagnos",
    r"\bshould i (?:stop|double|change|increase|decrease|skip)\b.{0,40}\b(?:dose|dosage|medication|pill|prescription)\b",
    r"\bwhat dose\b.{0,40}\bshould i take\b",
    r"\bhow (?:much|many)\b.{0,40}\bshould i take\b",
]

# Adversarial callers trying to override the agent's instructions.
INJECTION_PATTERNS = [
    r"\bignore (?:your|all|previous|these) instructions\b",
    r"\bpretend (?:you are|you're) a (?:doctor|clinician|nurse)\b",
    r"\bact as (?:my )?(?:doctor|clinician)\b",
    r"\breveal your (?:system prompt|instructions)\b",
]

# Caller is not following; count it, clarify, then hand off.
CONFUSED_PATTERNS = [
    r"\bconfused\b",
    r"\bi'?m lost\b",
    r"\bi am lost\b",
    r"\bi (?:still )?(?:don'?t|do not|dont) (?:understand|get it)\b",
    r"\bcan you repeat\b",
    r"\bsay that again\b",
    r"\bwhat do you mean\b",
    r"\bthat doesn'?t make sense\b",
    r"\bnot sure what you mean\b",
    r"\bwhat\?",
]

HANDOFF_PATTERNS = [
    r"\b(?:i want|i'?d like|i would like|can i|could i|may i|let me|please let me)(?: to)? (?:speak|talk|chat) (?:to|with) (?:a |an |the )?(?:human|person|representative|agent|someone|nurse|doctor|clinician)\b",
    r"(?<!do )\bi need to (?:speak|talk|chat) (?:to|with) (?:a |an |the )?(?:human|person|representative|agent|someone|nurse|doctor|clinician)\b",
    r"\b(?:transfer|connect|put) me\b",
    r"\btransfer(?: me| us)?(?: over)? (?:to|with) (?:a |an |the )?(?:human|person|representative|operator|someone|staff)\b",
    r"\b(?:speak|talk|chat) (?:to|with) (?:a |an |the )?(?:human|person|representative|operator|someone|staff)\b",
    r"\b(?:real|live|actual) (?:human|person|agent)\b",
    r"\bi need (?:a |an )?(?:human|person|representative|agent)\b",
    r"\b(?:is there|are there) (?:a |an |any )?(?:human|person|representative|agent|someone)\b",
    r"\b(?:get|give) me (?:a |an )?(?:human|person|representative|agent)\b",
]

PII_PATTERNS = {
    # Social Security number: 123-45-6789
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    # 16-digit card number, with optional spaces or dashes between groups
    "card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}

CONTEXT_CHARS = 30
CLAUSE_SPLIT = re.compile(r"[,.;!?]|\b(?:but|and|however|though)\b")
BEFORE_NEGATION = re.compile(
    r"\b(?:don'?t|do not|doesn'?t|didn'?t)\s+(?:have|want|plan|intend)\b"
    r"|\b(?:no|without)\s*$"
    r"|\bnever\s+(?:had|have)\b"
    r"|\b(?:denies|denied|ruled out)\b"
)
MATCH_NEGATION = re.compile(r"\bnot\s+having\b")
EDUCATION_CUES = re.compile(r"\b(?:what causes|what is|what are|why does)\b")


def _matches(text: str, patterns: list[str]) -> bool:
    # Lowercase once so every pattern can stay case-insensitive.
    t = text.lower()
    # True if any pattern matches anywhere in the text.
    return any(re.search(p, t) for p in patterns)


def _is_ambiguous(before: str, matched: str) -> bool:
    # Only the phrase's own clause can negate it or make it educational.
    clause = CLAUSE_SPLIT.split(before)[-1]
    return bool(
        BEFORE_NEGATION.search(clause)
        or MATCH_NEGATION.search(clause + matched)
        or EDUCATION_CUES.search(clause)
    )


def _detect(text: str, patterns: list[str]) -> bool:
    # Check each match separately so a mixed sentence still escalates.
    t = text.lower()
    for pattern in patterns:
        for match in re.finditer(pattern, t):
            before = t[max(0, match.start() - CONTEXT_CHARS):match.start()]
            if not _is_ambiguous(before, match.group(0)):
                return True
    return False


def detect_emergency(text: str) -> bool:
    """Physical emergency -> short-circuit everything, advise 911 now."""
    # Skip negated/educational mentions, but keep any other urgent phrase.
    return _detect(text, EMERGENCY_PATTERNS)


def detect_self_harm(text: str) -> bool:
    """Self-harm -> crisis resources + immediate human offer, separate path."""
    # "I don't want to kill myself" is not a crisis; "I want to" still is.
    return _detect(text, SELF_HARM_PATTERNS)


def detect_diagnosis_request(text: str) -> bool:
    """Diagnosis or med-change request -> deterministic refusal + redirect."""
    # Covers "do I have...", "diagnose me", and dosing/med-change requests.
    return _matches(text, DIAGNOSIS_PATTERNS)


def detect_injection(text: str) -> bool:
    """Prompt-injection attempt -> refuse to drop guardrails."""
    # Catches attempts to override instructions or role-play as a clinician.
    return _matches(text, INJECTION_PATTERNS)


def detect_handoff(text: str) -> bool:
    """Caller asks for a human -> be honest that this MVP cannot transfer."""
    # _detect (not _matches) so "I don't want to speak to a person" stays benign.
    return _detect(text, HANDOFF_PATTERNS)


def detect_confusion(text: str) -> bool:
    """Caller says they do not understand -> increment the unknown streak."""
    # Feeds the streak counter; two hits in a row triggers a handoff.
    return _matches(text, CONFUSED_PATTERNS)


def should_escalate_for_confusion(state: dict, max_unknowns: int = 2) -> bool:
    """Two unknowns in a row -> human. The agent never loops forever guessing."""
    # Missing key means zero; the default threshold is two confused turns.
    return state.get("unknown_streak", 0) >= max_unknowns


def redact_pii(text: str) -> str:
    """Mask PII before it lands in logs or handoff summaries."""
    # PII_PATTERNS maps a label ("ssn", "card") to the regex that finds it.
    for kind, pattern in PII_PATTERNS.items():
        # re.sub replaces every match, not just the first.
        # Example: "123-45-6789" -> "[REDACTED-SSN]".
        replacement = f"[REDACTED-{kind.upper()}]"
        text = re.sub(pattern, replacement, text)
    # Return the cleaned copy; the caller's original string is unchanged.
    return text


def check(text: str, state: dict | None = None) -> str | None:
    """Run every safety check in priority order; return the first action.

    Pass a shared ``state`` dict across turns to track confusion. A single
    confused turn returns None (let the model clarify); two in a row returns
    "confusion" so the caller is offered a person.

    "pii" means private numbers were found: mask them before logs or a handoff
    summary, then let the call continue normally.
    """
    # Order matters: life safety first, then refusals, then injection.
    # First match wins; None means let the model handle the turn.

    # Mask SSNs and card numbers once. Keep the cleaned copy in state so
    # logs and any handoff summary can use it instead of the raw text.
    redacted = redact_pii(text)
    if state is not None:
        state["redacted_text"] = redacted

    # Update the confusion streak before the early returns, so it keeps
    # counting even on turns that trigger a different action.
    if state is not None:
        if detect_confusion(text):
            state["unknown_streak"] = state.get("unknown_streak", 0) + 1
        else:
            state["unknown_streak"] = 0

    if detect_self_harm(text):
        return "crisis"  # caller may be in immediate danger
    if detect_emergency(text):
        return "emergency"  # tell the caller to call 911
    if detect_handoff(text):
        return "handoff"
    if detect_diagnosis_request(text):
        return "refuse"  # no diagnosis or medication changes
    if detect_injection(text):
        return "injection"  # keep the guardrails in place
    if state is not None and should_escalate_for_confusion(state):
        return "confusion"  # repeated confusion -> offer a person
    if redacted != text:
        return "pii"  # private numbers were masked; the call continues
    return None
