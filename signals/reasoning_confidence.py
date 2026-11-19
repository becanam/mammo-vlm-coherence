"""
Reasoning-derived confidence (0-100) from an explanation's LANGUAGE -- how certain the model *sounds*,
independent of its numeric scores. Rule-based lexical certainty (documented, deterministic):

  STRONG_HEDGE  the model doubts it can answer / recognizes no valid evidence  -> confidence := 15
  SOFT_HEDGE    epistemic hedging ("may", "appears", "likely")                 -> each -12 (cap 3)
  BOOSTER       certainty markers ("clearly", "consistent with", "shows")      -> each +10 (cap 3)
  neutral baseline = 65

  reasoning_confidence = clamp(65 + 10*min(#boost,3) - 12*min(#soft,3), 0, 100),  or 15 if any strong hedge.
  reasoning_uncertainty = 100 - reasoning_confidence.
  Returns None for an empty explanation (undefined -- excluded from means).

This measures RHETORICAL certainty (surface wording), not calibrated correctness: a booster attached to
a wrong claim still reads as confident, and "likely" is inherently ambiguous.
"""
import re

STRONG_HEDGE = [
    "cannot determine", "cannot be determined", "can't determine", "unable to determine", "unable to",
    "not possible to determine", "impossible to determine", "difficult to determine",
    "hard to determine", "cannot be assessed", "cannot be evaluated", "cannot classify",
    "not enough information", "insufficient", "indeterminate", "not clear enough",
    "without more", "without additional", "further evaluation is needed",
    # no-valid-evidence recognition (should raise uncertainty on blank/unrelated/no-image)
    "no image", "not a mammogram", "no mammogram", "not a valid", "blank image",
    "does not appear to be a mammogram", "not a medical image", "there is no image",
]
SOFT_HEDGE = [
    "may", "might", "could", "possibly", "perhaps", "appears", "appear to", "seems", "suggests",
    "suggestive", "likely", "probably", "potentially", "unclear", "uncertain", "presumably",
    "i think", "somewhat", "hard to tell", "not sure",
]
BOOSTER = [
    "clearly", "definitely", "certainly", "obviously", "evidently", "consistent with",
    "characteristic of", "diagnostic of", "confirms", "definitive", "classic", "typical of",
    "unambiguous", "demonstrates", "definitively",
]


def _count(text, terms):
    return sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) for term in terms)


def reasoning_confidence(expl):
    """0-100 rhetorical confidence of the explanation, or None if the explanation is empty."""
    t = (expl or "").lower().strip()
    if not t:
        return None
    if _count(t, STRONG_HEDGE) > 0:
        return 15.0
    conf = 65 + 10 * min(_count(t, BOOSTER), 3) - 12 * min(_count(t, SOFT_HEDGE), 3)
    return float(max(0.0, min(100.0, conf)))


def reasoning_uncertainty(expl):
    c = reasoning_confidence(expl)
    return None if c is None else 100.0 - c
