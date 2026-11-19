"""
Reasoning-answer matcher: recover the category an explanation supports, $\\hat{y}_e$.

Combines two matchers (validated precedence rule):
  * expl_answer   -- surface LABEL matcher (explicit "BI-RADS 3" / "Level C" mentions)
  * semantic_answer -- definition-based SEMANTIC matcher (maps described findings to a category,
                       and detects a "none of the above" conclusion)

reason_pred() applies the combined rule: use the semantic matcher's output when it concludes
"None of the above", otherwise prefer the label matcher, falling back to semantic.
"""
from .analyze_reasoning_gap import expl_answer          # label matcher
from .semantic_parse import semantic_answer             # semantic matcher

NOTA = "None of the above"

__all__ = ["expl_answer", "semantic_answer", "reason_pred", "NOTA"]


def reason_pred(expl, task, options):
    """Combined reasoning-answer for a single explanation, or None if neither matcher fires."""
    o = expl_answer(expl, task, options)
    s = semantic_answer(expl, task, options)
    return s if s == NOTA else (o if o is not None else s)
