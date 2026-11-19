"""
SEMANTIC (conclusion-aware) reasoning parser -- a SEPARATE alternative to the label-based
expl_answer in analyze_reasoning_gap.py (kept unchanged). It extracts the answer the model's
explanation *concludes*, in priority order:

  1. an explicit "None of the above" conclusion            -> abstain
  2. an explicitly STATED category (wide window)            -> that category
     ("BI-RADS classification is likely to be 2", "making it a Level C")
  3. an explicit "cannot determine / not possible" hedge    -> None (unparseable)
  4. otherwise map the CLINICAL DESCRIPTION to a category using the official definitions,
     with clause-level negation ("no malignancy" / "not entirely fatty" are handled).

Official definitions:
  Density  A almost entirely fatty   B scattered fibroglandular   C heterogeneously dense   D extremely dense
  BI-RADS  0 incomplete/needs more imaging   1 negative/normal   2 benign   3 probably benign / short follow-up
           4 suspicious / biopsy   5 highly suggestive of malignancy   6 known biopsy-proven malignancy
"""
import re

_NEGATORS = re.compile(r"\b(no|not|without|non|free of|negative for|absence of|lacks?|rather than)\b")


def _negated(text, start):
    """True if a negator appears in the same clause before position `start`
    (clause = text since the last comma / period / semicolon / 'but' / 'however')."""
    pre = text[:start]
    cut = max(pre.rfind(","), pre.rfind("."), pre.rfind(";"),
              pre.rfind(" but "), pre.rfind(" however"), pre.rfind(" therefore"))
    return _NEGATORS.search(pre[cut + 1:]) is not None


def _present(text, pat):
    """A non-negated occurrence of `pat` exists."""
    for m in re.finditer(pat, text):
        if not _negated(text, m.start()):
            return True
    return False


def _nota(text):
    """True only for a genuine 'none of the above' CONCLUSION -- not when the model REJECTS it
    ('none of the above is not correct') or negates it ('not none of the above')."""
    judge = (r"\bincorrect\b|\bwrong\b|\bunlikely\b|\binvalid\b|least likely|not likely|"
             r"not (?:the )?(?:correct|right|best|valid|answer)|not applicable|would not be|"
             r"is not (?:a )?valid|one of the (?:other )?options is (?:correct|valid)")
    for m in re.finditer(r"none of the above|not fit(?:s|ting)?(?: in| into)? any|"
                         r"does not (?:fit|belong)|fits? none", text):
        after = text[m.end():m.end() + 45]
        before = text[max(0, m.start() - 55):m.start()]
        window = before + text[m.start():m.end() + 45]
        # reject on: a judgement right AFTER; a negator immediately before; or an option-enumeration
        # rejection ("Option H ... unlikely/incorrect") anywhere in the local window.
        if re.search(judge, after) \
                or re.search(r"\b(not|rather than|isn'?t|besides|other than|except)\b\s*$", before) \
                or ("option h" in window and re.search(judge, window)):
            continue
        return True
    return False


def _cant(text):
    return re.search(r"cannot be (?:determined|classified|assessed|evaluated)|"
                     r"not possible to (?:determine|classify|assess)|unable to (?:determine|classify)|"
                     r"impossible to (?:determine|classify)|not (?:enough|sufficient)[\w\s]{0,20}determine", text) is not None


def semantic_birads(expl):
    t = " " + (expl or "").lower() + " "
    if not t.strip():
        return None
    if _nota(t):
        return "None of the above"
    # explicitly stated BI-RADS number (wide window; digit must be a standalone 0-6)
    for pat in (r"bi[\s-]*rads?\b[^0-9\n]{0,45}?([0-6])\b",
                r"(?:category|classification|rating|assessment)\b[^0-9\n]{0,25}?([0-6])\b"):
        mm = list(re.finditer(pat, t))
        if mm:
            return f"Bi-Rads {mm[-1].group(1)}"
    if _cant(t):
        return None
    # description-based, most-severe first, negation-aware
    if _present(t, r"biopsy[- ]proven") or _present(t, r"known[\w\s]{0,15}malignan"):
        return "Bi-Rads 6"
    if _present(t, r"highly (?:suggestive|suspicious)") or _present(t, r"strongly suggestive"):
        return "Bi-Rads 5"
    if _present(t, r"suspicious") or _present(t, r"malignan") or _present(t, r"\bbiopsy\b") \
            or _present(t, r"concerning") or _present(t, r"higher bi[\s-]*rads") \
            or _present(t, r"warrant[\w\s]{0,20}higher") \
            or (_present(t, r"\babnormal") and (_present(t, r"\bmass\b") or _present(t, r"\blesion"))):
        return "Bi-Rads 4"
    if _present(t, r"probably benign") or _present(t, r"short[- ]?interval") \
            or _present(t, r"short[- ]?term follow") or _present(t, r"low[- ]?risk"):
        return "Bi-Rads 3"
    if _present(t, r"\bbenign\b"):
        return "Bi-Rads 2"
    if _present(t, r"additional (?:imaging|view|mammograph)") or _present(t, r"\bincomplete\b") \
            or _present(t, r"further (?:imaging|evaluation|assessment|investigation|work[- ]?up)"):
        return "Bi-Rads 0"
    # negative / normal study -> BI-RADS 1
    if re.search(r"\b(negative|normal|unremarkable|benign[- ]appearing)\b", t):
        return "Bi-Rads 1"
    if re.search(r"\b(no|without|free of|negative for|absence of)\b[\w\s]{0,25}?"
                 r"(abnormal|\bmass|lesion|calcification|suspicious|malignan|\bfinding|significant)", t):
        return "Bi-Rads 1"
    return None


def semantic_acr(expl):
    t = " " + (expl or "").lower() + " "
    if not t.strip():
        return None
    if _nota(t):
        return "None of the above"
    # explicitly stated level (standalone letter a-d or digit 1-4 near the keyword)
    m = list(re.finditer(r"(?:level|category|density|acr|grade)\b.{0,20}?\b([a-d1-4])\b", t))
    if m:
        v = m[-1].group(1); mp = {"1": "A", "2": "B", "3": "C", "4": "D",
                                  "a": "A", "b": "B", "c": "C", "d": "D"}
        return f"Level {mp[v]}"
    if _cant(t):
        return None
    # description-based, negation-aware
    if _present(t, r"extremely dense") or _present(t, r"highly dense") or _present(t, r"very dense") \
            or _present(t, r"marked(?:ly)? dense") or _present(t, r"little to no (?:fat|adipose)"):
        return "Level D"
    if _present(t, r"heterogeneous") or _present(t, r"moderately dense") \
            or (_present(t, r"\bdense\b") and not _present(t, r"scattered")):
        return "Level C"
    if _present(t, r"scattered"):
        return "Level B"
    if _present(t, r"entirely fat") or _present(t, r"entirely adipose") or _present(t, r"predominant(?:ly)? fat") \
            or _present(t, r"mostly fat") or _present(t, r"\bfatty\b") or _present(t, r"\badipose\b"):
        return "Level A"
    return None


def semantic_answer(expl, task, options):
    """Definition-based category for the explanation, restricted to this item's options (or None)."""
    if task in ("birads", "birads012"):
        r = semantic_birads(expl)
    elif task == "acr":
        r = semantic_acr(expl)
    else:
        r = None
    return r if (r in (options or [])) else None
