"""
Map a per-option confidence vector to the model's answer f(A):

    f(A) = argmax_j score_j    if  max_j score_j >= tau     (some option is confident enough)
         = abstain (None of the above)   otherwise          (every score below the threshold)

The choice set includes an explicit "None of the above" (ABSTAIN) option, so the no-match
outcome is realised as predicting ABSTAIN. Two paths both resolve to ABSTAIN: the model scores the
abstain option highest, or every option is below tau (e.g. an all-zero vector -> max 0 < tau).

This avoids a subtle failure: a blind arg-max over an all-zero vector silently returns the FIRST
option, a pure position artifact. Here an all-zero / sub-threshold vector is treated as abstention,
not a confident pick of option A. `tau` is the minimum-confidence threshold; the default tau=1.0
makes an all-zero vector abstain while any positive score is eligible for arg-max.
"""

ABSTAIN = "None of the above"


def matchmaker_pred(scores, options, tau=1.0):
    """Map a per-option score vector to a predicted option text, via the faithful answer f(A).

    scores  : list aligned to `options`; entries may be None (unparsed for that option).
    returns : the predicted option text; ABSTAIN for the ∅ / no-match outcome;
              None only if NO score was parsed at all (caller may then fall back to text parsing).
    """
    idxs = [i for i, s in enumerate(scores) if s is not None]
    if not idxs:
        return None                                    # nothing parsed -> let caller decide
    mx = max(scores[i] for i in idxs)
    if mx < tau:                                        # all below threshold (all-zero incl.) -> ∅
        return ABSTAIN
    idx = max(idxs, key=lambda i: (scores[i], -i))      # arg-max, ties -> lowest index
    return options[idx]                                 # may itself be ABSTAIN if scored highest


def is_abstain(pred_text):
    return pred_text == ABSTAIN


def is_tie(scores, tau=1.0):
    """True if the top confidence score is a POSITIVE tie -- >=2 options share the maximum and that
    maximum is >= tau. All-zero / sub-threshold vectors are NOT ties (they abstain, handled by f(A)).
    Such items have no well-defined arg-max answer (the pipeline breaks the tie by position), so they
    should be discarded from analysis rather than counted as a prediction."""
    vals = [s for s in (scores or []) if s is not None]
    if not vals:
        return False
    mx = max(vals)
    if mx < tau:
        return False
    return sum(1 for s in vals if s == mx) >= 2
