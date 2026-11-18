"""Read-only diagnostics for mode-specific screening search parameters."""

from copy import deepcopy
import math
from numbers import Integral, Real
import warnings

from sklearn.base import clone

from .classifier import NMFClassifier
from .screening import BinaryScreening


def _scalar(value):
    """Keep diagnostics JSON-safe even when a candidate value is invalid."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def screening_grid_diagnostics(estimator, candidates):
    """Describe known screens and warn once about explicitly inactive grid keys.

    Existing constructor parameters do not trigger warnings. Each effective
    candidate is inspected after applying replacements and nested parameters
    to a clone. Supplied replacement estimators are cloned too, so nested
    ``set_params`` cannot modify objects in the caller's grid. Invalid candidate
    configurations remain the search's responsibility and get empty diagnostics.
    No fitting, mode changes, candidate rejection or deduplication occurs here.
    """
    rows = []
    inactive = {}
    for index, parameters in enumerate(candidates):
        row = {"index": index, "screens": []}
        rows.append(row)
        try:
            replacements = {key: clone(value, safe=False)
                            for key, value in parameters.items()}
            configured = clone(estimator).set_params(**replacements)
            nested = configured.get_params(deep=True)
        except Exception:
            # A failed candidate must retain its ordinary calibration failure,
            # rather than fail early because optional diagnostics inspected it.
            continue
        screens = [("", configured)]
        screens.extend((path, value) for path, value in sorted(nested.items())
                       if isinstance(value, (BinaryScreening, NMFClassifier)))
        for prefix, screen in screens:
            if not isinstance(screen, (BinaryScreening, NMFClassifier)):
                continue
            mode = screen.screening
            allowed = ("absolute", "percentile", "none") if isinstance(screen, NMFClassifier) else ("absolute", "percentile")
            if not isinstance(mode, str) or mode not in allowed:
                continue
            key_prefix = prefix + "__" if prefix else ""
            local_active = {"absolute": "score_threshold", "percentile": "percentile"}.get(mode)
            active = key_prefix + local_active if local_active else None
            ignored = [key_prefix + name for name in ("percentile", "score_threshold")
                       if name != local_active and key_prefix + name in parameters]
            row["screens"].append({
                "path": prefix or "(root)", "mode": mode,
                "active_parameter": active,
                "active_value": _scalar(getattr(screen, local_active)) if local_active else None,
                "inactive_grid_parameters": ignored,
            })
            for parameter in ignored:
                inactive.setdefault((parameter, mode), []).append(index)
    grouped = [{"parameter": parameter, "mode": mode, "candidate_indices": indices}
               for (parameter, mode), indices in sorted(inactive.items())]
    if grouped:
        details = "; ".join(
            f"{entry['parameter']} (screening={entry['mode']!r}, "
            f"{len(entry['candidate_indices'])} candidate(s))" for entry in grouped)
        warnings.warn(
            "Inactive screening grid parameters do not affect feature selection: "
            + details + ". Set screening explicitly and use mode-specific grids.",
            UserWarning, stacklevel=2,
        )
    return {"candidates": rows, "inactive_parameters": grouped}


def mark_inactive_refinement(refinement, diagnostics, candidate_index):
    """Annotate inactive selected-candidate keys without proposing refinement.

    The caller supplies the selected candidate's index in ``diagnostics``.
    Missing diagnostics or an unavailable selection preserve the old result.
    """
    result = deepcopy(refinement)
    if diagnostics is None or candidate_index is None:
        return result
    rows = diagnostics.get("candidates", ())
    selected = next((row for row in rows if row.get("index") == candidate_index), None)
    if selected is None:
        return result
    parameters = result.get("parameters", {})
    for screen in selected.get("screens", ()):
        for name in screen.get("inactive_grid_parameters", ()):
            if name in parameters:
                parameters[name] = {
                    **parameters[name],
                    "status": "inactive_for_selected_screening_mode",
                    "screening_mode": screen["mode"],
                }
    return result
