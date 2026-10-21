"""Shared recovery-report arithmetic, authored by Chang Cheng."""

import numpy as np


def unit_nonnegative_rows(matrix):
    """Normalize already validated finite nonnegative rows without raw norms.

    Scaling before the norm avoids squaring very large or very small entries.
    Zero rows stay zero and receive no matching credit. Callers retain their
    own validation and reporting conventions.
    """
    scale = np.max(matrix, axis=1, initial=0)
    scaled = np.divide(matrix, scale[:, None], out=np.zeros_like(matrix),
                       where=scale[:, None] > 0)
    norms = np.linalg.norm(scaled, axis=1)
    return (np.divide(scaled, norms[:, None], out=np.zeros_like(scaled),
                      where=norms[:, None] > 0), norms > 0)
