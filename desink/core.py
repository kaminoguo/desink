"""De-sinking: removing the leading direction from transformer hidden states.

The whole method is one line,

    C = H - (H @ s)[:, None] * s

with ``s`` the top right singular vector of the centered ``H``.  Everything else
in this file computes the spectral summaries the projection acts on, and checks
the identity that relates the two.
"""

import numpy as np
import torch

__all__ = [
    "compute_sink_direction",
    "desink",
    "spectral_metrics",
    "sink_strength",
    "identity_check",
]


def _as_torch(H):
    if isinstance(H, np.ndarray):
        return torch.from_numpy(H).float(), True
    return H, False


def compute_sink_direction(H):
    """Top right singular vector of the centered ``H``.

    Args:
        H: hidden states, shape ``(n_tokens, d_model)``.

    Returns:
        Unit vector of shape ``(d_model,)``, same type as ``H``.
    """
    Ht, was_numpy = _as_torch(H)
    _, _, Vt = torch.linalg.svd(Ht - Ht.mean(dim=0, keepdim=True), full_matrices=False)
    s = Vt[0]
    return s.numpy() if was_numpy else s


def desink(H, s=None):
    """Project the leading direction out of ``H``.

    Args:
        H: hidden states, shape ``(n_tokens, d_model)``.
        s: precomputed direction. Defaults to the top right singular vector of
           the centered ``H``.

    Returns:
        The residual ``C``, same shape and type as ``H``.
    """
    Ht, was_numpy = _as_torch(H)
    if s is None:
        s = compute_sink_direction(Ht)
    elif isinstance(s, np.ndarray):
        s = torch.from_numpy(s).float()
    C = Ht - (Ht @ s).unsqueeze(-1) * s.unsqueeze(0)
    return C.numpy() if was_numpy else C


def _exp_entropy(w):
    """exp of the Shannon entropy of ``w`` normalized to a distribution."""
    p = w / w.sum()
    p = p[p > 1e-12]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def spectral_metrics(H, n_aniso_sample=4096, generator=None):
    """The five summaries the paper studies.

    ``effective_rank`` applies the exponential entropy to normalized
    eigenvalues, the convention of the training-dynamics literature.
    ``rankme`` applies it to normalized singular values, the convention of
    Garrido et al.  Section 3 of the paper shows that this choice decides how
    much a single direction can distort the result.

    ``anisotropy`` is the mean pairwise cosine of the *uncentered* rows, the
    definition of Ethayarajh (2019).  ``anisotropy_centered`` is the same
    statistic after centering; it is much smaller, and it is reported only
    because some implementations use it.

    Args:
        H: hidden states, shape ``(n_tokens, d_model)``.
        n_aniso_sample: rows subsampled for the anisotropy estimate.
        generator: optional ``torch.Generator`` for that subsample.

    Returns:
        dict with keys ``e1``, ``effective_rank``, ``rankme``, ``anisotropy``,
        ``anisotropy_centered``, ``spectral_gap``.
    """
    Ht, _ = _as_torch(H)
    Hc = Ht - Ht.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(Hc).double()
    ev = sv ** 2

    def mean_pairwise_cosine(X):
        n = X.shape[0]
        if n > n_aniso_sample:
            idx = torch.randperm(n, generator=generator)[:n_aniso_sample]
            X = X[idx]
        U = X / (X.norm(dim=1, keepdim=True) + 1e-10)
        G = (U @ U.T).double()
        m = U.shape[0]
        return float((G.sum() - m) / (m * (m - 1)))

    return {
        "e1": float(ev[0] / ev.sum()),
        "effective_rank": _exp_entropy(ev),
        "rankme": _exp_entropy(sv),
        "anisotropy": mean_pairwise_cosine(Ht),
        "anisotropy_centered": mean_pairwise_cosine(Hc),
        "spectral_gap": float(sv[0] / sv[1]) if len(sv) > 1 else float("inf"),
    }


def sink_strength(H, seq_len):
    """Mean norm at sequence position 0 over the mean norm everywhere else.

    Args:
        H: hidden states of ``n_tokens // seq_len`` equal-length sequences
           stacked in order, shape ``(n_tokens, d_model)``.
        seq_len: tokens per sequence.

    Returns:
        The ratio the paper calls ``alpha``.
    """
    Ht, _ = _as_torch(H)
    norms = Ht.norm(dim=1)
    at0 = torch.zeros(len(norms), dtype=torch.bool)
    at0[::seq_len] = True
    return float(norms[at0].mean() / norms[~at0].mean())


def identity_check(H):
    """Proposition 2 of the paper, evaluated on ``H``.

    ``ER(H) = exp(h_b(E1)) * ER(C) ** (1 - E1)`` holds for every matrix, so the
    relative error returned here is the error of the eigendecomposition, around
    ``1e-15`` in double precision.

    Returns:
        dict with ``measured``, ``predicted``, ``relative_error``.
    """
    raw = spectral_metrics(H)
    res = spectral_metrics(desink(H))
    p = raw["e1"]
    hb = -(p * np.log(p) + (1 - p) * np.log1p(-p))
    predicted = float(np.exp(hb) * res["effective_rank"] ** (1 - p))
    return {
        "measured": raw["effective_rank"],
        "predicted": predicted,
        "relative_error": abs(predicted - raw["effective_rank"]) / raw["effective_rank"],
    }
