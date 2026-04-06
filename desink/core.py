"""De-sinking: removing the attention sink artifact from transformer hidden states."""

import torch
import numpy as np


def compute_sink_direction(H):
    """Compute the sink direction as the first right singular vector of centered H.

    Args:
        H: Hidden states, shape (n, d). Torch tensor or numpy array.

    Returns:
        s: Unit sink direction, shape (d,).
    """
    is_numpy = isinstance(H, np.ndarray)
    if is_numpy:
        H = torch.from_numpy(H).float()

    H_bar = H - H.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(H_bar, full_matrices=False)
    s = Vt[0]

    return s.numpy() if is_numpy else s


def desink(H, s=None):
    """Remove the sink direction from hidden states.

    Args:
        H: Hidden states, shape (n, d). Torch tensor or numpy array.
        s: Optional precomputed sink direction, shape (d,).
           If None, computed via SVD of centered H.

    Returns:
        H_ds: De-sinked hidden states, same shape and type as H.
    """
    is_numpy = isinstance(H, np.ndarray)
    if is_numpy:
        H = torch.from_numpy(H).float()

    if s is None:
        s = compute_sink_direction(H)
    elif isinstance(s, np.ndarray):
        s = torch.from_numpy(s).float()

    proj = H @ s
    H_ds = H - proj.unsqueeze(-1) * s.unsqueeze(0)

    return H_ds.numpy() if is_numpy else H_ds


def spectral_metrics(H):
    """Compute standard spectral metrics for representation analysis.

    Args:
        H: Hidden states, shape (n, d).

    Returns:
        dict with keys: e1 (first-PC variance share), er (effective rank),
        rankme (RankMe), anisotropy (avg pairwise cosine similarity).
    """
    is_numpy = isinstance(H, np.ndarray)
    if is_numpy:
        H = torch.from_numpy(H).float()

    H_bar = H - H.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(H_bar)
    S2 = S ** 2
    total = S2.sum()

    # E1: first-PC variance share
    e1 = (S2[0] / total).item()

    # Effective rank: exp(entropy of normalized eigenvalue spectrum)
    p = S2 / total
    p = p[p > 1e-12]
    er = torch.exp(-torch.sum(p * torch.log(p))).item()

    # RankMe: exp(entropy of normalized singular values)
    p_sv = S / S.sum()
    p_sv = p_sv[p_sv > 1e-12]
    rankme = torch.exp(-torch.sum(p_sv * torch.log(p_sv))).item()

    # Anisotropy: average pairwise cosine similarity
    H_norm = H_bar / (H_bar.norm(dim=1, keepdim=True) + 1e-8)
    if H_norm.shape[0] > 2000:
        idx = torch.randperm(H_norm.shape[0])[:2000]
        H_norm = H_norm[idx]
    cos_sim = H_norm @ H_norm.T
    n = cos_sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim.device)
    anisotropy = cos_sim[mask].mean().item()

    return {"e1": e1, "er": er, "rankme": rankme, "anisotropy": anisotropy}
