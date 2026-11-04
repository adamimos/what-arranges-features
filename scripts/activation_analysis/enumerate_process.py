"""
Enumerate ALL 2^n_ctx sequences for a GHMM process and compute
(belief at last position, probability of sequence). Substitute for
`prepare_process_data` when you want the full belief manifold (every
unique prefix appears once, weighted by true probability) rather than a
sample.
"""
from __future__ import annotations

import numpy as np
import torch

from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix, compute_projections,
    geometric_shuffle, local_geometric_shuffle,
)


def _stationary_belief(T, rev):
    """Stationary belief b0 with b0 @ rev = 1 (matches sampling convention)."""
    P = T.sum(axis=0)
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    b0 = np.real(eigvecs[:, idx])
    return b0 / (b0 @ rev)


def enumerate_process_data(config: dict, local_bin_size: int = 8,
                            seed: int = 42) -> dict:
    """Enumerate all sequences of length n_ctx, return same keys as
    `prepare_process_data` plus the seqs ordered by enumeration index.

    Returns dict with keys:
        seqs        : torch.LongTensor (V^n_ctx, n_ctx + 1)
        beliefs     : ndarray (V^n_ctx, latent_dim)  - belief at last position
        weights     : ndarray (V^n_ctx,)             - true sequence probability
        ntp         : ndarray (V^n_ctx, 1)           - p(token=0 | belief at last)
        shuffled    : geometric-shuffled beliefs (preserves NTP component)
        local_shuffled
        T, rev, E, P_E, P_E_perp
        local_bin_size
        mode        : "enumerated"

    Note: weights sum to 1.
    """
    pc = config["process_config"].copy()
    name = pc.pop("name")
    T = get_matrix_from_args(name=name, **pc)
    g = TransitionMatrixGHMM(T)
    rev = g.rev.squeeze() if hasattr(g, "rev") else None
    if rev is None:
        # GHMM may not expose rev directly; reconstruct as right-eigvec of sum_x T[x]
        P = T.sum(axis=0)
        eigvals, eigvecs = np.linalg.eig(P)
        idx = np.argmin(np.abs(eigvals - 1.0))
        rev = np.real(eigvecs[:, idx])
        rev = rev / rev.sum()  # normalize so 1^T rev makes sense
    rev = np.asarray(rev).squeeze()

    n_ctx = config["model_config"]["n_ctx"]
    vocab = T.shape[0]
    n_states = T.shape[1]
    n_seqs = vocab ** n_ctx

    b0 = _stationary_belief(T, rev)

    # Pre-fill — store seq tokens, final belief, sequence probability
    seqs = np.zeros((n_seqs, n_ctx + 1), dtype=np.int64)
    beliefs = np.zeros((n_seqs, n_states), dtype=np.float64)
    weights = np.zeros(n_seqs, dtype=np.float64)
    ntp_last = np.zeros((n_seqs, 1), dtype=np.float64)

    for i in range(n_seqs):
        # Decode sequence (lex order in base `vocab`, big-endian over time)
        s = i
        toks = np.zeros(n_ctx, dtype=np.int64)
        for t in range(n_ctx):
            toks[n_ctx - 1 - t] = s % vocab
            s //= vocab

        b = b0.copy()
        prob = 1.0
        for x in toks:
            p_x = max(float(b @ T[x] @ rev), 0.0)
            prob *= p_x
            b_new = b @ T[x]
            norm = float(b_new @ rev)
            if norm > 0:
                b = b_new / norm

        seqs[i, :n_ctx] = toks
        # n_ctx-th token: pretend uniform (not used by model, but kept for shape parity)
        seqs[i, n_ctx] = 0
        beliefs[i] = b
        weights[i] = prob
        ntp_last[i, 0] = max(float(b @ T[0] @ rev), 0.0)

    # Probabilities should sum to ~1; renormalize for numerical safety
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(n_seqs) / n_seqs

    # Same shuffle/projection scaffolding as prepare_process_data
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    rng_sh = np.random.default_rng(seed)
    shuffled = geometric_shuffle(beliefs, P_E, P_E_perp, rng_sh)
    rng_local = np.random.default_rng(seed)
    local_shuffled = local_geometric_shuffle(
        beliefs, P_E, P_E_perp, ntp_last[:, 0], rng_local,
        bin_size=local_bin_size,
    )

    return {
        "seqs": torch.from_numpy(seqs),
        "beliefs": beliefs,
        "weights": weights,
        "ntp": ntp_last,
        "shuffled": shuffled,
        "local_shuffled": local_shuffled,
        "local_bin_size": local_bin_size,
        "T": T, "rev": rev, "E": E,
        "P_E": P_E, "P_E_perp": P_E_perp,
        "mode": "enumerated",
    }
