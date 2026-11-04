"""
Process targets for probing: beliefs, next-token probabilities and NTP-preserving shuffled
beliefs for a GHMM process.

`enumerate_process_data` enumerates ALL vocab^n_ctx sequences and returns the belief at the
last position with the exact sequence probability (every unique prefix appears once).
`prepare_process_data` returns the same quantities from the full enumeration of length
n_ctx+1 sequences when that is small enough, or from sampled trajectories otherwise.
"""
from __future__ import annotations

import numpy as np
import torch

from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.analysis.shuffle_control import (
    compute_emission_matrix, compute_projections, geometric_shuffle, local_geometric_shuffle,
)
from epsilon_transformers.training.dataloader import generate_all_seqs


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


LOCAL_BIN_SIZE = 8  # stronger control: shuffle only within NTP-neighbors


def prepare_process_data(config, n_shuffles=1, seed=42,
                         max_enumerate_log2=20, n_sampled_prefixes=50000):
    """Generate beliefs, next-token probs, and shuffled beliefs from the process.

    For small n_ctx (2^(n_ctx+1) sequences <= 2^max_enumerate_log2), enumerate
    all sequences exactly (weights = p(prefix)).

    For large n_ctx (would exceed that budget), sample `n_sampled_prefixes`
    trajectories by running the process forward with online belief updates
    (Monte-Carlo approximation to the weighted expectation). Each sample gets
    uniform weight 1/N; larger N improves the regression estimate.
    """
    process_params = config["process_config"]
    T = get_matrix_from_args(**process_params)
    ghmm = TransitionMatrixGHMM(T)
    ghmm.name = process_params["name"]
    rev = ghmm.right_eigenvector.squeeze()

    n_ctx = config["model_config"]["n_ctx"]
    # Decide: enumerate or sample?
    if (n_ctx + 1) <= max_enumerate_log2:
        return _prepare_enumerated(T, ghmm, rev, n_ctx, seed)
    return _prepare_sampled(T, ghmm, rev, n_ctx, n_sampled_prefixes, seed)


def _prepare_enumerated(T, ghmm, rev, n_ctx, seed):
    """Exact belief cloud via full enumeration. Only works for small n_ctx."""
    seqs, probs, _ = generate_all_seqs(ghmm, n_ctx + 1, bos=False)
    tree = ghmm.derive_mixed_state_tree(depth=n_ctx + 2)

    # Dedupe to unique 8-token prefixes. generate_all_seqs enumerates length
    # n_ctx+1 sequences (so we can compute p(next|context)), but the model only
    # sees the first n_ctx tokens — so two sequences sharing the same first
    # n_ctx tokens produce the same activation and belief, and should be a
    # single sample with weight = sum of their sequence probabilities.
    seqs_np = seqs.numpy()
    probs_np = probs.numpy()
    prefixes = [tuple(int(t) for t in row[:-1]) for row in seqs_np]
    prefix_to_idx = {}
    prefix_weights = {}
    for i, p in enumerate(prefixes):
        if p not in prefix_to_idx:
            prefix_to_idx[p] = i
            prefix_weights[p] = 0.0
        prefix_weights[p] += probs_np[i]
    unique_indices = list(prefix_to_idx.values())
    seqs = seqs[unique_indices]
    # Belief states at the last position the MODEL sees.
    # seqs now has n_ctx+1 tokens; the model input is seqs[:, :-1] (n_ctx tokens).
    # bs[-1] on the truncated path is the belief after the model's last input token.
    beliefs_list = []
    for path in [list(s.numpy()) for s in seqs]:
        bs = tree.path_to_beliefs(path[:-1])
        beliefs_list.append(bs[-1].squeeze())
    beliefs = np.array(beliefs_list)
    weights = np.array([prefix_weights[prefixes[i]] for i in unique_indices])

    # Next-token probabilities: p(x=0 | belief)
    # p(0|eta) = eta @ T[0] @ rev / (eta @ rev)
    p0 = (beliefs @ T[0] @ rev) / (beliefs @ rev)
    ntp = p0.reshape(-1, 1)  # (N, 1) — single feature for binary tokens

    # Emission matrix and projections
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)

    # Generate shuffled beliefs (global) and local (within NTP-neighbor bins)
    rng = np.random.default_rng(seed)
    shuffled = geometric_shuffle(beliefs, P_E, P_E_perp, rng)
    rng_local = np.random.default_rng(seed)
    local_shuffled = local_geometric_shuffle(
        beliefs, P_E, P_E_perp, ntp[:, 0], rng_local, bin_size=LOCAL_BIN_SIZE
    )

    return {
        "seqs": seqs,
        "beliefs": beliefs,
        "weights": weights,
        "ntp": ntp,
        "shuffled": shuffled,
        "local_shuffled": local_shuffled,
        "local_bin_size": LOCAL_BIN_SIZE,
        "T": T,
        "rev": rev,
        "E": E,
        "P_E": P_E,
        "P_E_perp": P_E_perp,
        "mode": "enumerated",
    }


def _prepare_sampled(T, ghmm, rev, n_ctx, n_samples, seed):
    """Monte-Carlo belief cloud: sample n_samples trajectories from the process,
    compute the belief state at the last position of each. Uniform weights.

    This avoids enumerating 2^n_ctx prefixes — tractable for any n_ctx.

    Belief update: starting from the stationary belief b0 (scaled so b0 @ rev = 1),
    given an observed token k, update b <- (b @ T[k]) / (b @ T[k] @ rev), keeping
    it in the "normalized against right eigenvector" convention used by the tree.
    """
    vocab = T.shape[0]
    n_states = T.shape[-1]
    # Stationary belief: left eigenvector of sum_k T[k] (the process transition),
    # normalized so that b0 @ rev = 1 — matches tree.path_to_beliefs convention.
    P = T.sum(axis=0)  # sum over emissions = unconditional transition
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    b0 = np.real(eigvecs[:, idx])
    b0 = b0 / (b0 @ rev)

    rng = np.random.default_rng(seed)
    seqs = np.zeros((n_samples, n_ctx + 1), dtype=np.int64)
    beliefs = np.zeros((n_samples, n_states), dtype=np.float64)
    for i in range(n_samples):
        b = b0.copy()
        for t in range(n_ctx):
            # p(token k | b) = (b @ T[k] @ rev) / (b @ rev) with b @ rev = 1
            p = np.array([max(float(b @ T[k] @ rev), 0.0) for k in range(vocab)])
            s = p.sum()
            if s <= 0:
                tok = int(rng.integers(vocab))
            else:
                tok = int(rng.choice(vocab, p=p / s))
            seqs[i, t] = tok
            b_new = b @ T[tok]
            norm = float(b_new @ rev)
            if norm > 0:
                b = b_new / norm
            else:
                b = b0.copy()
        # Sample the (n_ctx+1)-th token too (not used by model but kept for shape parity)
        p = np.array([max(float(b @ T[k] @ rev), 0.0) for k in range(vocab)])
        s = p.sum()
        seqs[i, n_ctx] = int(rng.choice(vocab, p=p / s)) if s > 0 else 0
        beliefs[i] = b

    weights = np.ones(n_samples) / n_samples
    p0 = (beliefs @ T[0] @ rev) / np.maximum(beliefs @ rev, 1e-12)
    ntp = p0.reshape(-1, 1)

    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)

    rng_sh = np.random.default_rng(seed)
    shuffled = geometric_shuffle(beliefs, P_E, P_E_perp, rng_sh)
    rng_local = np.random.default_rng(seed)
    local_shuffled = local_geometric_shuffle(
        beliefs, P_E, P_E_perp, ntp[:, 0], rng_local, bin_size=LOCAL_BIN_SIZE
    )

    import torch as _torch
    seqs_t = _torch.from_numpy(seqs)
    return {
        "seqs": seqs_t,
        "beliefs": beliefs,
        "weights": weights,
        "ntp": ntp,
        "shuffled": shuffled,
        "local_shuffled": local_shuffled,
        "local_bin_size": LOCAL_BIN_SIZE,
        "T": T,
        "rev": rev,
        "E": E,
        "P_E": P_E,
        "P_E_perp": P_E_perp,
        "mode": f"sampled_N{n_samples}",
    }
