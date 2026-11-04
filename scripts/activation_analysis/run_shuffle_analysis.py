"""
Shuffle Control Analysis for Quantum RRXOR Experiments.

For each trained model, computes 6 regression curves across layers:
  1. Activations → beliefs           (main result)
  2. Activations → shuffled beliefs   (shuffle control)
  3. Next-token probs → beliefs       (NTP baseline)
  4. Beliefs → shuffled beliefs       (theoretical control)
  5. Next-token probs → shuffled      (shuffle indifference check)
  6. Random activations → beliefs     (random baseline)

Also supports running across multiple checkpoints for training dynamics.

Usage:
    uv run python scripts/activation_analysis/run_shuffle_analysis.py \
        --results_dir results/SWEEP_ID \
        --outdir shuffle_results

    # Single run, all checkpoints for training dynamics:
    uv run python scripts/activation_analysis/run_shuffle_analysis.py \
        --results_dir results/SWEEP_ID \
        --run_index 0 --all_checkpoints
"""
import argparse
import json
import os
import yaml
import joblib
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.training.dataloader import generate_all_seqs
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix,
    compute_projections,
    geometric_shuffle,
    local_geometric_shuffle,
    _paper_regression_r2,
)


DEVICE = "cuda:0"
N_SHUFFLES = 200
N_FOLDS = 5
LOCAL_BIN_SIZE = 8  # stronger control: shuffle only within NTP-neighbors


def _maybe_pca(features: np.ndarray, rank):
    """Return (transformed_features, explained_variance_ratio, n_components).

    Accepts rank as:
      - None or <=0: no PCA applied, returns features unchanged.
      - int >=1: keep that many PCs.
      - float in (0, 1): keep enough PCs to explain at least that fraction
        of total variance.

    Eliminates input-memorization artifact when activations are wider than
    needed: with d_model >> belief_dim and few unique inputs, raw activations
    let regression memorize the prefix lookup. Restricting to top-`rank`
    principal components of activations breaks that.
    """
    if rank is None:
        return features, None, None
    from sklearn.decomposition import PCA
    if isinstance(rank, float) and 0.0 < rank < 1.0:
        pca = PCA(n_components=rank, svd_solver="full")
    elif isinstance(rank, (int, np.integer)) and rank > 0 and rank < features.shape[1]:
        pca = PCA(n_components=int(rank))
    else:
        return features, None, None
    transformed = pca.fit_transform(features)
    return (transformed,
            pca.explained_variance_ratio_.tolist(),
            int(pca.n_components_))


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def load_run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        return yaml.safe_load(f)


def list_checkpoints(run_dir: str) -> list:
    """List all checkpoint files sorted by token count."""
    pts = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    pts.sort(key=lambda f: int(f.replace(".pt", "")))
    return pts


def load_model(run_dir: str, checkpoint: str, device: str) -> HookedTransformer:
    with open(os.path.join(run_dir, "hooked_model_config.json")) as f:
        model_cfg = json.load(f)
    model_cfg["dtype"] = getattr(torch, model_cfg["dtype"].split(".")[-1])
    model_cfg["device"] = device
    hooked_config = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hooked_config)
    model = model.to(device)
    state_dict = torch.load(os.path.join(run_dir, checkpoint),
                            map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_random_model(run_dir: str, device: str, seed: int = 999) -> HookedTransformer:
    """Load model architecture with random weights (no checkpoint)."""
    with open(os.path.join(run_dir, "hooked_model_config.json")) as f:
        model_cfg = json.load(f)
    model_cfg["dtype"] = getattr(torch, model_cfg["dtype"].split(".")[-1])
    model_cfg["device"] = device
    model_cfg["seed"] = seed
    hooked_config = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hooked_config)
    model = model.to(device)
    model.eval()
    return model


def extract_activations(model, inputs, device):
    """Extract activations from all layers at last token position.

    Includes: resid_pre (each layer), resid_post (last layer),
    post-layernorm (ln_final), and logits.
    """
    inputs = inputs.to(device)
    n_layers = model.cfg.n_layers

    hook_names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    hook_names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    hook_names.append("ln_final.hook_normalized")

    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs, names_filter=hook_names)

    acts = {}
    for name in hook_names:
        acts[name] = cache[name][:, -1, :].cpu().numpy()

    # Logits at last position: (batch, vocab_size)
    acts["logits"] = logits[:, -1, :].cpu().numpy()

    return acts


def get_layer_order(acts_keys):
    """Return layer keys in a sensible plotting order."""
    layers = [k for k in acts_keys if k.startswith("blocks.") and "resid_pre" in k]
    layers.sort(key=lambda k: int(k.split(".")[1]))
    post = [k for k in acts_keys if "resid_post" in k and k.startswith("blocks.")]
    ln = [k for k in acts_keys if "ln_final" in k]
    logit = [k for k in acts_keys if k == "logits"]
    return layers + post + ln + logit


# ---------------------------------------------------------------------------
# Prepare process data (beliefs, next-token probs, shuffled targets)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Compute all 6 curves for one checkpoint
# ---------------------------------------------------------------------------

def compute_curves(acts, random_acts, proc_data, device="cpu", pca_rank=None):
    """Compute R² for all regression curves across layers.

    If pca_rank is set, activations and random_acts are projected onto their
    top `pca_rank` PCs before regression (kills input-memorization artifact
    in random baseline). Each layer/source gets its own PCA basis.

    Returns dict: layer_name -> {curve_name: r2, "pca_evr_acts"?, "pca_evr_random"?}
    """
    beliefs = proc_data["beliefs"]
    shuffled = proc_data["shuffled"]
    local_shuffled = proc_data["local_shuffled"]
    ntp = proc_data["ntp"]
    weights = proc_data["weights"]

    layer_order = get_layer_order(acts.keys())
    results = {}

    for layer in layer_order:
        act_raw = acts[layer]
        rand_raw = random_acts[layer]
        act, evr_acts, n_acts = _maybe_pca(act_raw, pca_rank)
        rand_act, evr_rand, n_rand = _maybe_pca(rand_raw, pca_rank)

        r = {}
        r["acts_to_beliefs"] = _paper_regression_r2(act, beliefs, weights, device=device)
        r["acts_to_shuffled"] = _paper_regression_r2(act, shuffled, weights, device=device)
        r["acts_to_local_shuffled"] = _paper_regression_r2(act, local_shuffled, weights, device=device)
        r["ntp_to_beliefs"] = _paper_regression_r2(ntp, beliefs, weights, device=device)
        r["beliefs_to_shuffled"] = _paper_regression_r2(beliefs, shuffled, weights, device=device)
        r["ntp_to_shuffled"] = _paper_regression_r2(ntp, shuffled, weights, device=device)
        r["random_to_beliefs"] = _paper_regression_r2(rand_act, beliefs, weights, device=device)
        r["random_to_shuffled"] = _paper_regression_r2(rand_act, shuffled, weights, device=device)
        r["random_to_local_shuffled"] = _paper_regression_r2(rand_act, local_shuffled, weights, device=device)
        if evr_acts is not None:
            r["pca_evr_acts"] = evr_acts
            r["pca_evr_random"] = evr_rand
            r["pca_evr_acts_total"] = float(sum(evr_acts))
            r["pca_evr_random_total"] = float(sum(evr_rand))
            r["pca_n_acts"] = n_acts
            r["pca_n_random"] = n_rand

        results[layer] = r

    return results


# ---------------------------------------------------------------------------
# Shuffle distribution (for p-values)
# ---------------------------------------------------------------------------

def compute_shuffle_distribution(acts, proc_data, n_shuffles=N_SHUFFLES,
                                  seed=42, device="cpu"):
    """Run multiple shuffles on the final layer to get distribution of R²."""
    rng = np.random.default_rng(seed)
    beliefs = proc_data["beliefs"]
    weights = proc_data["weights"]
    P_E = proc_data["P_E"]
    P_E_perp = proc_data["P_E_perp"]

    # Use the last resid_post layer
    layer_keys = get_layer_order(acts.keys())
    last_layer = [k for k in layer_keys if "resid_post" in k][-1]
    act = acts[last_layer]

    # Original
    r2_orig = _paper_regression_r2(act, beliefs, weights, device=device)

    # Shuffled distribution
    r2_shuffled = []
    for _ in tqdm(range(n_shuffles), desc="Shuffle distribution", leave=False):
        shuf = geometric_shuffle(beliefs, P_E, P_E_perp, rng)
        r2 = _paper_regression_r2(act, shuf, weights, device=device)
        r2_shuffled.append(r2)

    return {
        "r2_original": r2_orig,
        "r2_shuffled": r2_shuffled,
        "layer": last_layer,
        "p_value": float(np.mean(np.array(r2_shuffled) >= r2_orig)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_checkpoint(run_dir, checkpoint, proc_data, random_acts,
                       device, n_shuffles, pca_rank=None):
    """Full analysis for a single checkpoint."""
    model = load_model(run_dir, checkpoint, device)
    acts = extract_activations(model, proc_data["seqs"][:, :-1].long(), device)
    del model
    torch.cuda.empty_cache()

    reg_device = device if "cuda" in device else "cpu"
    curves = compute_curves(acts, random_acts, proc_data, device=reg_device,
                            pca_rank=pca_rank)
    shuffle_dist = compute_shuffle_distribution(
        acts, proc_data, n_shuffles=n_shuffles, device=reg_device
    )

    tokens_seen = int(checkpoint.replace(".pt", ""))
    return {
        "checkpoint": checkpoint,
        "tokens_seen": tokens_seen,
        "curves": curves,
        "shuffle_dist": shuffle_dist,
    }


def analyze_single_run(run_dir, outdir, device=DEVICE, n_shuffles=N_SHUFFLES,
                       all_checkpoints=False, pca_rank=None):
    """Run full analysis on a single training run."""
    config = load_run_config(run_dir)
    run_name = os.path.basename(run_dir)
    print(f"\n{'='*60}")
    print(f"Analyzing: {run_name}")
    print(f"{'='*60}")

    # Prepare process data (independent of checkpoint)
    proc_data = prepare_process_data(config)
    print(f"  Process: {config['process_config']['name']}")
    print(f"  Sequences: {proc_data['seqs'].shape[0]}, "
          f"Beliefs: {proc_data['beliefs'].shape}")

    # Random baseline activations (computed once)
    print("  Computing random baseline...")
    random_model = load_random_model(run_dir, device)
    random_acts = extract_activations(
        random_model, proc_data["seqs"][:, :-1].long(), device
    )
    del random_model
    torch.cuda.empty_cache()

    # Select checkpoints
    all_ckpts = list_checkpoints(run_dir)
    if all_checkpoints:
        # Sample ~20 checkpoints evenly spaced + initial + final
        n_total = len(all_ckpts)
        if n_total <= 20:
            selected = all_ckpts
        else:
            indices = np.linspace(0, n_total - 1, 20, dtype=int)
            selected = [all_ckpts[i] for i in indices]
            # Ensure initial and final are included
            if all_ckpts[0] not in selected:
                selected = [all_ckpts[0]] + selected
            if all_ckpts[-1] not in selected:
                selected.append(all_ckpts[-1])
    else:
        # Just initial and final
        selected = [all_ckpts[0], all_ckpts[-1]]

    print(f"  Analyzing {len(selected)} checkpoints: "
          f"{selected[0]} ... {selected[-1]}")

    # Analyze each checkpoint
    checkpoint_results = []
    for ckpt in tqdm(selected, desc="Checkpoints"):
        print(f"\n  Checkpoint: {ckpt}")
        result = analyze_checkpoint(
            run_dir, ckpt, proc_data, random_acts, device, n_shuffles,
            pca_rank=pca_rank,
        )
        checkpoint_results.append(result)

        # Print summary for this checkpoint (use last resid_post layer)
        layer_keys = list(result["curves"].keys())
        last_layer = [k for k in layer_keys if "resid_post" in k][-1]
        c = result["curves"][last_layer]
        sd = result["shuffle_dist"]
        print(f"    [{last_layer}] R² acts→beliefs: {c['acts_to_beliefs']:.4f}")
        print(f"    [{last_layer}] R² acts→shuffled: {c['acts_to_shuffled']:.4f}")
        print(f"    [{last_layer}] R² ntp→beliefs:   {c['ntp_to_beliefs']:.4f}")
        print(f"    [{last_layer}] R² random→beliefs:{c['random_to_beliefs']:.4f}")
        if "pca_evr_acts_total" in c:
            print(f"    [{last_layer}] PCA(acts)   evr total: {c['pca_evr_acts_total']:.4f}")
            print(f"    [{last_layer}] PCA(random) evr total: {c['pca_evr_random_total']:.4f}")
        print(f"    shuffle p-val: {sd['p_value']:.4f}")

    # Save
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{run_name}_shuffle.joblib")
    save_data = {
        "config": config,
        "process_params": config["process_config"],
        "checkpoint_results": checkpoint_results,
        "beliefs_shape": proc_data["beliefs"].shape,
        "n_shuffles": n_shuffles,
        "pca_rank": pca_rank,
    }
    joblib.dump(save_data, out_path)
    print(f"\n  Saved to {out_path}")

    del random_acts
    torch.cuda.empty_cache()
    return save_data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="shuffle_results")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--n_shuffles", type=int, default=N_SHUFFLES)
    parser.add_argument("--run_index", type=int, default=None,
                        help="Only analyze this run index (0-7)")
    parser.add_argument("--all_checkpoints", action="store_true",
                        help="Analyze ~20 checkpoints for training dynamics")
    parser.add_argument("--pca_rank", type=str, default=None,
                        help="PCA rank to project activations onto before regression. "
                             "Int (e.g. '8') = keep top-K PCs. Float in (0,1) "
                             "(e.g. '0.95') = keep enough PCs to explain that fraction "
                             "of variance. Unset = no PCA.")
    args = parser.parse_args()
    if args.pca_rank is not None:
        try:
            v = float(args.pca_rank)
            args.pca_rank = int(v) if v >= 1 else v
        except ValueError:
            args.pca_rank = None

    results_dir = Path(args.results_dir)
    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])

    if args.run_index is not None:
        run_dirs = [run_dirs[args.run_index]]

    print(f"Found {len(run_dirs)} runs | device={args.device} | "
          f"shuffles={args.n_shuffles} | all_ckpts={args.all_checkpoints}")

    for run_dir in run_dirs:
        try:
            analyze_single_run(
                str(run_dir), args.outdir, pca_rank=args.pca_rank,
                device=args.device,
                n_shuffles=args.n_shuffles,
                all_checkpoints=args.all_checkpoints,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
