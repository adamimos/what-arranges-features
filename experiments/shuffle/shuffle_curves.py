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
    uv run python experiments/shuffle/shuffle_curves.py \
        --results_dir results/SWEEP_ID \
        --outdir shuffle_results

    # Single run, all checkpoints for training dynamics:
    uv run python experiments/shuffle/shuffle_curves.py \
        --results_dir results/SWEEP_ID \
        --run_index 0 --all_checkpoints
"""
import argparse
import os
import joblib
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm


from epsilon_transformers.analysis.shuffle_control import (
    geometric_shuffle, _paper_regression_r2,
)
from epsilon_transformers.analysis.checkpoints import (
    _maybe_pca, load_run_config, list_checkpoints, load_model, load_random_model,
    extract_activations,
)
from epsilon_transformers.analysis.process_targets import prepare_process_data


DEVICE = "cuda:0"
N_SHUFFLES = 200
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------


def get_layer_order(acts_keys):
    """Return layer keys in a sensible plotting order."""
    layers = [k for k in acts_keys if k.startswith("blocks.") and "resid_pre" in k]
    layers.sort(key=lambda k: int(k.split(".")[1]))
    post = [k for k in acts_keys if "resid_post" in k and k.startswith("blocks.")]
    ln = [k for k in acts_keys if "ln_final" in k]
    logit = [k for k in acts_keys if k == "logits"]
    return layers + post + ln + logit


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
