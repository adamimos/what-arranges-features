"""Run directories: configs, checkpoint lists, transformer loading, activation extraction, PCA."""

import json
import os

import numpy as np
import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig


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
