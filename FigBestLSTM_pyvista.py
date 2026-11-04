"""
Final main figure (PyVista): hero LSTM (R²=0.977) with proper 3D rendering.

4 panels in 2x2:
  (a) Ground-truth beliefs in PCA-3 of beliefs, with overlaid arrows for:
      - col(E) directions (NTP readouts, one per token)
      - kernel(E^T) directions (perpendicular subspace — what shuffle scrambles)
  (b) Ground-truth shuffled beliefs (geometric shuffle preserves col(E),
      randomizes ker(E^T))
  (c) LSTM fit of beliefs (regression of CONCAT-h activations onto beliefs)
  (d) LSTM fit of shuffled (regression of CONCAT-h activations onto shuffled)

Default: best LSTM seed = run_2 of 20260504050111_lstm_b64_full
(seed 456, R²=0.977 to beliefs).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.linalg import null_space

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv

from scripts.activation_analysis.run_shuffle_analysis import (
    list_checkpoints, _maybe_pca,
)
from scripts.activation_analysis.enumerate_process import enumerate_process_data
from scripts.activation_analysis.regression import (
    run_activation_to_beliefs_regression_kf,
)
from scripts.activation_analysis.quick_lstm_analyze import (
    load_lstm, extract_lstm_activations,
)


RCOND_SWEEP_LIST = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]


def regress_predict(acts, target, weights, n_folds=10):
    """Paper-style rcond-CV regression. Returns predictions on full data."""
    acts_t = torch.tensor(acts, dtype=torch.float32)
    tgt_t = torch.tensor(target, dtype=torch.float32)
    w_t = torch.tensor(weights, dtype=torch.float32)
    N = len(acts)
    idx = np.arange(N); fs = N // n_folds
    kf = []
    for i in range(n_folds):
        ts = i * fs
        te = ts + fs if i < n_folds - 1 else N
        kf.append((np.concatenate([idx[:ts], idx[te:]]), idx[ts:te]))
    return np.asarray(run_activation_to_beliefs_regression_kf(
        regression_analyzer=None, activations=acts_t, beliefs=tgt_t,
        probs=w_t, kf=kf, rcond_values=RCOND_SWEEP_LIST,
    )["predictions"])


def hsv_sphere_colors(coords3d):
    """Each point on the sphere gets a unique RGB from spherical HSV
    (hue=azimuth, sat=1, val=polar height). Pure colors at the equator.
    The value floor is lifted so low-z points stay vivid instead of going
    near-black, which would otherwise read as muddy brown when alpha-blended."""
    x, y, z = coords3d[:, 0], coords3d[:, 1], coords3d[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    sat = np.full_like(hue, 1.0)
    val = 0.55 + 0.45 * z_norm
    return mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1))


def ntp_colors(ntp_values, cmap_name=None):
    """Color points by their next-token probability via a 1D colormap.

    Default is a custom blue → dark-gray → red diverging map (matches the
    paper's blue/red convention for acts→beliefs / acts→shuffled, and avoids
    the white midpoint of `coolwarm`/`bwr` which disappears on white bg)."""
    import matplotlib
    if cmap_name is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "blue_gray_red",
            ["#1E88E5", "#4a4a4a", "#E53935"],
            N=256,
        )
    else:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    n = (ntp_values - ntp_values.min()) / (ntp_values.max() - ntp_values.min() + 1e-12)
    return cmap(n)[:, :3]


def alpha_from_w(w, min_a=0.05, power=2.0):
    return min_a + (1.0 - min_a) * (w / w.max()) ** power


def depth_alpha(coords3d, camera_eye, min_a=0.65):
    """Reduce opacity for points far from the camera — fakes a fog effect.
    Higher min_a keeps back points more vivid (less muddy when blended)."""
    d = np.linalg.norm(coords3d - np.asarray(camera_eye), axis=1)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-12)
    return 1.0 - (1.0 - min_a) * d_norm


def weighted_r2(target, pred, weights):
    w = weights / weights.sum()
    Y_mean = (target * w[:, None]).sum(axis=0)
    ss_res = (w[:, None] * (target - pred) ** 2).sum()
    ss_tot = (w[:, None] * (target - Y_mean) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def project_dirs(dirs, pca):
    """Project 4-D direction vectors into the PCA-3 basis.
    For directions (offsets, not points), don't subtract mean."""
    return dirs @ pca.components_.T


def render_panel(plotter, row, col, coords3d, rgba_per_point, sizes_per_point,
                 title, lims, ref_sphere_r=None, double_arrow=None,
                 camera_pos=None):
    """Render one panel.

    double_arrow: optional dict with keys 'center' (3-vec, origin of the arrow),
    'direction' (unit 3-vec), 'length' (per-side, so total span is 2*length),
    'color', 'shaft_radius', 'tip_length', 'tip_radius', 'label' (str).
    Draws two arrows from `center` in ±direction.
    """
    plotter.subplot(row, col)
    plotter.set_background("white")
    # Panel-letter labels are added later via matplotlib so they match the
    # serif-bold styling of (e)-(h). Skip the in-VTK title.

    # Translucent reference sphere
    if ref_sphere_r is not None:
        sphere = pv.Sphere(radius=ref_sphere_r, center=(0, 0, 0),
                           theta_resolution=24, phi_resolution=24)
        plotter.add_mesh(sphere, opacity=0.05, color="lightsteelblue",
                         smooth_shading=True)

    # Points as spheres with per-point color & size
    poly = pv.PolyData(coords3d)
    rgba_uint = (np.clip(rgba_per_point, 0, 1) * 255).astype(np.uint8)
    poly.point_data["rgba"] = rgba_uint
    poly.point_data["sizes"] = sizes_per_point.astype(np.float32)
    glyph = poly.glyph(orient=False, scale="sizes", factor=1.0,
                        geom=pv.Sphere(radius=1.0, theta_resolution=18,
                                       phi_resolution=18))
    plotter.add_mesh(glyph, scalars="rgba", rgba=True,
                     smooth_shading=True, lighting=True,
                     ambient=0.4, diffuse=0.65, specular=0.35,
                     specular_power=20)

    # Double-sided arrow centered at `center`, spanning ±length along direction
    if double_arrow is not None:
        center = np.asarray(double_arrow.get("center", (0.0, 0.0, 0.0)),
                             dtype=float)
        direction = np.asarray(double_arrow["direction"], dtype=float)
        L = float(double_arrow["length"])
        c = double_arrow.get("color", "#888888")
        sr = double_arrow.get("shaft_radius", 0.012)
        tl = double_arrow.get("tip_length", 0.16)
        tr = double_arrow.get("tip_radius", 0.04)
        for sign in (+1.0, -1.0):
            arr = pv.Arrow(start=tuple(center),
                            direction=tuple(sign * direction),
                            scale=L, tip_length=tl, tip_radius=tr,
                            shaft_radius=sr)
            plotter.add_mesh(arr, color=c, smooth_shading=True,
                             lighting=True, ambient=0.5, diffuse=0.6)
        label = double_arrow.get("label")
        if label:
            tip = center + direction * (L * 1.10)
            plotter.add_point_labels(
                np.array([tip]), [label],
                font_size=22, text_color=c, point_color=c, point_size=1,
                shape="rect", shape_color="white", shape_opacity=0.7,
                always_visible=True, render_points_as_spheres=False,
                bold=True,
            )

    if camera_pos is not None:
        plotter.camera_position = camera_pos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir",
                   default="results/20260504050111_lstm_b64_full/run_2_L4_H64_LSTM_seed456_quantum_rrxor",
                   help="LSTM run dir to use (default: best seed, R²=0.977)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="figures/beyond_ntp/best_lstm")
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=2400)
    p.add_argument("--variant", choices=["4panel", "6panel"], default="6panel",
                   help="4panel = a/b/c/d. 6panel = adds an HSV-colored "
                        "NTP-preserving-shuffle column between (a,c) and (b,d).")
    args = p.parse_args()

    rd = Path(args.run_dir)
    print(f"Using {rd.name}")
    ckpts = list_checkpoints(str(rd))
    model, cfg = load_lstm(str(rd), ckpts[-1], args.device)
    proc = enumerate_process_data(cfg)
    beliefs = proc["beliefs"]
    weights = proc["weights"]
    shuffled = proc["shuffled"]
    P_E = proc["P_E"]
    P_E_perp = proc["P_E_perp"]
    E = proc["E"]
    print(f"  N={beliefs.shape[0]}  d_belief={beliefs.shape[1]}  "
          f"d_vocab={E.shape[1]}  rank(E)={np.linalg.matrix_rank(E)}")

    # Extract activations
    acts = extract_lstm_activations(model, proc["seqs"][:, :-1].long(),
                                     vocab_size=2, device=args.device)
    h_layers = [acts[f"layer_{i}_h_n"] for i in range(4)
                if f"layer_{i}_h_n" in acts]
    A_raw = np.concatenate(h_layers, axis=1)
    A, _, _ = _maybe_pca(A_raw, 0.95)
    print(f"  PCA-95 CONCAT h: kept {A.shape[1]} of {A_raw.shape[1]}")

    # Run regressions
    pred_bel = regress_predict(A, beliefs, weights)
    pred_shuf = regress_predict(A, shuffled, weights)
    r2_bel = weighted_r2(beliefs, pred_bel, weights)
    r2_shuf_target = weighted_r2(shuffled, pred_shuf, weights)
    print(f"  R²(acts → beliefs)  = {r2_bel:.3f}")
    print(f"  R²(acts → shuffled) = {r2_shuf_target:.3f}")

    # PCA-3 basis fit on beliefs
    pca = PCA(n_components=3)
    pca.fit(beliefs)
    coords_bel = pca.transform(beliefs)
    coords_shuf = pca.transform(shuffled)
    coords_pbel = pca.transform(pred_bel)
    coords_pshuf = pca.transform(pred_shuf)
    print(f"  PCA-3 evr: {pca.explained_variance_ratio_.sum():.3f}")

    # NTP axis: direction in belief space along which p(token=0) varies.
    # That gradient is E[:, 0] (E maps belief → token probs row-wise).
    extent = max(np.ptp(coords_bel[:, i]) for i in range(3))
    ntp_axis_3d = E[:, 0] @ pca.components_.T
    ntp_axis_3d = ntp_axis_3d / (np.linalg.norm(ntp_axis_3d) + 1e-12)

    # Two color schemes:
    #   (a) GT beliefs and (c) LSTM fit beliefs — HSV-sphere (each point unique)
    #   (b) GT shuffled and (d) LSTM fit shuffled — colored by NTP (showing
    #       that the col(E) component / NTP is preserved through shuffle)
    rgb_belief = hsv_sphere_colors(coords_bel)
    ntp_values = proc["ntp"][:, 0]
    rgb_ntp = ntp_colors(ntp_values, cmap_name="viridis")

    # Predictions slightly bolder + bigger than GT (predictions cluster tighter,
    # so making them prominent surfaces the learned structure). Modest contrast.
    # Pred slightly bolder than GT but only marginally — shift values closer.
    pred_alpha = alpha_from_w(weights, min_a=0.18, power=1.85)
    gt_alpha = alpha_from_w(weights, min_a=0.14, power=2.0)

    w_norm = weights / weights.max()
    pred_sizes = (0.44 + 0.56 * w_norm ** 1.7) * (0.031 * extent)
    gt_sizes = (0.40 + 0.60 * w_norm ** 1.8) * (0.030 * extent)

    # Compute shared bounds for all panels
    all_pts = np.vstack([coords_bel, coords_shuf, coords_pbel, coords_pshuf])
    pad = 0.10 * np.ptp(all_pts, axis=0)
    lims = tuple((all_pts[:, i].min() - pad[i], all_pts[:, i].max() + pad[i])
                 for i in range(3))

    # Reference sphere radius — fit to GT beliefs (centered)
    bel_center = coords_bel.mean(axis=0)
    ref_r = np.linalg.norm(coords_bel - bel_center, axis=1).max() * 1.0

    # 2x2 plotter
    img_arr = _render_pyvista_with_retry(args, coords_bel, coords_shuf,
        coords_pbel, coords_pshuf, rgb_belief, rgb_ntp, ntp_axis_3d,
        bel_center, ref_r, extent, gt_alpha, pred_alpha, gt_sizes, pred_sizes,
        r2_bel, r2_shuf_target)
    raw = Path(args.outdir) / f"FigBestLSTM_pyvista_raw_{args.variant}.png"
    raw.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image as _Image
    _Image.fromarray(img_arr).save(str(raw))
    print(f"Saved raw {raw}")
    _build_composite(Path(args.outdir), img_arr, r2_bel, r2_shuf_target,
                     hero_seed=cfg.get("model_config", {}).get("seed",
                                  cfg.get("seed", 456)),
                     proc=proc, device=args.device, variant=args.variant)


def _render_pyvista_with_retry(args, coords_bel, coords_shuf, coords_pbel,
                                coords_pshuf, rgb_belief, rgb_ntp, ntp_axis_3d,
                                bel_center, ref_r, extent, gt_alpha, pred_alpha,
                                gt_sizes, pred_sizes, r2_bel, r2_shuf_target,
                                max_tries=6):
    """PyVista off-screen rendering is intermittently black on this VTK build
    (~30% failure rate). Render in a fresh Plotter each attempt, screenshot,
    and retry if the result is all-zeros."""
    cam = [(extent * 1.7, extent * 1.7, extent * 1.0), (0, 0, 0), (0, 0, 1)]
    cam_eye = np.array(cam[0])

    def _make_rgba(rgb, coords3d, kind):
        d_alpha = depth_alpha(coords3d, cam_eye, min_a=0.65)
        prob_a = gt_alpha if kind == "gt" else pred_alpha
        a = prob_a * d_alpha
        return np.concatenate([rgb, a[:, None]], axis=1)

    # NTP-axis arrow disabled (user adds annotation in Figma).
    ntp_arrow = None
    n_cols = 3 if args.variant == "6panel" else 2
    # Auto-size width so each panel ends up square: 2x2 → square image,
    # 2x3 → 3:2 image (each panel width = height = args.height // 2).
    pv_w = (args.height // 2) * n_cols
    pv_h = args.height
    for attempt in range(1, max_tries + 1):
        plotter = pv.Plotter(shape=(2, n_cols), off_screen=True,
                             window_size=[pv_w, pv_h],
                             border=False)
        # Row 0
        render_panel(plotter, 0, 0, coords_bel,
                     _make_rgba(rgb_belief, coords_bel, "gt"), gt_sizes,
                     "", None, ref_sphere_r=ref_r,
                     double_arrow=ntp_arrow, camera_pos=cam)
        if args.variant == "6panel":
            render_panel(plotter, 0, 1, coords_shuf,
                         _make_rgba(rgb_belief, coords_shuf, "gt"), gt_sizes,
                         "", None, ref_sphere_r=ref_r, camera_pos=cam)
            render_panel(plotter, 0, 2, coords_shuf,
                         _make_rgba(rgb_ntp, coords_shuf, "gt"), gt_sizes,
                         "", None, ref_sphere_r=ref_r, camera_pos=cam)
        else:
            render_panel(plotter, 0, 1, coords_shuf,
                         _make_rgba(rgb_ntp, coords_shuf, "gt"), gt_sizes,
                         "", None, ref_sphere_r=ref_r, camera_pos=cam)
        # Row 1
        render_panel(plotter, 1, 0, coords_pbel,
                     _make_rgba(rgb_belief, coords_pbel, "pred"), pred_sizes,
                     "", None, ref_sphere_r=ref_r, camera_pos=cam)
        if args.variant == "6panel":
            render_panel(plotter, 1, 1, coords_pshuf,
                         _make_rgba(rgb_belief, coords_pshuf, "pred"), pred_sizes,
                         "", None, ref_sphere_r=ref_r, camera_pos=cam)
            render_panel(plotter, 1, 2, coords_pshuf,
                         _make_rgba(rgb_ntp, coords_pshuf, "pred"), pred_sizes,
                         "", None, ref_sphere_r=ref_r, camera_pos=cam)
        else:
            render_panel(plotter, 1, 1, coords_pshuf,
                         _make_rgba(rgb_ntp, coords_pshuf, "pred"), pred_sizes,
                         "", None, ref_sphere_r=ref_r, camera_pos=cam)
        plotter.show(auto_close=False)
        img = plotter.screenshot(None, return_img=True,
                                  transparent_background=False)
        plotter.close()
        if img is not None and img.mean() > 1.0:
            print(f"PyVista render OK on try {attempt}")
            return img
        print(f"PyVista render returned black on try {attempt}, retrying ...")
    raise RuntimeError(f"PyVista rendering failed after {max_tries} tries")


# ---------------------------------------------------------------------------
# Paper-style subpanels (line per-layer + bar at CONCAT) + composite
# ---------------------------------------------------------------------------
CURVE_STYLES = {
    "acts_to_beliefs":  {"color": "#1565C0", "ls": "-", "lw": 2.4, "marker": "o",
                         "label": r"Activations $\to$ beliefs", "zorder": 10, "ms": 6},
    "acts_to_shuffled": {"color": "#C62828", "ls": "-", "lw": 2.4, "marker": "s",
                         "label": r"Activations $\to$ shuffled", "zorder": 9, "ms": 6},
    "ntp_to_beliefs":   {"color": "#7B1FA2", "ls": "--", "lw": 1.6, "marker": "^",
                         "label": r"Next-token $\to$ beliefs", "zorder": 5, "ms": 6},
}
LSTM_LAYER_ORDER = ["embed", "layer_0_h_n", "layer_1_h_n", "layer_2_h_n",
                    "layer_3_h_n", "layer_3_c_n", "CONCAT_h", "logits"]
LSTM_LAYER_LABELS = {
    "embed": "in", "layer_0_h_n": r"$h^0$",
    "layer_1_h_n": r"$h^1$", "layer_2_h_n": r"$h^2$",
    "layer_3_h_n": r"$h^3$", "layer_3_c_n": r"$c^3$",
    "CONCAT_h": "concat", "logits": "logit",
}


def _per_layer_curves_for_seed(joblib_path, seed):
    import joblib
    runs = joblib.load(joblib_path)
    for r in runs:
        if r.get("seed") == seed:
            return r["curves"]
    raise ValueError(f"seed {seed} not found in {joblib_path}")


def _panel_line(ax, layer_order, labels, curves, title, concat_key=None):
    """Line plot of R² across per-layer activations.

    If concat_key is in layer_order, it is plotted as a *floating* dot off to
    the right (separated by a gap, not connected by line) — the CONCAT
    representation is conceptually outside the per-layer sequence.
    """
    seq = [l for l in layer_order if l != concat_key]
    seq_labels = [labels[layer_order.index(l)] for l in seq]
    x_seq = np.arange(len(seq))

    has_concat = concat_key is not None and concat_key in layer_order
    gap = 1.05  # x-gap between last per-layer point and the floating concat dot
    if has_concat:
        x_concat = len(seq) - 1 + gap
        # subtle highlight band for the floating concat column
        ax.axvspan(x_concat - 0.40, x_concat + 0.40,
                   color="#e8eef7", alpha=0.6, zorder=0)

    # shaded gap between acts→beliefs and acts→shuffled (per-layer only)
    bel = np.array([curves[l].get("acts_to_beliefs", np.nan) for l in seq])
    shuf = np.array([curves[l].get("acts_to_shuffled", np.nan) for l in seq])
    ax.fill_between(x_seq, shuf, bel, alpha=0.12, color="#1565C0")

    for cn, st in CURVE_STYLES.items():
        # connected line over the per-layer sequence
        vs = np.array([curves[l].get(cn, np.nan) for l in seq])
        valid = ~np.isnan(vs)
        if valid.any():
            ax.plot(x_seq[valid], vs[valid], color=st["color"], ls=st["ls"],
                    lw=st["lw"], marker=st["marker"], markersize=st["ms"],
                    label=st["label"], zorder=st["zorder"])
        # floating CONCAT dot — same marker/color, no connecting line
        if has_concat and cn in curves[concat_key]:
            ax.plot([x_concat], [curves[concat_key][cn]],
                    ls="none", color=st["color"],
                    marker=st["marker"], markersize=st["ms"] + 1,
                    markeredgecolor=st["color"], markeredgewidth=0.5,
                    zorder=st["zorder"])

    if has_concat:
        all_x = list(x_seq) + [x_concat]
        all_labels = seq_labels + [labels[layer_order.index(concat_key)]]
    else:
        all_x = list(x_seq); all_labels = seq_labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=9, fontweight="medium",
                       rotation=30, ha="right")
    if has_concat:
        ax.axvline(x_concat - gap / 2, color="#aaaaaa", lw=0.6,
                   ls=":", alpha=0.7, zorder=0)
    ax.set_ylabel(r"$R^2$", fontsize=11, fontweight="medium")
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # No per-panel legend — one shared legend goes at the figure bottom.


def _panel_bars(ax, curves, layer_key, title):
    bar_curves = [
        ("acts_to_beliefs", r"Acts $\to$ Beliefs", "#1565C0"),
        ("ntp_to_beliefs", r"NTP $\to$ Beliefs", "#7B1FA2"),
        ("acts_to_shuffled", r"Acts $\to$ Shuffled", "#C62828"),
    ]
    items = [(lab, col, curves[layer_key].get(c, np.nan))
             for c, lab, col in bar_curves
             if c in curves[layer_key]]
    items = [it for it in items if not np.isnan(it[2])]
    labels = [it[0] for it in items]; colors = [it[1] for it in items]
    means = np.array([it[2] for it in items])
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.78, color=colors, alpha=0.85, edgecolor="none")
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.03, f"{m:.2f}", ha="center", va="bottom",
                fontsize=8, color="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#666666")
    ax.set_ylabel(r"$R^2$", fontsize=10, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color="#333333", rotation=18, ha="right")
    ax.tick_params(axis="x", length=0, width=0, pad=4)
    ax.tick_params(axis="y", labelsize=8, length=3, width=0.5, color="#666666")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, pad=8, color="#333333")
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)


TF_CURVES_JOBLIB = ("shuffle_results_quantum_rrxor_hero_3seed/"
                    "run_1_L4_H4_DH16_DM64_quantum_rrxor_shuffle.joblib")
TF_RUN_DIR = "results/20260424220339/run_1_L4_H4_DH16_DM64_quantum_rrxor"


def _tf_sort_key(k):
    if "resid_pre" in k:
        return (0, int(k.split(".")[1]))
    if "resid_post" in k and k.startswith("blocks."):
        return (1, int(k.split(".")[1]))
    if "ln_final" in k:
        return (2, 0)
    if k == "logits":
        return (3, 0)
    if k == "CONCAT_resid":
        return (5, 0)  # always last
    return (4, 0)


def _tf_label(k):
    if "resid_pre" in k:
        return f"L{k.split('.')[1]}"
    if "resid_post" in k and k.startswith("blocks."):
        return f"L{k.split('.')[1]}+"
    if "ln_final" in k:
        return "LN"
    if k == "logits":
        return "logit"
    if k == "CONCAT_resid":
        return "concat"
    return k


def _compute_transformer_concat(run_dir, proc, device):
    """Compute the transformer CONCAT-resid R² curves on the fly: concatenate
    resid_pre activations across all layers, PCA-95, regress to beliefs/shuffled
    and report the headline R² values. Returns dict in CURVE_STYLES schema."""
    if proc is None:
        return None
    try:
        from scripts.activation_analysis.run_shuffle_analysis import (
            list_checkpoints as _list_ckpts, load_model as _load_tf,
            extract_activations as _extract_tf, _maybe_pca as _pca,
        )
    except Exception as e:
        print(f"  [tf-concat] could not import transformer helpers: {e}")
        return None
    try:
        ckpts = _list_ckpts(str(run_dir))
        model = _load_tf(str(run_dir), ckpts[-1], device)
        acts = _extract_tf(model, proc["seqs"][:, :-1].long(), device)
    except Exception as e:
        print(f"  [tf-concat] could not load transformer / extract acts: {e}")
        return None
    pre_keys = sorted([k for k in acts if "resid_pre" in k],
                      key=lambda k: int(k.split(".")[1]))
    if not pre_keys:
        return None
    A_raw = np.concatenate([acts[k] for k in pre_keys], axis=1)
    A, _, _ = _pca(A_raw, 0.95)
    beliefs = proc["beliefs"]; shuffled = proc["shuffled"]
    weights = proc["weights"]
    pred_bel = regress_predict(A, beliefs, weights)
    pred_shuf = regress_predict(A, shuffled, weights)
    r2_bel = weighted_r2(beliefs, pred_bel, weights)
    r2_shuf = weighted_r2(shuffled, pred_shuf, weights)
    # NTP→beliefs is layer-agnostic (uses NTP probs as features). Read from any
    # cached layer's value or recompute trivially.
    ntp_to_bel = float("nan")
    return {
        "acts_to_beliefs": r2_bel,
        "acts_to_shuffled": r2_shuf,
        "ntp_to_beliefs": ntp_to_bel,
    }


def _build_composite(outdir, pv_img_arr, r2_bel, r2_shuf, hero_seed,
                     proc=None, device="cpu", variant="4panel"):
    """Combined figure: big LSTM PyVista on the left, 2x2 line/bar grid on the
    right with LSTM subpanels in the top row and transformer subpanels in the
    bottom row."""
    import joblib as _joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # LSTM curves (single hero seed) — drop the input embedding layer.
    lstm_curves = _per_layer_curves_for_seed(
        "figures/beyond_ntp/lstm_curves_b64_full.joblib", hero_seed)
    lstm_layers = [l for l in LSTM_LAYER_ORDER
                   if l in lstm_curves and l != "embed"]
    lstm_labels = [LSTM_LAYER_LABELS.get(l, l) for l in lstm_layers]

    # Transformer curves (best seed run_2 from the 3-seed hero sweep)
    tf_data = _joblib.load(TF_CURVES_JOBLIB)
    tf_curves = tf_data["checkpoint_results"][-1]["curves"]
    tf_seed = tf_data["config"]["model_config"].get("seed", "?")
    tf_run_dir = TF_RUN_DIR

    # Compute transformer CONCAT (concat resid_pre across all layers) live
    tf_concat_curves = _compute_transformer_concat(tf_run_dir, proc, device)
    if tf_concat_curves is not None:
        # NTP→beliefs is independent of layer (NTP probs are layer-agnostic
        # features); reuse the cached value from any layer.
        ref = next(iter(tf_curves.values()))
        if "ntp_to_beliefs" in ref:
            tf_concat_curves["ntp_to_beliefs"] = ref["ntp_to_beliefs"]
        tf_curves = dict(tf_curves)  # avoid mutating cached dict
        tf_curves["CONCAT_resid"] = tf_concat_curves

    tf_layers = sorted(
        [k for k in tf_curves if k != "blocks.0.hook_resid_pre"],
        key=_tf_sort_key,
    )
    tf_labels = [_tf_label(l) for l in tf_layers]
    tf_r2 = (tf_concat_curves or {}).get("acts_to_beliefs",
                tf_curves["blocks.3.hook_resid_pre"].get("acts_to_beliefs",
                    float("nan")))

    n_cols = 3 if variant == "6panel" else 2
    # Wider figure for 6-panel so the left half can fit 3 columns of squares.
    fig_w = 18.0 if variant == "6panel" else 16.0
    fig = plt.figure(figsize=(fig_w, 7.5))
    TOP, BOT = 0.92, 0.12
    # Left GridSpec covers the proportional width needed for n_cols squares.
    if variant == "6panel":
        left_lo, left_hi = 0.02, 0.66
        right_lo, right_hi = 0.72, 0.98
    else:
        left_lo, left_hi = 0.02, 0.58
        right_lo, right_hi = 0.65, 0.97
    gs_left = fig.add_gridspec(
        2, n_cols, left=left_lo, right=left_hi, top=TOP, bottom=BOT,
        wspace=0.05, hspace=0.20,
    )
    gs_right = fig.add_gridspec(
        2, 2, left=right_lo, right=right_hi, top=TOP, bottom=BOT,
        width_ratios=[1.55, 1.0],
        wspace=0.32, hspace=0.45,
    )
    h_img, w_img, _ = pv_img_arr.shape
    cell_w = w_img // n_cols
    cell_h = h_img // 2
    if variant == "6panel":
        # Letters a,b,c (top row) and d,e2,f2 (bottom). Use ascii a-f.
        quad = {
            "a": pv_img_arr[:cell_h, 0:cell_w],
            "b": pv_img_arr[:cell_h, cell_w:2 * cell_w],
            "c": pv_img_arr[:cell_h, 2 * cell_w:3 * cell_w],
            "d": pv_img_arr[cell_h:, 0:cell_w],
            "e2": pv_img_arr[cell_h:, cell_w:2 * cell_w],
            "f2": pv_img_arr[cell_h:, 2 * cell_w:3 * cell_w],
        }
        pv_titles = {
            "a": "Ground-truth beliefs",
            "b": "NTP-preserving shuffle (HSV)",
            "c": "NTP-preserving shuffle (NTP)",
            "d": f"Fit beliefs  ($R^2$={r2_bel:.2f})",
            "e2": "Fit shuffle (HSV)",
            "f2": f"Fit shuffle (NTP, $R^2$={r2_shuf:.2f})",
        }
        positions = [
            ("a", 0, 0), ("b", 0, 1), ("c", 0, 2),
            ("d", 1, 0), ("e2", 1, 1), ("f2", 1, 2),
        ]
    else:
        quad = {
            "a": pv_img_arr[:cell_h, :cell_w],
            "b": pv_img_arr[:cell_h, cell_w:],
            "c": pv_img_arr[cell_h:, :cell_w],
            "d": pv_img_arr[cell_h:, cell_w:],
        }
        pv_titles = {
            "a": "Ground-truth beliefs",
            "b": "NTP-preserving shuffle",
            "c": f"Fit beliefs  ($R^2$={r2_bel:.2f})",
            "d": f"Fit shuffle  ($R^2$={r2_shuf:.2f})",
        }
        positions = [("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 1)]
    pv_axes = {}
    for letter, r, c in positions:
        ax = fig.add_subplot(gs_left[r, c])
        ax.imshow(quad[letter])
        ax.set_box_aspect(1.0)
        ax.axis("off")
        ax.set_title(pv_titles[letter], fontsize=11, pad=8)
        pv_axes[letter] = ax

    ax_e = fig.add_subplot(gs_right[0, 0])
    ax_f = fig.add_subplot(gs_right[0, 1])
    ax_g = fig.add_subplot(gs_right[1, 0])
    ax_h = fig.add_subplot(gs_right[1, 1])

    _panel_line(ax_e, lstm_layers, lstm_labels, lstm_curves,
                title="LSTM", concat_key="CONCAT_h")
    _panel_bars(ax_f, lstm_curves, "CONCAT_h", "LSTM")
    _panel_line(ax_g, tf_layers, tf_labels, tf_curves,
                title="Transformer",
                concat_key="CONCAT_resid" if "CONCAT_resid" in tf_curves else None)
    tf_bar_key = ("CONCAT_resid" if "CONCAT_resid" in tf_curves
                  else "blocks.3.hook_resid_pre")
    _panel_bars(ax_h, tf_curves, tf_bar_key, "Transformer")

    # Panel letters — same styling for left PyVista panels and right plots.
    if variant == "6panel":
        # 6 PyVista letters a-f, then 4 right-side letters g-j.
        letter_axes = [pv_axes["a"], pv_axes["b"], pv_axes["c"],
                       pv_axes["d"], pv_axes["e2"], pv_axes["f2"],
                       ax_e, ax_f, ax_g, ax_h]
        letters = "abcdefghij"
    else:
        letter_axes = [pv_axes["a"], pv_axes["b"], pv_axes["c"], pv_axes["d"],
                       ax_e, ax_f, ax_g, ax_h]
        letters = "abcdefgh"
    for ax, letter in zip(letter_axes, letters):
        ax.text(-0.05, 1.16, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                family="serif")

    # Shared legend (curve styles only, lifted from one of the line plots).
    handles, labels = ax_e.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.76, 0.025), ncol=3, framealpha=0.0,
                   edgecolor="none", fancybox=False, fontsize=8,
                   handlelength=1.6, columnspacing=1.2, handletextpad=0.4)

    fig.text(0.5, -0.02,
             f"LSTM hero seed {hero_seed} (4-layer, $h$=64, $b$=64) · "
             f"transformer hero seed {tf_seed} (4-layer muP, $d$=64, $b$=16) · "
             "qRRXOR ($\\varphi=\\pi/2$, $\\theta=\\pi/4$, $\\varepsilon$=0.02) · "
             "95\\%-variance PCA · rcond CV-selected · weighted $R^2$",
             ha="center", va="top", fontsize=8, color="#555555",
             family="serif", style="italic")

    # 6panel is the canonical figure (no suffix); 4panel is the alternate.
    suffix = "" if variant == "6panel" else f"_{variant}"
    base = f"FigBestLSTM_pyvista{suffix}"
    out = outdir / f"{base}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(outdir / f"{base}.pdf", bbox_inches="tight")
    # SVG for Figma — text & layout stay editable; the imshow PyVista panels
    # are still embedded as raster but the rest is vector.
    fig.savefig(outdir / f"{base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} (.png/.pdf/.svg)")


if __name__ == "__main__":
    main()
