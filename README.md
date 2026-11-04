# What Arranges Features in Activation Space? Non-Classical Predictive Geometry in Next-Token Predictors

Code for the NeurIPS 2026 paper by Adam Shai, Thomas Elliott, and Paul Riechers
([arXiv:2507.07432](https://arxiv.org/abs/2507.07432)).

We train small transformers, LSTMs, GRUs and vanilla RNNs on next-token prediction for
stochastic processes whose minimal generators are classical (Mess3), quantum (Bloch Walk,
FRDN, qRRXOR) or post-quantum (Moon). We then test whether an affine probe from activations
recovers the predictive-state geometry of the minimal generator, compare it against
finite-order classical baselines and an NTP-preserving shuffle, and run causal interventions
on the qRRXOR models.

This repository contains the process definitions, training code and configs, the probing and
scoring scripts behind every figure and number in the paper, and the figure scripts. Trained
checkpoints and pre-computed analysis outputs are in two Hugging Face datasets (see
[Data](#data)).

## Install

Python 3.11 or 3.12. Dependencies are pinned in `uv.lock`.

```bash
# with uv (recommended; reproduces the exact environment)
pip install uv
uv sync --all-extras

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command below **from the repository root with `PYTHONPATH=.`**, e.g.
`PYTHONPATH=. uv run python <script> ...`. Scripts read and write repo-relative paths
(`results/`, `figures/`, `shuffle_results_*`).

## Data

| Dataset | Contents | Used by |
|---|---|---|
| [`SimplexAI/quantum-representations`](https://huggingface.co/datasets/SimplexAI/quantum-representations) (already public) | The original single-seed (seed 42) runs: 16 models (4 architectures x 4 processes), 201 checkpoints each, plus the pre-computed regression analysis | `Fig2.py`, `Fig3.py`, `Fig4.py`, `FigAppendix.py` (downloaded automatically with `--data-source huggingface`) |
| `HF_DATASET_URL_TODO` | The 5-seed grid, the qRRXOR models, and pre-computed CSVs (below) | everything under `scripts/` |

Download the second dataset into the repository root so that its `results/` directory lands
at `./results/` (the `--include` filters keep the dataset's own README from overwriting this
one):

```bash
huggingface-cli download HF_DATASET_REPO_TODO --repo-type dataset --local-dir . \
    --include "results/*" "shuffle_results_quantum_rrxor_hero_3seed/*" "figures/*"
```

Add `--include "results/multiseed_mess3_transformer/*"` style filters instead to fetch one
sweep at a time; the full dataset is about 11 GB.

Run directories (`results/<sweep>/run_*/`) hold one checkpoint every 100 epochs
(`<tokens_seen>.pt`; `0.pt` is the untrained network), `loss.csv`, and the run config.

## Reproducing the paper

Runtimes are wall-clock, measured on one NVIDIA H100/H200 GPU unless noted. "Not timed"
means we did not record it. Scripts that score many runs read every converged run in the
directories they name. A run counts as converged when the last `val_loss_mean` in its
`loss.csv` (already normalized by the optimal loss) is below 1.005.

### Main text

| Paper item | Script | Command | Runtime |
|---|---|---|---|
| Fig. 2, geometry panels (seed 42) | `Fig2.py` | `python Fig2.py --data-source huggingface` | not timed (CPU; download-bound) |
| Fig. 2, RMSE bars (5 seeds, 95% t-intervals); "12-37x below random init" | `scripts/paper_metrics/fig2_metrics_multiseed.py` then `scripts/paper_figures/fig2_barcol.py` | `python scripts/paper_metrics/fig2_metrics_multiseed.py --device cuda:0` then `python scripts/paper_figures/fig2_barcol.py results/paper_metrics/fig2_metrics_multiseed.csv Figs/Fig2.png fig2_mainresult_5seed.png` | 3.1 h for 60 runs (about 3 min/run) |
| Fig. 3, geometry over training | `Fig3.py` | `python Fig3.py --data-source huggingface` | not timed |
| Fig. 4, qRRXOR NTP-preserving shuffle (LSTM seed 456, transformer seed 123) | `FigBestLSTM_pyvista.py` | `python FigBestLSTM_pyvista.py --variant 6panel` | not timed (CPU; PyVista off-screen) |
| §6, 5-seed numbers (R² 0.85 / 0.99 / 0.99 / 0.70, 200-draw shuffle null, p < 0.005, 65% vs 84-99%) | `scripts/activation_analysis/qrrxor_shuffle_null_analysis.py` | `python scripts/activation_analysis/qrrxor_shuffle_null_analysis.py --device cuda:0` | not timed |
| §5 Interventions, main-text summary | see App. Interventions below | | |

Fig. 4 also reads two small cached files, which are in the dataset. To regenerate them:

```bash
python scripts/activation_analysis/run_shuffle_analysis.py --results_dir results/20260424220339 --outdir shuffle_results_quantum_rrxor_hero_3seed
python scripts/activation_analysis/compute_lstm_curves.py --sweep_dirs results/20260504050111_lstm_b64_full --out figures/beyond_ntp/lstm_curves_b64_full.joblib
```

The published Fig. 4 is this script's output with panel titles restyled by hand.
`FigBestLSTM_pyvista.py` needs an off-screen OpenGL backend for PyVista (OSMesa/EGL, or
`xvfb-run`).

### Appendix

| Paper item | Script | Command | Runtime |
|---|---|---|---|
| Pairwise geometry preservation (RSA figure) | `Fig4.py` | `python Fig4.py --data-source huggingface` | not timed |
| Extended architecture figures (one per architecture): geometry panels (seed 42) | `FigAppendix.py` | `python FigAppendix.py --data-source huggingface --model-type all` | not timed |
| Extended architecture figures: 5-seed bars (FRDN row uses in-sample RMSE) | `scripts/paper_metrics/fig2_metrics_multiseed.py`, `scripts/paper_metrics/frdn_metrics_multiseed.py`, then `scripts/paper_figures/fig_appendix_bars.py` | `python scripts/paper_metrics/frdn_metrics_multiseed.py --device cuda:0` then `python scripts/paper_figures/fig_appendix_bars.py results/paper_metrics/fig2_metrics_multiseed.csv,results/paper_metrics/frdn_metrics_multiseed.csv <FigAppendix output>.pdf transformer <out>.pdf` (ARCHKEY: `transformer`, `LSTM`, `GRU`, `vanilla_RNN`) | FRDN: not timed |
| Higher-order classical baselines: figure, table, "order 4: R² 0.27-0.35" | `scripts/activation_analysis/orderk_sweep.py` then `scripts/paper_figures/fig_orderk.py` | `python scripts/activation_analysis/orderk_sweep.py --device cuda:0` then `python scripts/paper_figures/fig_orderk.py results/orderk_sweep_agg.csv orderk_baselines.pdf` | 27 min for 60 runs |
| Cross-entropy of order-k predictors vs the entropy rate (exact, no models) | `scripts/paper_metrics/orderk_loss.py` | `python scripts/paper_metrics/orderk_loss.py` | seconds (CPU) |
| qRRXOR hyperparameter selection on dev seeds 7/11/13 (6k-epoch screen) | `scripts/activation_analysis/hpsearch_qrrxor_score.py` | `python scripts/activation_analysis/hpsearch_qrrxor_score.py --device cuda:0` (transformer + LSTM); `... --dirs results/hpsearch_qrrxor_grurnn --out results/hpsearch_qrrxor_grurnn_scores.csv` (GRU + RNN) | not timed |
| qRRXOR convergence of the 20 report runs; comparison with the lr=1e-4 / batch-16 setting | `scripts/activation_analysis/qrrxor_robust_score.py`, `qrrxor_robust_grurnn_score.py` | `python scripts/activation_analysis/qrrxor_robust_score.py --device cuda:0` then `python scripts/activation_analysis/qrrxor_robust_grurnn_score.py --device cuda:0` | not timed |
| Interventions, recurrent patching (96.0 / 95.2 / 92.7%; invisible-only writes 89.3 / 83.4 / 59.5%; matched random 7-13%) | `scripts/activation_analysis/causal_ntp_neutral.py` | `python scripts/activation_analysis/causal_ntp_neutral.py --run_dir results/qrrxor_robust_{gru,lstm,rnn}/run_<i>_... --device cuda:0` (once per run) | about 30 s/run |
| Interventions, scored against the donor's future (98.6-99.1%) | `scripts/activation_analysis/causal_verify_target_adjudication.py` | `python scripts/activation_analysis/causal_verify_target_adjudication.py --device cuda:0` | not timed |
| Interventions, untrained GRUs (0.0%; probe CV R² about 0.7) | `scripts/activation_analysis/causal_verify_rnn.py` | `python scripts/activation_analysis/causal_verify_rnn.py --device cuda:0` | not timed |
| Interventions, erasure (recurrent and transformer) | `scripts/activation_analysis/causal_ker_ablation_rnn_audit.py`, `causal_ker_ablation_audit.py`, then `causal_ker_audit_aggregate.py` | `python scripts/activation_analysis/causal_ker_ablation_rnn_audit.py --run_dir results/qrrxor_robust_rnn/run_<i>_... --device cuda:0`; `python scripts/activation_analysis/causal_ker_ablation_audit.py --run_dir results/qrrxor_robust_transformer/run_<i>_... --device cuda:0`; `python scripts/activation_analysis/causal_ker_audit_aggregate.py` | not timed |
| Interventions, transformer visible-component writes (7 / 90 / 96%) | `scripts/activation_analysis/causal_local_readout.py` | `python scripts/activation_analysis/causal_local_readout.py --device cuda:0` | not timed |
| Interventions, transformer routing (89.8% / 18.2%) | `scripts/activation_analysis/causal_xf_channels.py` | `python scripts/activation_analysis/causal_xf_channels.py --run_dir results/qrrxor_robust_transformer/run_<i>_... --device cuda:0` (once per run) | not timed |
| Interventions, random readout-orthogonal edits (0.00007-0.0002 nats) | `scripts/activation_analysis/causal_verify_orthonoise.py` | `python scripts/activation_analysis/causal_verify_orthonoise.py --device cuda:0` | not timed |
| Interventions, "Verification" (second implementation) | `scripts/activation_analysis/causal_verify_xf.py`, `causal_verify_rnn.py`, `causal_verify_xf_profile.py`, `causal_verify_target_adjudication.py` | `python scripts/activation_analysis/causal_verify_xf.py --device cuda:0` (and likewise for the others) | not timed |
| "Represented but not used": 91% / untrained 0.53 / beyond-last-3-tokens 0.83 vs 0.00 | `scripts/activation_analysis/qrrxor_repr_audit.py` | `python scripts/activation_analysis/qrrxor_repr_audit.py --device cuda:0 --parts abc` | not timed |
| "Represented but not used": held-out 3-token ending (GRU 0.91, transformer -0.58) | `scripts/activation_analysis/heldout_suffix_extrapolation.py` | `python scripts/activation_analysis/heldout_suffix_extrapolation.py --device cuda:0` | not timed |
| Top-3 principal components of activations | `scripts/paper_metrics/pca_views.py` | `python scripts/paper_metrics/pca_views.py --device cuda:0` | not timed |
| Process definitions | `epsilon_transformers/process/transition_matrices.py` (`get_matrix_from_args`), `GHMM.py` | | |

`scripts/activation_analysis/run_regression_analysis.py` is the original analysis pipeline
whose outputs are in `SimplexAI/quantum-representations`. It reads checkpoints from S3 and
needs AWS credentials in a `.env` file (see `epsilon_transformers/analysis/load_data.py` for
the variable names). You do not need it to reproduce the figures.

## Training

All models are trained on next-token cross-entropy only. Configs live in
`scripts/experiment_config_*.yaml`; each launches a sweep into `results/<sweep_id>/`.
Transformers use `scripts/launcher_cuda_parallel.py`, recurrent models
`scripts/launcher_cuda_parallel_rnn.py` (the two submission-era qRRXOR LSTM sweeps use
`scripts/launcher_rnn_parallel.py`). The launchers call `python`, so run them inside the
environment (`uv run` or an activated venv).

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. uv run python scripts/launcher_cuda_parallel.py \
    --config scripts/experiment_config_transformer_bloch_multiseed.yaml --runs_per_gpu 2
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run python scripts/launcher_cuda_parallel_rnn.py \
    --config scripts/experiment_config_rnn_bloch_multiseed.yaml --runs_per_gpu 3
```

| Config(s) | What | Output | Runtime per run |
|---|---|---|---|
| `experiment_config_transformer_{mess3_bloch,moon,frdn}.yaml`, `experiment_config_rnn.yaml` | Original single-seed (42) grid; these produced the `SimplexAI/quantum-representations` runs | `results/<timestamp>/` | about 1 GPU-hour |
| `experiment_config_transformer_{bloch,mess3,moon,frdn}_multiseed.yaml` | 5-seed grid (seeds 0, 1, 2, 3, 42), transformer | `results/multiseed_<process>_transformer/` | Bloch/Mess3 about 8 h (200 x 128 batches, `torch.compile`); Moon/FRDN about 0.8 h |
| `experiment_config_rnn_{bloch,stage2,frdn}_multiseed.yaml` | 5-seed grid, LSTM / GRU / vanilla RNN (`stage2` = Mess3 + Moon) | `results/multiseed_{bloch,stage2,frdn}_rnn/` | about 2.5-3.5 h |
| `experiment_config_transformer_quantum_rrxor_hpsearch.yaml`, `experiment_config_rnn_quantum_rrxor_hpsearch{,_grurnn}.yaml` | qRRXOR learning-rate x batch-size screen on dev seeds 7, 11, 13 (6k epochs) | `results/hpsearch_qrrxor_*` | about 0.1-0.2 h |
| `experiment_config_transformer_quantum_rrxor_robust.yaml`, `experiment_config_rnn_quantum_rrxor_robust{,_gru,_rnn}.yaml` | qRRXOR report runs (seeds 42, 123, 456, 789, 1000) at the selected settings: lr 3e-4 / batch 64 (transformer, LSTM, GRU), lr 3e-5 / batch 64 (vanilla RNN) | `results/qrrxor_robust_*` | 0.2-0.8 h |
| `experiment_config_transformer_quantum_rrxor_submission.yaml`, `experiment_config_rnn_quantum_rrxor_submission_lstm.yaml` | qRRXOR models shown in Fig. 4 | `results/20260424220339/`, `results/20260504050111_lstm_b64_full/` | about 1.1-1.3 h |
| `experiment_config_rnn_quantum_rrxor_fragile_lstm.yaml` | qRRXOR LSTM at lr 1e-4 / batch 16 (some seeds collapse to uniform output) | `results/20260503215800_lstm_default_cpu/` | not timed |

Notes:

- Put `seed` only under `sweep_config.model_config`. The launchers merge the base
  `model_config` into each sweep combination, so a base `seed` overrides the swept one.
- The multi-seed transformer configs set `compile: true` (`torch.compile`). Checkpoints are
  saved from the uncompiled module, so their state-dict keys are the usual ones.
- W&B logging is off in every config (`wandb: false`). Metrics always go to `loss.csv` and
  `log.json` in the run directory.

## Repository layout

```
epsilon_transformers/      process definitions (process/), models and training utilities (training/)
scripts/
  train.py, train_rnn.py   single-run training
  launcher_*.py            sweep launchers
  experiment_config_*.yaml training configs
  activation_analysis/     probing, scoring, shuffle and intervention analyses
  paper_metrics/           Fig. 2 / appendix bar metrics, order-k loss, PCA views
  paper_figures/           bar overlays for Fig. 2 and the appendix figures, order-k figure
  data_manager.py          Hugging Face download helper for Fig*.py
Fig2.py Fig3.py Fig4.py FigAppendix.py FigBestLSTM_pyvista.py   figure scripts
```

`scripts/activation_analysis/causal_belief_transplant.py` and `causal_phase0.py` are
libraries for the intervention scripts. Their own `main()` functions are earlier designs whose
numbers the paper does not quote.

## Citation

See `CITATION.cff`.

```bibtex
@inproceedings{shai2026arranges,
  title     = {What Arranges Features in Activation Space? Non-Classical Predictive Geometry in Next-Token Predictors},
  author    = {Shai, Adam and Elliott, Thomas and Riechers, Paul},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {TODO: confirm author list and add volume/pages once published}
}
```

## License

TODO: see `LICENSE_TODO.md`.
