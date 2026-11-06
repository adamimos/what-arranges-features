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
| [`SimplexAI/quantum-representations`](https://huggingface.co/datasets/SimplexAI/quantum-representations) (already public) | The original single-seed (seed 42) runs: 16 models (4 architectures x 4 processes), 201 checkpoints each, plus the pre-computed regression analysis | `paper_figures/fig2_geometry.py`, `paper_figures/fig3_training.py`, `paper_figures/appendix_rsa.py`, `paper_figures/appendix_architectures.py` (downloaded automatically with `--data-source huggingface`) |
| [`SimplexAI/what-arranges-features`](https://huggingface.co/datasets/SimplexAI/what-arranges-features) | The 5-seed grid, the qRRXOR models, and pre-computed CSVs (below) | everything under `experiments/` |

Download the second dataset into the repository root so that its `results/` directory lands
at `./results/` (the `--include` filters keep the dataset's own README from overwriting this
one):

```bash
huggingface-cli download SimplexAI/what-arranges-features --repo-type dataset --local-dir . \
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
| Fig. 2, geometry panels (seed 42) | `paper_figures/fig2_geometry.py` | `python paper_figures/fig2_geometry.py --data-source huggingface` | not timed (CPU; download-bound) |
| Fig. 2, RMSE bars (5 seeds, 95% t-intervals); "12-37x below random init" | `experiments/main_grid/fig2_rmse_5seed.py` then `paper_figures/fig2_rmse_bars.py` | `python experiments/main_grid/fig2_rmse_5seed.py --device cuda:0` then `python paper_figures/fig2_rmse_bars.py results/paper_metrics/fig2_metrics_multiseed.csv Figs/Fig2.png fig2_mainresult_5seed.png` | 3.1 h for 60 runs (about 3 min/run) |
| Fig. 3, geometry over training | `paper_figures/fig3_training.py` | `python paper_figures/fig3_training.py --data-source huggingface` | not timed |
| Fig. 4, qRRXOR NTP-preserving shuffle (LSTM seed 456, transformer seed 123) | `paper_figures/fig4_qrrxor_shuffle.py` | `python paper_figures/fig4_qrrxor_shuffle.py --variant 6panel` | not timed (CPU; PyVista off-screen) |
| §6, 5-seed numbers (R² 0.85 / 0.99 / 0.99 / 0.70, 200-draw shuffle null, p < 0.005, 65% vs 84-99%) | `experiments/shuffle/shuffle_null_5seed.py` | `python experiments/shuffle/shuffle_null_5seed.py --device cuda:0` | not timed |
| §5 Interventions, main-text summary | see App. Interventions below | | |

Fig. 4 also reads two small cached files, which are in the dataset. To regenerate them:

```bash
python experiments/shuffle/shuffle_curves.py --results_dir results/20260424220339 --outdir shuffle_results_quantum_rrxor_hero_3seed
python experiments/shuffle/shuffle_curves_lstm.py --sweep_dirs results/20260504050111_lstm_b64_full --out figures/beyond_ntp/lstm_curves_b64_full.joblib
```

The published Fig. 4 is this script's output with panel titles restyled by hand.
`paper_figures/fig4_qrrxor_shuffle.py` needs an off-screen OpenGL backend for PyVista (OSMesa/EGL, or
`xvfb-run`).

### Appendix

| Paper item | Script | Command | Runtime |
|---|---|---|---|
| Pairwise geometry preservation (RSA figure) | `paper_figures/appendix_rsa.py` | `python paper_figures/appendix_rsa.py --data-source huggingface` | not timed |
| Extended architecture figures (one per architecture): geometry panels (seed 42) | `paper_figures/appendix_architectures.py` | `python paper_figures/appendix_architectures.py --data-source huggingface --model-type all` | not timed |
| Extended architecture figures: 5-seed bars (FRDN row uses in-sample RMSE) | `experiments/main_grid/fig2_rmse_5seed.py`, `experiments/main_grid/frdn_rmse_5seed.py`, then `paper_figures/appendix_architecture_bars.py` | `python experiments/main_grid/frdn_rmse_5seed.py --device cuda:0` then `python paper_figures/appendix_architecture_bars.py results/paper_metrics/fig2_metrics_multiseed.csv,results/paper_metrics/frdn_metrics_multiseed.csv <appendix_architectures.py output>.pdf transformer <out>.pdf` (ARCHKEY: `transformer`, `LSTM`, `GRU`, `vanilla_RNN`) | FRDN: not timed |
| Higher-order classical baselines: figure, table, "order 4: R² 0.27-0.35" | `experiments/orderk/orderk_probe_sweep.py` then `paper_figures/appendix_orderk.py` | `python experiments/orderk/orderk_probe_sweep.py --device cuda:0` then `python paper_figures/appendix_orderk.py results/orderk_sweep_agg.csv orderk_baselines.pdf` | 27 min for 60 runs |
| Cross-entropy of order-k predictors vs the entropy rate (exact, no models) | `experiments/orderk/orderk_cross_entropy.py` | `python experiments/orderk/orderk_cross_entropy.py` | seconds (CPU) |
| qRRXOR hyperparameter selection on dev seeds 7/11/13 (6k-epoch screen) | `experiments/qrrxor_hparams/score_hpsearch.py` | `python experiments/qrrxor_hparams/score_hpsearch.py --device cuda:0` (transformer + LSTM); `... --dirs results/hpsearch_qrrxor_grurnn --out results/hpsearch_qrrxor_grurnn_scores.csv` (GRU + RNN) | not timed |
| qRRXOR convergence of the 20 report runs; comparison with the lr=1e-4 / batch-16 setting | `experiments/qrrxor_hparams/score_report_runs.py`, `experiments/qrrxor_hparams/score_report_runs_gru_rnn.py` | `python experiments/qrrxor_hparams/score_report_runs.py --device cuda:0` then `python experiments/qrrxor_hparams/score_report_runs_gru_rnn.py --device cuda:0` | not timed |
| Interventions, recurrent patching (96.0 / 95.2 / 92.7%; invisible-only writes 89.3 / 83.4 / 59.5%; matched random 7-13%) | `experiments/interventions/recurrent_patching.py` | `python experiments/interventions/recurrent_patching.py --run_dir results/qrrxor_robust_{gru,lstm,rnn}/run_<i>_... --device cuda:0` (once per run) | about 30 s/run |
| Interventions, scored against the donor's future (98.6-99.1%) | `experiments/interventions/verify_donor_future.py` | `python experiments/interventions/verify_donor_future.py --device cuda:0` | not timed |
| Interventions, untrained GRUs (0.0%; probe CV R² about 0.7) | `experiments/interventions/verify_recurrent.py` | `python experiments/interventions/verify_recurrent.py --device cuda:0` | not timed |
| Interventions, erasure (recurrent and transformer) | `experiments/interventions/erasure_recurrent.py`, `experiments/interventions/erasure_transformer.py`, then `experiments/interventions/erasure_aggregate.py` | `python experiments/interventions/erasure_recurrent.py --run_dir results/qrrxor_robust_rnn/run_<i>_... --device cuda:0`; `python experiments/interventions/erasure_transformer.py --run_dir results/qrrxor_robust_transformer/run_<i>_... --device cuda:0`; `python experiments/interventions/erasure_aggregate.py` | not timed |
| Interventions, transformer visible-component writes (7 / 90 / 96%) | `experiments/interventions/transformer_visible_writes.py` | `python experiments/interventions/transformer_visible_writes.py --device cuda:0` | not timed |
| Interventions, transformer routing (89.8% / 18.2%) | `experiments/interventions/transformer_routing.py` | `python experiments/interventions/transformer_routing.py --run_dir results/qrrxor_robust_transformer/run_<i>_... --device cuda:0` (once per run) | not timed |
| Interventions, random readout-orthogonal edits (0.00007-0.0002 nats) | `experiments/interventions/verify_orthogonal_noise.py` | `python experiments/interventions/verify_orthogonal_noise.py --device cuda:0` | not timed |
| Interventions, "Verification" (second implementation) | `experiments/interventions/verify_transformer.py`, `experiments/interventions/verify_recurrent.py`, `experiments/interventions/verify_transformer_profile.py`, `experiments/interventions/verify_donor_future.py` | `python experiments/interventions/verify_transformer.py --device cuda:0` (and likewise for the others) | not timed |
| "Represented but not used": 91% / untrained 0.53 / beyond-last-3-tokens 0.83 vs 0.00 | `experiments/represented_not_used/representation_audit.py` | `python experiments/represented_not_used/representation_audit.py --device cuda:0 --parts abc` | not timed |
| "Represented but not used": held-out 3-token ending (GRU 0.91, transformer -0.58) | `experiments/represented_not_used/heldout_suffix.py` | `python experiments/represented_not_used/heldout_suffix.py --device cuda:0` | not timed |
| Top-3 principal components of activations | `experiments/pca/top3_pcs.py` | `python experiments/pca/top3_pcs.py --device cuda:0` | not timed |
| Process definitions | `epsilon_transformers/process/transition_matrices.py` (`get_matrix_from_args`), `GHMM.py` | | |

`experiments/main_grid/seed42_regression_pipeline.py` is the original analysis pipeline
whose outputs are in `SimplexAI/quantum-representations`. It reads checkpoints from S3 and
needs AWS credentials in a `.env` file (see `epsilon_transformers/analysis/load_data.py` for
the variable names). You do not need it to reproduce the figures.

## Training

All models are trained on next-token cross-entropy only. Configs live in `configs/`; each
launches a sweep into `results/<sweep_id>/`. Transformers use
`training/launch_transformer_sweep.py`, recurrent models `training/launch_rnn_sweep.py`
(the two submission-era qRRXOR LSTM sweeps in `configs/qrrxor/fig4/lstm.yaml` and
`configs/qrrxor/comparison/` use `training/launch_rnn_sweep_seed_named.py`, which names run
directories by seed). The launchers start one `training/train_transformer.py` or
`training/train_rnn.py` process per sweep combination and call `python`, so run them from the
repository root inside the environment (`uv run` or an activated venv).

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. uv run python training/launch_transformer_sweep.py \
    --config configs/multiseed/transformer_bloch.yaml --runs_per_gpu 2
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run python training/launch_rnn_sweep.py \
    --config configs/multiseed/rnn_bloch.yaml --runs_per_gpu 3
```

| Config(s) | What | Output | Runtime per run |
|---|---|---|---|
| `configs/seed42/transformer_{mess3_bloch,moon,frdn}.yaml`, `configs/seed42/rnn.yaml` | Original single-seed (42) grid; these produced the `SimplexAI/quantum-representations` runs | `results/<timestamp>/` | about 1 GPU-hour |
| `configs/multiseed/transformer_{bloch,mess3,moon,frdn}.yaml` | 5-seed grid (seeds 0, 1, 2, 3, 42), transformer | `results/multiseed_<process>_transformer/` | Bloch/Mess3 about 8 h (200 x 128 batches, `torch.compile`); Moon/FRDN about 0.8 h |
| `configs/multiseed/rnn_{bloch,mess3_moon,frdn}.yaml` | 5-seed grid, LSTM / GRU / vanilla RNN | `results/multiseed_{bloch,stage2,frdn}_rnn/` (`stage2` = Mess3 + Moon) | about 2.5-3.5 h |
| `configs/qrrxor/hpsearch/{transformer,lstm,gru_rnn}.yaml` | qRRXOR learning-rate x batch-size screen on dev seeds 7, 11, 13 (6k epochs) | `results/hpsearch_qrrxor_*` | about 0.1-0.2 h |
| `configs/qrrxor/report/{transformer,lstm,gru,rnn}.yaml` | qRRXOR report runs (seeds 42, 123, 456, 789, 1000) at the selected settings: lr 3e-4 / batch 64 (transformer, LSTM, GRU), lr 3e-5 / batch 64 (vanilla RNN) | `results/qrrxor_robust_*` | 0.2-0.8 h |
| `configs/qrrxor/fig4/{transformer,lstm}.yaml` | qRRXOR models shown in Fig. 4 | `results/20260424220339/`, `results/20260504050111_lstm_b64_full/` | about 1.1-1.3 h |
| `configs/qrrxor/comparison/lstm_lr1e-4_b16.yaml` | qRRXOR LSTM at lr 1e-4 / batch 16 (some seeds collapse to uniform output) | `results/20260503215800_lstm_default_cpu/` | not timed |

Notes:

- Put `seed` only under `sweep_config.model_config`. The launchers merge the base
  `model_config` into each sweep combination, so a base `seed` overrides the swept one.
- The multi-seed transformer configs set `compile: true` (`torch.compile`). Checkpoints are
  saved from the uncompiled module, so their state-dict keys are the usual ones.
- W&B logging is off in every config (`wandb: false`). Metrics always go to `loss.csv` and
  `log.json` in the run directory.

## Repository layout

```
configs/                      training configs
  seed42/                     original single-seed grid (the public dataset's runs)
  multiseed/                  5-seed grid
  qrrxor/                     report/, hpsearch/, fig4/, comparison/
training/                     train_transformer.py, train_rnn.py, sweep launchers
epsilon_transformers/
  process/                    process definitions (transition matrices, GHMMs)
  training/                   models, data loaders, logging
  analysis/                   shared analysis code: checkpoint loading, process targets,
                              probes and regression, shuffle control, intervention machinery
experiments/                  analysis entry points, by part of the paper
  main_grid/                  5-seed R² / RMSE scorers (Fig. 2, appendix); seed-42 pipeline
  orderk/                     higher-order classical baselines
  pca/                        top principal components of activations
  qrrxor_hparams/             qRRXOR hyperparameter screen and report-run convergence
  shuffle/                    NTP-preserving shuffle (§6 numbers, Fig. 4 cached curves)
  represented_not_used/       "represented but not used" analyses
  interventions/              patching, erasure, routing and verification experiments
paper_figures/                one script per paper figure, plus data_manager.py (HF download)
```

Scripts in `experiments/` and `paper_figures/` import shared code only from
`epsilon_transformers`, never from each other.

## Citation

See `CITATION.cff`.

```bibtex
@inproceedings{shai2026arranges,
  title     = {What Arranges Features in Activation Space? Non-Classical Predictive Geometry in Next-Token Predictors},
  author    = {Shai, Adam and Elliott, Thomas J. and Riechers, Paul M.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the MIT license (`LICENSE`). On Hugging Face, `SimplexAI/quantum-representations` is released under the MIT license and `SimplexAI/what-arranges-features` under CC-BY-4.0.
