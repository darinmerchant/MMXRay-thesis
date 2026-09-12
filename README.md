# MMXRay

Code for the master thesis *Self-Supervised Contrastive Learning on Physics-Informed Generated Spectral Data for Nanoparticle Characterization* (Darin Merchant, ETH Zurich, 2026; supervised by Prof. Rafael Gómez-Bombarelli, MIT, and Prof. Juan Carrasquilla Álvarez, ETH Zurich).

A 1D convolutional encoder is pretrained with VICReg on simulated pair distribution functions (PDF) or powder X-ray diffraction patterns (XRD) of Materials Project structures. The two views of each structure are the same structure simulated under two random draws of the instrument parameters, so the encoder learns a representation that is insensitive to the instrument. The representation is then read out on nanoparticle datasets (CHILI-3K, CHILI-100K) and, exploratorily, on measured mineral patterns (RRUFF), with a frozen linear probe, with fine-tuning, and against a supervised baseline trained from scratch.

This repository is the subset of the working codebase that the thesis rests on: the pipeline, the configurations of the runs the thesis reports, and one script per figure or table. Hyperparameter sweeps, abandoned experiments, tests, and the thesis source are not included.

The code is released under the MIT license (see `LICENSE`).

## What is and is not here

Included:

- the simulators, datasets, encoders, losses, and the pretraining loop
- the builders that turn the raw downloads into the registries the code reads
- the downstream readouts (linear probe, fine-tuning, cross-validated variants)
- the representation analyses behind the tables
- the plotting scripts behind the figures
- the three pretraining configurations the thesis reports

Not included, and not distributed with this repository:

- raw data (see *Data* for where each dataset comes from)
- banked simulated views and registries; each is rebuilt by a command below
- checkpoints other than the two encoders the thesis reports, which are released under `checkpoints/`
- run outputs; every command writes under `runs/`, which is ignored by git

Scripts were copied verbatim from the working repository. Their docstrings sometimes cite files that live only there (`RESULTS.md`, `PROGRESS.md`, `docs/TRAPS.md`) and session numbers. Those references are the working history of the project and can be ignored here.

## Layout

| Path | What it holds |
|---|---|
| `core/` | registry format, splits, simulate-transform, two-view datasets, encoders, losses, config, training loop |
| `modalities/pdf/`, `modalities/xrd/` | the PDF and XRD simulators (periodic via diffpy or pymatgen, finite particles via the Debye sum) and the fixed signal grids |
| `data/builders/` | one builder per dataset: full Materials Project, CHILI-3K, CHILI-100K, RRUFF |
| `checkpoints/` | the pretrained PDF and XRD encoders the thesis reports, each with its resolved config and per-epoch metrics |
| `configs/pretrain/` | the PDF pretraining config, the XRD pretraining config, and the config the supervised baseline borrows for its architecture |
| `tools/` | banking simulated views, augmenting CHILI-3K, the prediction store for the parity figures, and the figure scripts that read data or registries |
| `analysis/` | the downstream readouts, the representation analyses, and the figure scripts that read trained runs |
| `environment.yml` | the pinned conda environment |

Every script is run from the repository root as a module, for example `python -m core.train --config ...`. Each one has a docstring at the top that states what it does and what it reads and writes.

## Environment

```bash
conda env create -f environment.yml
conda activate mmxray
pip install --no-deps debyecalculator==1.0.14
```

The environment is CPU torch, which is enough for simulation, banking, the readouts, and every figure. Pretraining and the fine-tuning grids were run on GPUs with a CUDA build of torch 2.5 and are impractical on CPU at the reported budgets (300 epochs over 4.3 million banked views).

The one fragile dependency is diffpy, which has a compiled backend. The pins in `environment.yml` (numpy 1.26, diffpy.srreal 1.4, libdiffpy 1.4.1, scikit-learn 1.5) keep it working; numpy 2 or a newer scikit-learn silently breaks it.

Anything that calls scikit-learn from several processes at once needs the BLAS thread caps, or it deadlocks:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Data

Raw sources, staged by hand under `data/raw/`:

| Dataset | Source | Staged at |
|---|---|---|
| Materials Project, 2019.04.01 dump (133,420 structures) | HuggingFace `materials-toolkits/materials-project`, the HDF5 file | `data/raw/mp2019/data.hdf5` |
| CHILI-3K | the processed `.pt` files of the CHILI release | `data/raw/chili/processed/` |
| CHILI-100K | the per-structure `.h5` files of the CHILI release | `data/raw/chili100k/cod_output_080124/` |
| RRUFF | the 148 measured mineral patterns with their CIFs | `data/raw/rruff/` |

Registries are parquet tables with one row per material, built once:

```bash
python -m data.builders.mp_full                      # -> data/registry/material_registry_mpfull.parquet
python -m data.builders.chili                        # -> data/downstream/chili/chili_registry.parquet
python -m data.builders.chili100k                    # -> data/downstream/chili100k/chili100k_registry.parquet
python -m data.builders.chili100k --fractions 0.8 0.2 0.0 \
    --out data/downstream/chili100k/chili100k_registry_val20.parquet   # the fit set used for the material shift
python -m data.builders.rruff                        # -> data/downstream/rruff/rruff_registry.parquet
```

Pretraining reads pre-simulated views rather than simulating on the fly. Banking 32 views per structure of the full Materials Project registry takes on the order of a thousand CPU hours per modality and was sharded over a cluster; the block directory name embeds a hash of the banking parameters, and the pretraining configs reference the block by that name.

```bash
python -m tools.bank_pdf --registry data/registry/material_registry_mpfull.parquet --modality pdf \
    --out data/banked --n-views 32 --shards 64 --shard 0 --workers 32      # one of 64 shards
python -m tools.bank_pdf --registry data/registry/material_registry_mpfull.parquet --modality xrd \
    --out data/banked --n-views 32 --shards 64 --shard 0 --workers 32
```

The instrument-shift condition on CHILI-3K is a second signal channel: every particle re-simulated once with the Debye sum at a random draw of the pretraining instrument ranges. It is added to the CHILI-3K registry in place:

```bash
python -m tools.augment_chili --registry data/downstream/chili/chili_registry.parquet \
    --raw-dir data/raw/chili100k/cod_output_080124 --merge
```

## Pretraining

One config is one experiment. Each run writes a self-contained directory named `{date}_{time}_{config stem}_{git sha}` under the config's output directory, holding the resolved config, one metrics line per epoch, and the best and last checkpoints.

```bash
python -m core.train --config configs/pretrain/cnn_vicreg_mpfull_final.yaml            # PDF, -> runs/pdf/sweep/
python -m core.train --config configs/pretrain/xrd_mpfull/cnn_vicreg_cov50_mpfull.yaml # XRD, -> runs/xrd/pretrain_mpfull/
python -m core.train --resume runs/pdf/sweep/<run dir>                                 # continue an interrupted run in place
```

The thesis reports one PDF encoder and one XRD encoder, and both are released in this repository, so pretraining does not have to be repeated to reproduce anything downstream.

## Pretrained weights

`checkpoints/` holds the two encoders, each in a directory named after its original run, with the weights, the resolved training config, and one metrics line per epoch:

| Directory | Encoder | Trained on |
|---|---|---|
| `2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372` | PDF, CNN + VICReg, 1.56 M parameters | 32 banked PDF views of 133,420 Materials Project structures, 300 epochs |
| `2026-08-12_165057_cnn_vicreg_cov50_mpfull_be24b0e` | XRD, CNN + VICReg, 1.56 M parameters | 32 banked XRD views of the same structures, 300 epochs |

`ckpt_best.pt` is a dictionary with `model_state` (encoder and projection head), `config` (the resolved training config), `epoch`, and the final loss terms. The encoder expects a PDF as 5,000 points of G(r) on the grid in `modalities/pdf/simulate.py`, or an XRD pattern as 4,999 points on the grid in `modalities/xrd/grid.py`, normalized as `core/transforms.py` does. The representation is the 256-dimensional output of `model.encode`, before the projection head.

```python
from analysis.downstream_eval import load_encoder, embed
model, cfg, epoch = load_encoder("checkpoints/2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372/ckpt_best.pt", device="cpu")
h = embed(model, X, "cpu")   # X: float tensor of shape (N, 5000) -> h: (N, 256)
```

The downstream and figure scripts look for the runs under `runs/`, at the paths the thesis used. Two symlinks make the released encoders visible there without editing any script:

```bash
mkdir -p runs/pdf/sweep runs/xrd/pretrain_mpfull
ln -s ../../../checkpoints/2026-07-30_152128_cnn_vicreg_mpfull_final_5bfa372 runs/pdf/sweep/
ln -s ../../../checkpoints/2026-08-12_165057_cnn_vicreg_cov50_mpfull_be24b0e runs/xrd/pretrain_mpfull/
```

Readouts then write their JSON into the checkpoint directories through the links, which is where the figure scripts read them. A fresh pretraining run gets a new directory name of its own; to use it with the figure scripts, edit the run-name constant in the scripts that carry it.

## Downstream runs behind the results

All readouts take the pretrained run directory through `--runs <tree> --filter <glob>` and write JSON next to the checkpoint or under `--out-root`. The supervised baseline is the same fine-tuning loop started from random weights, which is why it takes a pretraining config (`--config`) for the architecture only.

Instrument shift on CHILI-3K (four training schemes, fit on the clean channel, evaluated on the clean and augmented channels):

```bash
# frozen linear probe (closed-form ridge head)
python -m analysis.downstream_eval --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' --ridge
# fine-tuned encoder, three seeds
python -m analysis.finetune --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' \
    --lr-enc 1e-5 --lr-head 1e-4 --epochs 300 --seed 42 --out-root runs/pdf/datafrac_full/seed42
# supervised baseline: learning rate chosen on the validation split over 3e-5 .. 3e-3, then three seeds
python -m analysis.finetune --config configs/pretrain/cnn_infonce_mpfull_final.yaml \
    --lr-enc 1e-3 --lr-head 1e-3 --epochs 300 --seed 42
```

Label ladder (the same three schemes at a fraction of the labeled rows; fractions 0.0277, 0.05, 0.1, 0.25, 0.5, 1.0, three seeds each; the probe additionally at 0.005, 0.01, 0.02):

```bash
python -m analysis.finetune --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' \
    --lr-enc 1e-5 --lr-head 1e-4 --epochs 300 --n-train-frac 0.1 --seed 42
python -m analysis.finetune --config configs/pretrain/cnn_infonce_mpfull_final.yaml \
    --lr-enc 1e-3 --lr-head 1e-3 --epochs 300 --n-train-frac 0.1 --seed 42
python -m analysis.downstream_eval --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' --ridge \
    --n-train-frac 0.1 --subsample-seed 42 --out-name probe_frac0.1_seed42.json
```

Material shift (fit on CHILI-100K metal oxides, evaluate on the CHILI-3K test rows, matched fit-set size; `--fit-signal xpdf_aug` adds the instrument shift on top):

```bash
# baseline learning-rate search, one run per learning rate in 3e-5 .. 3e-3, read by analysis/chem_lr_select.py
python -m analysis.finetune --config configs/pretrain/cnn_infonce_mpfull_final.yaml \
    --lr-enc 1e-3 --lr-head 1e-3 --epochs 300 --seed 42 \
    --registry data/downstream/chili/chili_registry.parquet \
    --fit-registry data/downstream/chili100k/chili100k_registry_val20.parquet \
    --target target_np_size target_mo_bond target_mean_bond --n-train-frac <eval train rows / fit train rows> \
    --out-root runs/pdf/chem_lr_scratch
# fine-tuned and baseline at the selected rates, seeds 42-44, -> runs/pdf/chem_seeds/{ft,scratch}[_aug]
# probe: analysis.downstream_eval with the same --registry / --fit-registry / --target flags
```

Per-row predictions for the parity and confusion figures (`runs/pdf/parity/`):

```bash
python -m analysis.finetune --config configs/pretrain/cnn_vicreg_mpfull_final.yaml --lr-enc 1e-3 --lr-head 1e-3 \
    --seed 42 --epochs 300 --target target_np_size target_mo_bond target_oxidation target_cn --out-root runs/pdf/parity/baseline
python -m analysis.finetune --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' --lr-enc 1e-5 --lr-head 1e-4 \
    --seed 42 --epochs 300 --target target_np_size target_mo_bond target_oxidation target_cn --out-root runs/pdf/parity/finetuned
python -m analysis.downstream_eval --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' --ridge --out-name probe_chili_parity.json
```

Encoders saved with weights for the embedding figure (one fine-tuned and one supervised encoder per target, `--save-weights`, into `runs/pdf/embed_fig/<target>/`), and for the augmentation-PCA figure (one fine-tuned encoder per instrument parameter on a 3,000-structure Materials Project subset):

```bash
python -m tools.derive_mpfull_symmetry            # crystal-system labels for the full-MP registry
python -m tools.carve_mp_aug3k                    # -> data/downstream/mp_aug3k/mp_aug3k.parquet
python -m analysis.finetune --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' \
    --registry data/downstream/mp_aug3k/mp_aug3k.parquet --signal mpaug --target aug_qmin \
    --lr-enc 1e-5 --lr-head 1e-4 --epochs 300 --seed 42 --out-root runs/pdf/aug_pca/aug_qmin --save-weights
```

Exploratory RRUFF evaluation (crystal system from measured XRD, five-fold cross-validation repeated three times, fit rows matched across schemes):

```bash
python -m analysis.probe_cv --registry data/downstream/rruff/rruff_registry.parquet --signal xrd \
    --target target_crystal_system --classify --group-col material_id --folds 5 --repeats 3 \
    --runs runs/xrd/pretrain_mpfull --filter '*cnn_vicreg_cov50_mpfull*' --match-gradient-rows \
    --out analysis/out/rruff_probe_cv_xrd_crystal_matched_preds.json
python -m analysis.finetune_cv --registry data/downstream/rruff/rruff_registry.parquet --signal xrd \
    --ckpt runs/xrd/pretrain_mpfull/<run dir>/ckpt_best.pt --target target_crystal_system \
    --group-col material_id --folds 5 --repeats 3 --arms finetuned --warm-start-head --seed 42 \
    --batch-size 16 --epochs 100 --patience 15 --lr-enc 0 1e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 --lr-head 1e-4 \
    --out runs/xrd/rruff_cv/ft_cnn_vicreg_cov50_mpfull_crystal_system.json
python -m analysis.finetune_cv --registry data/downstream/rruff/rruff_registry.parquet --signal xrd \
    --ckpt runs/xrd/pretrain_mpfull/<run dir>/ckpt_best.pt --target target_crystal_system \
    --group-col material_id --folds 5 --repeats 3 --arms supervised --seed 42 \
    --batch-size 16 --epochs 100 --patience 15 --sup-lr 1e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 \
    --out runs/xrd/rruff_cv/sup_cnn_crystal_system.json
```

## Representation analyses behind the tables

Measured on banked pretraining views of the encoder's own corpus, or on CHILI-3K. `<pdf block>` and `<xrd block>` are the block directories under `data/banked/`.

```bash
# geometry and view separation (participation ratio, leading-direction share, alignment, uniformity)
python -m analysis.latent --block <pdf block> --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*'
# linear decodability of each instrument parameter, pooled out-of-fold R^2
python -m analysis.theta_probe --block <pdf block> --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*' --out analysis/out/theta_probe.json
python -m analysis.theta_probe --block <xrd block> --runs runs/xrd/pretrain_mpfull --filter '*cnn_vicreg_cov50_mpfull*' --out analysis/out/theta_probe_xrd_inherited.json
# the same two on CHILI-3K particles, clean view against augmented view
python -m analysis.chili_theta_probe
python -m analysis.chili_invariance_hist --paper
# displacement of the representation along one instrument parameter at a time, inside and outside the pretraining ranges
python -m tools.simulate_ladder --n-materials 256                 # -> runs/pdf/latent/invariance/
python -m analysis.invariance_decay --runs runs/pdf/sweep --filter '*cnn_vicreg_mpfull_final*'
```

## Figures

Figure scripts save under `figures/` (or `figures/appendix/`), which git ignores. Each is run as `python -m tools.<name>` or `python -m analysis.<name>`; `--paper` selects the thesis sizing where the script has both.

| Thesis figure | Script | Reads |
|---|---|---|
| PDF and XRD of a particle against its size (introduction) | `tools/plot_xrd_pdf_vs_size.py` | CHILI-100K raw files |
| Effect of each PDF augmentation (theory) | `tools/plot_appendix_aug_effect.py` | the full-MP registry |
| View-separation histograms, PDF pretraining corpus | `analysis/plot_pdf_invariance.py` | the PDF run and block |
| Invariance decay along each parameter | `analysis/plot_invariance_decay.py` | the run's `invariance_decay.json`, written by `analysis/invariance_decay.py` |
| View-separation histograms, CHILI-3K | `analysis/chili_invariance_hist.py --paper` | the CHILI-3K registry with the augmented channel |
| Embedding PCA on CHILI-3K | `tools/plot_appendix_embedding.py --columns <subset>` | `runs/pdf/embed_fig/` |
| Instrument shift, four schemes | `tools/plot_chili_four_arms.py --paper` | `runs/pdf/sweep`, `runs/pdf/supervised`, `runs/pdf/datafrac_full` |
| Label ladder | `tools/plot_probe_datafrac.py --paper` | `runs/pdf/datafrac`, `runs/pdf/datafrac_full`, `runs/pdf/supervised` |
| Material shift interaction | `tools/plot_chem_shift_interaction.py --paper` | `runs/pdf/chem_seeds`, `runs/pdf/chem_lr_scratch*` |
| View-separation histograms, XRD | `analysis/plot_invariance_hist.py --run <xrd run> --block <xrd block> --paper` | the XRD run and block |
| Pretraining curves (appendix) | `tools/plot_appendix_training.py` | the PDF run's `metrics.jsonl` |
| Label distributions, CHILI-3K (appendix) | `tools/plot_appendix_labels.py` | the CHILI-3K registry |
| Material coverage of CHILI-100K (appendix) | `tools/plot_appendix_chem_shift.py` | both CHILI registries |
| Augmentation PCA (appendix) | `tools/plot_appendix_aug_pca.py` | `runs/pdf/aug_pca/` |
| Label distributions, RRUFF (appendix) | `tools/plot_appendix_rruff_labels.py` | the RRUFF registry |
| RRUFF confusion matrices (appendix) | `tools/plot_rruff_confusion.py` | `runs/xrd/rruff_cv/`, the probe JSON above |
| CHILI-3K parity and confusion (appendix) | `tools/plot_chili_parity.py`, `tools/plot_chili_confusion.py` | `runs/pdf/parity/` |

The label-ladder and material-shift scripts also emit their appendix tables with `--table` (and `--table --sd` for the standard deviations). The overview figure, the task-specific schematic, and the scattering-branches figure were drawn by hand and have no script.
