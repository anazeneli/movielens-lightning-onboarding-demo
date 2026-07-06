# MovieLens Two-Tower Recommender

A [two-tower](https://research.google/blog/) recommender on **MovieLens 100K**:
learn an embedding for each user and each movie; their dot product predicts
whether a user will like a movie (implicit feedback — rating ≥ 4 is a positive).
Train it, then serve top-K recommendations.

Uses three Lightning "Lit" products end to end:
[**LitData**](https://github.com/Lightning-AI/litData/tree/main) (streaming
the training data),
[**LitLogger**](https://github.com/Lightning-AI/LitLogger/tree/main)
(experiment tracking + checkpoints),
[**LitServe**](https://github.com/Lightning-AI/litserve/tree/main) (the
inference API).

## Layout

| Folder | What lives here |
|---|---|
| [`recsys/`](recsys/) | **Shared library** — model + data pipeline, imported by both training and serving. |
| [`training/`](training/) | **Training** — one-time data-optimize step, train script, a tensor-metrics variant, and the hyperparameter sweep launcher. |
| [`serving/`](serving/) | **Serving** — the inference API (LitServe), a Streamlit UI, and a standalone demo. |
| `Explainer videos/` | Walkthrough recording. |

The shared code is a package so there's one source of truth for the model and
data loading. It's installed in editable mode (`pip install -e .`, from the
repo root — see `pyproject.toml`), so `from recsys.model import ...` resolves
in `training/` and `serving/` scripts, and in the editor, without any
`sys.path` hacks.

## Data — the shared teamspace drive

The dataset is **not** stored in this studio — it's on a teamspace **Drive**
named `data` (shows up in the Lightning UI, mounts on this studio *and* on
remote job machines). Two copies live there, for two different consumers:

| Path | Format | Used by |
|---|---|---|
| `/teamspace/lightning_storage/data/ml-100k/` | Raw MovieLens files (`u.data`, `u.item`, ...) | `serving/` (needs `u.item` for movie titles), `training/optimize_data.py` (reads `u.data`) |
| `/teamspace/lightning_storage/data/ml-100k-litdata/` | LitData-optimized streaming chunks | `recsys/movielens_datamodule.py` (training) — built once via `python training/optimize_data.py` |

Env vars `MOVIELENS_DATA_DIR` (raw) and `MOVIELENS_LITDATA_DIR` (optimized)
override each path. `prepare_data()` in both the datamodule and `optimize_data.py`
**fails fast** if its input is missing rather than silently downloading —
important since a remote job's local disk isn't where you want a surprise
download landing.

> ⚠️ **LitData here is demonstrational.** MovieLens 100K is 100K rows — it
> fits comfortably in memory, so streaming it isn't necessary; a plain pandas
> load works fine. It's used to show the pattern (and complete the
> LitData/LitLogger/LitServe trio), not because this dataset needs it. See
> [`training/README.md`](training/README.md) for how to skip it.

> The drive itself was created with `Teamspace().new_folder("data")` (or via
> the UI: Teamspace → Drives → New folder). **Its mount path isn't
> guaranteed** — `new_folder`'s own docstring claims `/teamspace/folders/data`,
> but in this environment it actually mounted at
> `/teamspace/lightning_storage/data`. If you set this up fresh, confirm your
> own mount path (`ls /teamspace/lightning_storage` after creating the drive)
> and update `MOVIELENS_DATA_DIR` / `MOVIELENS_LITDATA_DIR` above to match —
> don't assume either path without checking.

## Workflow

The demo's full lifecycle, from a fresh clone to a served model:

**1. Debug/test locally** — `train_movielens.py` (monitors `val_ap`, has
early stopping) or `train_movielens_tensor.py` (monitors `val_acc`, no early
stopping) both work the same way:

```bash
python training/optimize_data.py     # one-time: build the LitData copy
python training/train_movielens.py --lr 1e-2 --batch_size 256 --max_epochs 20
```

**2. Smoke test locally** — same real pipeline at minimal scale (1 epoch, 2
batches); still exercises litlogger + checkpoint + PR-curve artifact for real:

```bash
python training/train_movielens.py --smoke_test
```

**3. Smoke test the remote job path** — one job on the actual target machine,
before spending on a full sweep (catches issues like a missing package or a
bad working-directory assumption that only show up remotely):

```bash
python training/sweep_launcher.py --smoke_test
```

**4. Launch the full hyperparameter sweep** as remote jobs, all grouped under
one experiment so you can compare configs side by side:

```bash
python training/sweep_launcher.py
```

See [`training/README.md`](training/README.md), "Grouping experiments" for how
that grouping works and its caveats (versions are timestamps, re-runs never
overwrite, machine type isn't varied automatically).

**5. Run the winning config for real** — pick the best version's `lr`/
`batch_size` from the sweep experiment in the Lightning UI, then launch one
longer job with those hyperparameters under its own `--logger_name`, reusing
the same `Job.run` pattern `sweep_launcher.py` uses:

```python
from lightning_sdk import Studio, Job, Machine
import pathlib

studio = Studio()
REPO_ROOT = pathlib.Path(".").resolve()  # run from the repo root
cmd = (
    f"python {REPO_ROOT}/training/train_movielens.py "
    f"--lr <best_lr> --batch_size <best_batch_size> --max_epochs 100 "
    f"--logger_name ml100k-best"
)
Job.run(name="ml100k-best-full-run", machine=Machine.CPU, studio=studio, command=cmd)
```

**6. Serve the result** — `server.py` pulls its checkpoint from the litlogger
model registry (not a local path) via `EXPERIMENT_NAME` (defaults to
`"ml100k-default"`; set it to whatever `--logger_name` you used in step 5):

```bash
EXPERIMENT_NAME=ml100k-best python serving/server.py   # LitServe API on :8011
streamlit run serving/app.py                            # UI that calls the API
```

To serve the seeded demo checkpoint instead of your own run:

```bash
EXPERIMENT_NAME=movielens-demo python serving/server.py
```

See [`serving/README.md`](serving/README.md) for the full checkpoint
resolution details.

## Checkpoints & artifacts — nothing is kept in the studio

See [`training/README.md`](training/README.md) for how logging to **litlogger**
works. In short: checkpoints and artifacts are uploaded to the **experiment
manager**, not stored locally, which keeps the studio lean.

## Onboarding: two ways to get this into a new Studio

Both work within your teamspace; pick whichever fits. (Duplicating into a
*different* teamspace than this one may need an admin to enable that.)

**Option A — Duplicate this Studio** (Lightning's native feature): Studio →
⋮ menu → Duplicate. Copies the whole studio, including installed packages,
not just the git-tracked files — the most reliable way to get an identical
environment.

**Option B — Clone the repo into a Studio.** Public at
`https://github.com/anazeneli/movielens-lightning-onboarding-demo`:

```bash
git clone https://github.com/anazeneli/movielens-lightning-onboarding-demo
cd movielens-lightning-onboarding-demo
pip install -e .                                  # installs `recsys` in editable mode
python training/train_movielens.py --smoke_test   # verify the pipeline works end to end
python training/optimize_data.py
python training/train_movielens.py --max_epochs 5 --logger_name my-first-run
```

Cloning brings the code, not the environment — but every dependency here is
already part of the standard Lightning Studio base image. Run `--smoke_test`
first to confirm; if anything's missing, install it individually.

No data setup needed either way: the `data` Drive is scoped to the
**teamspace**, not any one studio, so it's already mounted and populated in
any Studio in the same teamspace. In a *different* teamspace, `prepare_data()`
fails fast with a clear message instead of silently downloading — see "Data"
above.

## Version control

This repo is a git repo pushed to GitHub (public):
`https://github.com/anazeneli/movielens-lightning-onboarding-demo`. Because
the studio directory doubles as the home directory, `.gitignore` denies
everything by default and explicitly allows only `recsys/`, `training/`,
`serving/`, `README.md`, `pyproject.toml`, `.lightningignore`, and
`scratch.ipynb` — so shell
configs, `.ssh`, editor state, and `.claude/` session data never get
committed (verified before making the repo public — never appeared in any
commit).
