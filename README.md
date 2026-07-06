# MovieLens Two-Tower Recommender

A [two-tower](https://research.google/blog/) recommender on **MovieLens 100K**:
learn an embedding for each user and each movie; their dot product predicts
whether a user will like a movie (implicit feedback — rating ≥ 4 is a positive).
Train it, then serve top-K recommendations.

Uses three Lightning "Lit" products end to end:
[**LitData**](https://github.com/Lightning-AI/litData/tree/main) (streaming),
[**LitLogger**](https://github.com/Lightning-AI/LitLogger/tree/main)
(experiment tracking), [**LitServe**](https://github.com/Lightning-AI/litserve/tree/main)
(inference API).

## Layout

| Folder | What lives here |
|---|---|
| [`recsys/`](recsys/) | Shared library — model + data pipeline, used by both training and serving. |
| [`training/`](training/) | Data fetch/optimize, train script, tensor-metrics variant, sweep launcher. |
| [`serving/`](serving/) | Inference API (LitServe), Streamlit UI, standalone demo. |

Installed as an editable package (`pip install -e .`) so `from recsys...`
resolves everywhere — see `pyproject.toml`.

## Onboarding

### 1. Connect to your data — the shared drive, not local

Raw MovieLens files live on the teamspace's shared **Drive**, not in this
studio's local disk, so any Studio in the teamspace already has them:

| Path | What | Built by |
|---|---|---|
| `/teamspace/lightning_storage/ml-100k/` | Raw files (`u.data`, `u.item`, ...) | `training/fetch_data.py` |
| `/teamspace/lightning_storage/ml-100k/ml-100k-optimized/` | LitData-optimized streaming chunks | `training/optimize_data.py` — an example LitData workflow (see `training/README.md` for why LitData's used on a dataset this small) |

Both paths are defined once in `recsys/constants.py` (`RAW_DATA_DIR` /
`LITDATA_DIR`) and imported everywhere else that needs them -- change the
root there, or override `MOVIELENS_DATA_DIR` / `MOVIELENS_LITDATA_DIR`, which
take precedence over the file's defaults.

**Nothing auto-downloads.** Both scripts fail fast with a clear error if their
input is missing, rather than silently fetching data on a remote job machine.
We recommend storing even small datasets like this one on the shared drive
rather than locally, so every Studio and remote job in the teamspace can use
it without separate setup.
If the raw data isn't already on the drive:

```bash
python training/fetch_data.py       # downloads ml-100k, but only if not already there
python training/optimize_data.py    # builds the LitData-optimized copy from it
```

> Drive mount path isn't guaranteed — if setting this up fresh, confirm yours
> (`ls /teamspace/lightning_storage`) and update the env vars above if different.

### 2. Connect to your code — duplicate this Studio, or pull from git

Either gets you the code; pick whichever fits:

- **Duplicate this Studio** (Lightning's native feature): Studio → ⋮ menu →
  Duplicate. Copies the whole environment, including installed packages, not
  just the git-tracked files.
- **Clone the repo**:
  ```bash
  git clone https://github.com/anazeneli/movielens-lightning-onboarding-demo
  cd movielens-lightning-onboarding-demo
  pip install -e .   # installs `recsys` in editable mode
  ```
  Every dependency is already part of the standard Lightning Studio base
  image — nothing else to install.

Either way, data setup is unchanged: it's step 1, and it's teamspace-wide, not
studio-specific.

### 3. Smoke test locally, then experiment

```bash
python training/train_movielens.py --smoke_test   # 1) local smoke test
python training/sweep_launcher.py --smoke_test    # 2) remote smoke test (one job, real grouping)
python training/sweep_launcher.py                 # 3) full sweep, grouped under one folder in the experiment manager
```

Pick the best config from that folder in the Lightning UI, then use
`training/launch_job.py` to run it longer as its own remote job -- it's
anchored to its own file location, so it works regardless of which directory
you run it from (unlike a one-off snippet using the repo-root-relative
`Path(".").resolve()`, which breaks if you're not standing in the repo when
you run it):

```bash
# 4) longer run on the winning config
python training/launch_job.py \
    --lr <best_lr> --batch_size <best_batch_size> --max_epochs 100 \
    --logger_name ml100k-best --machine H100
```

```bash
# 5) serve the result
EXPERIMENT_NAME=ml100k-best python serving/server.py   # LitServe API on :8011
streamlit run serving/app.py                            # UI that calls it
```

Or skip straight to serving the seeded demo checkpoint (also `server.py`'s
default if `EXPERIMENT_NAME` isn't set):

```bash
EXPERIMENT_NAME=movielens-demo python serving/server.py
```

See [`training/README.md`](training/README.md) and
[`serving/README.md`](serving/README.md) for the mechanics behind each step —
why smoke tests exist, how sweep grouping actually works, checkpoint
resolution details.

## Experiment organization

Each launched job is its own LitLogger experiment. A slash-delimited
`--logger_name` creates real folder hierarchy in the experiment manager UI
(confirmed by testing — undocumented in the public API):

```text
--logger_name = {project}/{workflow}/{experiment_group}/{experiment_name}
       example = ml-100k/train_movielens/20260706-192010/sweep-lr0.01-bs256
```

`sweep_launcher.py` generates the `experiment_group` (one per sweep
invocation) and a unique `experiment_name` per job; `train_movielens.py`
never hardcodes any of this — run it standalone and it defaults to a flat
`run-lr<lr>-bs<batch_size>` name instead (no folder). Full naming convention
and caveats: [`training/README.md`](training/README.md), "Grouping experiments".

## Checkpoints & artifacts

Nothing is kept in the studio: litlogger uploads checkpoints and artifacts
straight to the experiment manager on every run (see
[`training/README.md`](training/README.md)). This keeps a remote job's
ephemeral disk irrelevant, and this repo checkpoint-free.

## Version control

Public repo: `https://github.com/anazeneli/movielens-lightning-onboarding-demo`.
Since the studio directory doubles as the home directory, `.gitignore` denies
everything except the project files (`recsys/`, `training/`, `serving/`,
`README.md`, `pyproject.toml`, `.lightningignore`, `scratch.ipynb`) — shell
configs, credentials, and editor state never get committed.
