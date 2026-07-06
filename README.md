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
data loading. Entry scripts in `training/` and `serving/` add the repo root to
`sys.path` so `from recsys.model import ...` resolves regardless of where they're launched.

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
> the UI: Teamspace → Drives → New folder).

## Train

```bash
python training/optimize_data.py     # one-time: build the LitData copy
python training/train_movielens.py --lr 1e-2 --batch_size 256 --max_epochs 20
```

Launch the lr × batch-size sweep as remote jobs, all grouped under one
experiment so you can compare configs and pick a winner:

```bash
python training/sweep_launcher.py
```

See [`training/README.md`](training/README.md), "Grouping experiments" for how
that grouping works and its caveats (versions are timestamps, re-runs never
overwrite, machine type isn't varied automatically).

## Serve

```bash
python serving/server.py          # LitServe API on :8011
streamlit run serving/app.py      # UI that calls the API
```

> Note: the serving scripts currently load a checkpoint from a local path that
> no longer exists — point them at a checkpoint pulled from the experiment
> manager before running inference. See [`serving/README.md`](serving/README.md).

## Checkpoints & artifacts — nothing is kept in the studio

See [`training/README.md`](training/README.md) for how logging to **litlogger**
works. In short: checkpoints and artifacts are uploaded to the **experiment
manager**, not stored locally, which keeps the studio lean.

## Version control

This repo is a local git repo (`git log` to see history). Because the studio
directory doubles as the home directory, `.gitignore` denies everything by
default and explicitly allows only `recsys/`, `training/`, `serving/`,
`README.md`, and `.lightningignore` — so shell configs, `.ssh`, editor state,
and `.claude/` session data never get committed. There's no remote configured;
this history exists only in this studio unless you push it somewhere.
