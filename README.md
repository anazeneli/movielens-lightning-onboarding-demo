# MovieLens Two-Tower Recommender

A [two-tower](https://research.google/blog/) recommender on **MovieLens 100K**:
learn an embedding for each user and each movie; their dot product predicts
whether a user will like a movie (implicit feedback — rating ≥ 4 is a positive).
Train it, then serve top-K recommendations.

## Layout

| Folder | What lives here |
|---|---|
| [`recsys/`](recsys/) | **Shared library** — model + data pipeline, imported by both training and serving. |
| [`training/`](training/) | **Training** — train script, a tensor-metrics variant, and the hyperparameter sweep launcher. |
| [`serving/`](serving/) | **Serving** — the inference API (LitServe), a Streamlit UI, and a standalone demo. |
| `Explainer videos/` | Walkthrough recording. |

The shared code is a package so there's one source of truth for the model and
data loading. Entry scripts in `training/` and `serving/` add the repo root to
`sys.path` so `from recsys.model import ...` resolves regardless of where they're launched.

## Data — the shared teamspace drive

The dataset is **not** stored in this studio. It lives on a teamspace **Drive**
named `data`, which mounts at:

```
/teamspace/lightning_storage/data/ml-100k/    # u.data, u.item, ...
```

This drive shows up in the Lightning UI and mounts on **remote job machines**
too, so sweeps can read it without re-downloading. Override the path with the
`MOVIELENS_DATA_DIR` env var. `MovieLens100K.prepare_data()` **fails fast** if
the data is missing (no silent internet download on a remote machine).

> The drive was created with `Teamspace().new_folder("data")` (or via the UI:
> Teamspace → Drives → New folder).

## Train

```bash
python training/train_movielens.py --lr 1e-2 --batch_size 256 --max_epochs 20
```

Launch the lr × batch-size sweep as remote jobs:

```bash
python training/sweep_launcher.py
```

## Serve

```bash
python serving/server.py          # LitServe API on :8011
streamlit run serving/app.py      # UI that calls the API
```

> Note: the serving scripts currently load a checkpoint from a local path that
> no longer exists — point them at a checkpoint pulled from the experiment
> manager before running inference.

## Checkpoints & artifacts — nothing is kept in the studio

See [`training/README.md`](training/README.md) for how logging to **litlogger**
works. In short: checkpoints and artifacts are uploaded to the **experiment
manager**, not stored locally, which keeps the studio lean.
