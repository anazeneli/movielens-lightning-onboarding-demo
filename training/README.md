# training/

Everything needed to train the two-tower model and run hyperparameter sweeps.

| File | Purpose |
|---|---|
| `fetch_data.py` | **One-time step.** Downloads the raw MovieLens 100K files to the shared drive -- but only if they're not already there. |
| `optimize_data.py` | **One-time step.** Converts the raw MovieLens ratings into LitData's streamable chunk format on the shared drive. See "Data pipeline: LitData" below. |
| `train_movielens.py` | **Main train script.** litlogger + `ModelCheckpoint` (monitors `val_ap`) + `EarlyStopping`. |
| `train_movielens_tensor.py` | Near-duplicate variant that monitors `val_acc` and has no early stopping. |
| `sweep_launcher.py` | Fans out a `lr × batch_size` grid as Lightning **jobs** (`Machine.CPU` by default, swap for whatever fits your budget); each job is its own experiment, grouped into one folder in the experiment manager (see "Grouping experiments" below). `--smoke_test` launches a single job instead of the full grid, to verify the remote path first. |
| `launch_job.py` | Launches a **single** remote job running `train_movielens.py` with whatever hyperparameters you give it -- e.g. a longer run on a sweep's winning config. Anchored to its own file location, so it works from any cwd (unlike a one-off `Path(".").resolve()` snippet). |

First-time setup (from the repo root), then train locally:

```bash
pip install -e .                                                        # one-time, installs `recsys` in editable mode
python training/fetch_data.py                                           # one-time, only downloads if not already on the shared drive
python training/optimize_data.py                                        # one-time, builds the LitData copy
python training/train_movielens.py --lr 1e-2 --batch_size 256 --max_epochs 20
```

Run the sweep (remote jobs) -- smoke test the remote path first, then run the full grid:

```bash
python training/sweep_launcher.py --smoke_test
python training/sweep_launcher.py
```

## Data pipeline: LitData

`recsys/movielens_datamodule.py` streams training data with
[LitData](https://github.com/Lightning-AI/litData/tree/main)'s
`StreamingDataset` / `StreamingDataLoader`, reading from a LitData-optimized copy of the ratings on
the shared drive (`/teamspace/lightning_storage/ml-100k/ml-100k-optimized`, built by
`optimize_data.py` from the raw files in `.../ml-100k`).

### Changing the shared-drive root

`RAW_DATA_DIR` / `LITDATA_DIR` are defined once in `recsys/constants.py` and
imported by every script that needs them (`fetch_data.py`, `optimize_data.py`,
`movielens_datamodule.py`, `serving/server.py`, `serving/recommender_demo.py`)
-- change the root there and it takes effect everywhere. `MOVIELENS_DATA_DIR`
/ `MOVIELENS_LITDATA_DIR` env vars override each path without editing code.

| Constant | Env var override | Default |
|---|---|---|
| `RAW_DATA_DIR` | `MOVIELENS_DATA_DIR` | `/teamspace/lightning_storage/ml-100k` |
| `LITDATA_DIR` | `MOVIELENS_LITDATA_DIR` | `<RAW_DATA_DIR>/ml-100k-optimized` |

Both defaults must resolve to a path *inside* one of the shared drive's
mounted subfolders (e.g. `ml-100k/...`) -- `/teamspace/lightning_storage/`
itself is not writable by the studio user, only its mounted subfolders are.
Writing to it silently hangs `optimize_data.py` instead of raising (the
worker subprocess dies without reporting an error, and the main process
blocks forever waiting on it).

> **⚠️ This is demonstrational, not necessary for this dataset.** MovieLens
> 100K is 100K rows — it fits in memory easily, and a plain pandas
> `DataFrame` + `torch.utils.data.Dataset` (the original approach) works fine
> and is simpler. LitData's real value is streaming data **that doesn't fit in
> RAM** straight from cloud/drive storage without a full local copy — it's
> included here to show the pattern (and the three "Lit" products — LitData /
> LitLogger / LitServe — together), not because this dataset needs it.

If you'd rather skip LitData for this dataset: revert
`recsys/movielens_datamodule.py` to load `u.data` directly with pandas (no
`optimize_data.py` step needed), and skip building the litdata directory.

## How we log to litlogger (the experiment manager)

We use [`litlogger`](https://github.com/Lightning-AI/LitLogger/tree/main)'s `LightningLogger`, which writes to
the teamspace **experiment manager** — **not** to local disk. Nothing
accumulates in the studio; everything is viewable in the Lightning UI under
*Experiments*. Here's the full flow, step by step:

**1. Create the experiment.**
```python
logger = LightningLogger(name=args.logger_name, teamspace=..., log_model=True)
```
`log_model=True` is the key flag — it tells the logger to upload checkpoints as
model artifacts (see step 4).

**2. Log run metadata (hyperparameters).**
```python
logger.log_metadata({"dataset": "MovieLens100K", "lr": args.lr, ...})
```
Static key/values attached to the experiment for later comparison.

**3. Log metrics.** The logger is handed to the Trainer:
```python
trainer = L.Trainer(logger=logger, ...)
```
Then every `self.log("val_ap", ...)` / `self.log("train_loss", ...)` in
[`recsys/model.py`](../recsys/model.py) is captured by the logger, **buffered
locally, and uploaded to the experiment manager in rate-limited batches** (so
it doesn't hammer the API every step).

**4. Log the model checkpoint.** Because `log_model=True`, the logger registers
an `after_save_checkpoint` hook. When `ModelCheckpoint` saves the best model
(`save_top_k=1`), the logger **uploads that `.ckpt` to the experiment manager /
model registry** (via `litmodels.upload_model`). With `save_top_k=1` the upload
is deferred until `finalize()` (step 6). We deliberately set **no `dirpath`** on
`ModelCheckpoint`, so the checkpoint stages in a transient run dir and never
lands in a persisted `checkpoints/` folder — the uploaded copy is the source of
truth. Pull it back later with `litmodels.download_model`.

Verified with a real run (`--logger_name smoke-test`, 2 epochs): only one file
was uploaded — the single best-`val_ap` checkpoint, not one per epoch — and
`litmodels.download_model` retrieved it back successfully. Two things worth
knowing, found only by actually doing this (not documented upstream):

- **Model name format:** `download_model`/`upload_model` key checkpoints as
  `{owner}/{teamspace}/{experiment_name}[:version]` (e.g.
  `lightning-ai/mle-demo/smoke-test`). This comes from reading litlogger's
  `ModelArtifact` class, not from any docstring.
- **`download_model`'s return value is not what its docstring says.** It
  claims to return "the absolute path to the downloaded model file," but in
  practice it returns just the bare filename(s) (e.g.
  `['ml100k-epoch=01-val_ap=0.57.ckpt']`), relative to `download_dir`. See
  `serving/server.py`'s `_resolve_checkpoint()` for the defensive handling
  this requires (join against `download_dir` rather than trusting the result
  is absolute).

**5. Log file artifacts.** The PR-curve plot in `recsys/model.py` is uploaded as
a file artifact:
```python
self.logger.experiment.log_file(path, remote_path="pr_curve.png")
```
It's stored remotely under `experiments/<name>/pr_curve.png`.

**6. Finalize.**
```python
logger.finalize()
```
Flushes any buffered metrics and performs the deferred checkpoint upload.

### Why this keeps the studio lean
The local `lightning_logs/` dir is only *transient staging* — the real copies of
metrics, checkpoints, and artifacts live in the experiment manager. That's why
this repo has no `checkpoints/` folder and `lightning_logs/` is ignored: on a
remote job machine the local disk is ephemeral anyway, but the uploaded results
survive.

## Grouping experiments

litlogger's public API only documents a flat `name` -- no `folder`/`group`
parameter. An earlier version abused a **slash-delimited `--logger_name`** to
fake folder hierarchy in the UI, but with `log_model=True` litlogger registers
the best checkpoint in the model registry *under the experiment name*,
recombined as `{owner}/{teamspace}/{name}`. The registry uses `/` only as the
`owner/teamspace/model_name` delimiter, so a multi-slash name is unparseable
and the checkpoint upload fails:

```text
ValueError: Model name must be in the format `organization/teamspace/model_name`
```

So `--logger_name` is now a single flat, hyphen-delimited segment; grouping is
by shared **name prefix** instead of folders:

```text
--logger_name = {project}-{sweep_id}-lr{lr}-bs{bs}
       example = ml-100k-20260706-192010-lr0.01-bs256
```

`sweep_launcher.py` still passes each piece down as its own CLI flag, logged as
metadata -- so everything stays filterable/searchable regardless of the name:

| Flag | Meaning | Example |
|---|---|---|
| `--project` | The broader body of work; leads the name prefix. | `ml-100k` |
| `--workflow` | The repeatable code path (provenance metadata; not in the name). | `train_movielens` |
| `--experiment_group` | One sweep -- all jobs from one `sweep_launcher.py` invocation share this `sweep_id`; second part of the name prefix. | `20260706-192010` |
| `--experiment_name` | This one job's full flat name (== `--logger_name`). | `ml-100k-20260706-192010-lr0.01-bs256` |
| `--sweep_id` | Same value as `experiment_group`. | `20260706-192010` |

All of a sweep's runs share the `{project}-{sweep_id}-` prefix, so filter/sort
by it in the experiment manager to compare them.

**Responsibility split:**
- `train_movielens.py` owns one experiment -- its own config, metrics,
  checkpoint, artifacts. It never hardcodes what it belongs to: `--project`
  (default `ml-100k`), `--workflow` (default `train_movielens`),
  `--experiment_group`, `--experiment_name`, and `--sweep_id` (all default to
  `""`, unset) are just logged as metadata, whatever they're set to. If
  `--logger_name` isn't given either, it defaults to a flat
  `run-lr<lr>-bs<batch_size>` (or `run-smoke-test` for `--smoke_test`), since a
  standalone run has no experiment group.
  **It has no dependency on `sweep_launcher.py`** -- no import, no requirement
  that it's running -- so it's a complete, standalone experiment either way.
- `sweep_launcher.py` owns the grouping: it generates one `sweep_id` per
  invocation (a timestamp) and builds a flat, unique `--logger_name` per job,
  `{project}-{sweep_id}-lr<lr>-bs<batch_size>` -- every lr/batch_size combo in
  the grid is unique, so the full string is unique within the sweep. It sets
  `--experiment_name` to that same string and passes all of it down as plain
  CLI args -- the only thing connecting the two scripts.

A few things worth knowing:

- **Naming convention:** experiment names are flat and hyphen-delimited --
  `{project}-{sweep_id}-lr..-bs..` for sweep jobs, `run-lr..-bs..` (or
  `run-smoke-test`) for standalone `train_movielens.py` runs. The **Jobs** UI
  label (a separate system from experiments) is `sweep-<experiment_name>` and is
  never used as a `--logger_name`.
- **Nothing is ever overwritten.** litlogger does a strict get-or-create keyed
  on the full `--logger_name` string -- reusing one exactly reuses the *same*
  experiment (metrics/steps from different runs collide in it), so every job's
  `--logger_name` must be unique. Confirmed this the hard way: an earlier
  attempt at reusing one flat name across a sweep just overwrote itself
  instead of creating separate comparable runs.
- **Machine type isn't a sweep dimension today.** `sweep_launcher.py` launches
  every job on the same `machine=`. To compare a CPU run against a GPU run,
  launch jobs with different `machine=` values yourself, keeping the same
  `experiment_group`.
- **Local/standalone runs** default to `experiment_group=""` -- a real
  experiment, just not part of a sweep's prefix group.
- **Smoke tests** get their own `experiment_group` (a fresh timestamp for
  `sweep_launcher.py --smoke_test`, so repeated smoke tests don't collide) and
  `run-smoke-test` for direct local smoke tests.

To run this sweep's demo: launch it, then in the experiment manager filter by
the printed `ml-100k-<sweep_id>-` name prefix, compare the jobs' `val_ap` to
find the best config, then kick off a full run of that config under its own
`--logger_name` (e.g. `ml100k-best`) -- see the root [README.md](../README.md)'s
"Workflow" section, step 5.
