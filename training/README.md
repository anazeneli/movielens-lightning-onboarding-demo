# training/

Everything needed to train the two-tower model and run hyperparameter sweeps.

| File | Purpose |
|---|---|
| `optimize_data.py` | **One-time step.** Converts the raw MovieLens ratings into LitData's streamable chunk format on the shared drive. See "Data pipeline: LitData" below. |
| `train_movielens.py` | **Main train script.** litlogger + `ModelCheckpoint` (monitors `val_ap`) + `EarlyStopping`. |
| `train_movielens_tensor.py` | Near-duplicate variant that monitors `val_acc` and has no early stopping. |
| `sweep_launcher.py` | Fans out a `lr × batch_size` grid as Lightning **jobs** (`Machine.CPU` by default, swap for whatever fits your budget), grouped into one experiment. `--smoke_test` launches a single job instead of the full grid, to verify the remote path first. |

First-time setup (from the repo root), then train locally:

```bash
pip install -e .                                                        # one-time, installs `recsys` in editable mode
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
the shared drive (`/teamspace/lightning_storage/data/ml-100k-litdata`, built by
`optimize_data.py` from the raw files in `.../ml-100k`).

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

`--logger_name` is the experiment's name **and** its group: every run that
passes the same `--logger_name` becomes a **version** of that one experiment,
instead of its own separate experiment. `sweep_launcher.py` uses this — every
`lr × batch_size` job in a grid shares one `EXPERIMENT_GROUP`, so they land as
versions under a single experiment you can compare side by side to pick the
winning config. The current grid (`ml100k-sweep-wide-lr`) is 15 jobs (5
learning rates × 3 batch sizes); an earlier, narrower grid is kept commented
out in the file for reference.

A few things worth knowing about how this actually behaves:

- **Versions are timestamps, not labels.** litlogger auto-generates each
  version as the UTC time the run started — there's no way to name a version.
  Distinguish versions using the metadata logged in step 2 above (`lr`,
  `batch_size`, `embedding_dim`, ...), not the version string itself.
- **Nothing is ever overwritten.** Re-running with the same `--logger_name`
  (even with byte-identical arguments) always creates a brand new version —
  litlogger has no notion of "this config already ran, update it in place."
  Two runs of the same config will typically still show different metric
  curves anyway, since neither the embedding init nor the data shuffling is
  seeded.
- **Machine type isn't a sweep dimension today.** `sweep_launcher.py` launches
  every job on the same `machine=`. To compare, say, a CPU run against a GPU
  run in one group, launch jobs with different `machine=` values yourself,
  using the same `--logger_name` for all of them.

To run this experiment group's demo: launch the sweep, open the
`ml100k-sweep-wide-lr` experiment in the Lightning UI, compare the versions'
`val_ap` to find the best config, then kick off a full run of that config
under its own `--logger_name` (e.g. `ml100k-best`) -- see the root
[README.md](../README.md)'s "Workflow" section, step 5.
