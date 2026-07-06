# training/

Everything needed to train the two-tower model and run hyperparameter sweeps.

| File | Purpose |
|---|---|
| `train_movielens.py` | **Main train script.** litlogger + `ModelCheckpoint` (monitors `val_ap`) + `EarlyStopping`. |
| `train_movielens_tensor.py` | Near-duplicate variant that monitors `val_acc` and has no early stopping. |
| `sweep_launcher.py` | Fans out a `lr × batch_size` grid as Lightning **jobs** (one H100 job per combo). |

Run locally:

```bash
python training/train_movielens.py --lr 1e-2 --batch_size 256 --max_epochs 20
```

Run the sweep (remote jobs):

```bash
python training/sweep_launcher.py
```

## How we log to litlogger (the experiment manager)

We use [`litlogger`](https://lightning.ai)'s `LightningLogger`, which writes to
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
