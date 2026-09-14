# train_movielens.py
import os
import argparse
import warnings
from urllib.parse import quote

# Lightning calls logger.log_graph() on every logger unconditionally, and
# litlogger answers with a UserWarning because it doesn't implement it. Nothing
# is wrong and there's no trainer flag to skip the call, so drop the one message
# rather than leave permanent noise in every run's output.
warnings.filterwarnings("ignore", message="LightningLogger does not support `log_graph`")

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning_sdk import Studio
from litlogger import LightningLogger
from litmodels import upload_model
from recsys.movielens_datamodule import MovieLens100K
from recsys.model import TwoTowerModel

DATA_ROOT = os.environ.get("MOVIELENS_LITDATA_DIR", "/teamspace/lightning_storage/data/ml-100k-litdata")


class PushCheckpointMidRun(Callback):
    """Publish the best-so-far checkpoint to the model registry *during* training.

    litlogger's log_model=True only registers at logger.finalize(), so while a
    remote job is still running its weights can't be pulled: `lightning model
    download` answers "Either the model doesn't exist or you don't have access
    to it", and the job's Drive artifacts dir is an empty placeholder until the
    job goes terminal.

    This uploads ModelCheckpoint's current best under a separate "-live" name,
    so the in-progress run is retrievable:

        lightning model download {owner}/{teamspace}/{experiment_name}-live

    Each push is a new version of that model, so you always get the latest by
    pulling the name without a version suffix. The "-live" suffix keeps this
    clear of the final log_model registration, which stays the canonical artifact.

    Uploads ONLY on improvement. ModelCheckpoint rewrites `best_model_path` only
    when its monitored metric improves, so comparing that path against the last
    one uploaded is an exact "did this get better?" test -- no re-uploading an
    unchanged file every epoch, which would otherwise spend a multi-MB upload and
    a registry version per epoch to store the same bytes.

    Note the gate follows whatever ModelCheckpoint monitors, which in this script
    is `val_ap` (mode="max"), NOT `val_loss`. EarlyStopping is the callback
    watching `val_loss`. To gate pushes on loss instead, change ckpt_cb's
    monitor -- don't add a second criterion here, or the uploaded checkpoint and
    the registered-at-finalize one stop being the same model.
    """

    def __init__(self, ckpt_cb, model_name, every_n_epochs=1):
        self.ckpt_cb = ckpt_cb
        self.model_name = model_name
        self.every_n_epochs = every_n_epochs
        self.n_pushed = 0
        self._last_pushed_path = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.every_n_epochs <= 0:
            return
        # Only rank 0 uploads -- every rank holds the same best checkpoint, so
        # letting all of them push would just race to write identical versions.
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        path = self.ckpt_cb.best_model_path
        if not path or not os.path.isfile(path):
            return
        # Unchanged path == the monitored metric did not improve since the last
        # push. Nothing new to publish, so don't spend the upload.
        if path == self._last_pushed_path:
            return
        score = self.ckpt_cb.best_model_score
        score_str = f"{float(score):.4f}" if score is not None else "n/a"
        try:
            upload_model(
                name=self.model_name,
                model=path,
                progress_bar=False,
                verbose=0,
                metadata={
                    "epoch": str(trainer.current_epoch),
                    "in_progress": "true",
                    "monitor": str(self.ckpt_cb.monitor),
                    "best_score": score_str,
                },
            )
        except Exception as e:
            # A failed mid-run push must never take the training run down with
            # it -- the run's real output is the final log_model registration.
            # Deliberately do NOT record _last_pushed_path here: leaving it unset
            # means the next improvement-free epoch retries this same checkpoint
            # rather than skipping it as already-published.
            log(f"WARNING: mid-run checkpoint push failed at epoch {trainer.current_epoch}: {e}")
        else:
            self._last_pushed_path = path
            self.n_pushed += 1
            log(
                f"pushed mid-run checkpoint (epoch {trainer.current_epoch}, "
                f"{self.ckpt_cb.monitor}={score_str}) -> {self.model_name}"
            )


def log(msg):
    print(f"[train] {msg}", flush=True)


def main():
    # ── 1) CLI args ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train two-tower recommender on MovieLens 100K"
    )
    # Data & model hyperparameters
    parser.add_argument("--batch_size",    type=int,   default=256,  help="DataLoader batch size")
    parser.add_argument("--val_split",     type=float, default=0.1,  help="Validation split fraction")
    parser.add_argument("--embedding_dim", type=int,   default=64,   help="Size of embedding vectors")
    parser.add_argument("--lr",            type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--max_epochs",    type=int,   default=20,   help="Number of epochs")
    parser.add_argument("--precision",     type=int,   default=32,   choices=[16,32], help="Trainer precision")
    # Logger settings
    parser.add_argument(
        "--logger_name", type=str, default=None,
        help="This run's experiment name. Defaults to run-lr<lr>-bs<batch_size> "
             "(or run-smoke-test for --smoke_test); sweep_launcher.py overrides this with "
             "its own sweep-* naming.",
    )
    # Grouping metadata -- sweep_launcher.py sets these so many experiments can
    # be filtered/compared as one sweep; this script just logs whatever it's given.
    parser.add_argument("--project", type=str, default="ml-100k")
    parser.add_argument("--workflow", type=str, default="train_movielens")
    parser.add_argument("--experiment_group", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument(
        "--push_mid_run_every", type=int, default=0,
        help="Publish the best-so-far checkpoint to the model registry every N "
             "validation epochs, under '{experiment_name}-live', so weights are "
             "retrievable WHILE the run is still going (log_model=True alone only "
             "registers at finalize). 0 (default) disables it.",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help=(
            "Run the REAL pipeline at minimal scale (1 epoch, 2 batches) to "
            "verify everything works end to end: litlogger experiment and "
            "checkpoint upload -- both still go to the "
            "experiment manager, not local disk. NOT side-effect-free: it "
            "creates a real (tiny) experiment. Deliberately does NOT use "
            "Trainer(fast_dev_run=True) -- Lightning forcibly swaps any real "
            "logger for a no-op DummyLogger under fast_dev_run, which would "
            "skip the very litlogger/checkpoint integration this is "
            "meant to verify."
        ),
    )
    args = parser.parse_args()
    if args.logger_name is None:
        args.logger_name = (
            "run-smoke-test" if args.smoke_test
            else f"run-lr{args.lr}-bs{args.batch_size}"
        )

    # ── 2) Logger setup ─────────────────────────────────────────
    # Resolve the current teamspace from the Lightning SDK instead of hardcoding
    teamspace = Studio().teamspace
    teamspace_name = teamspace.name

    # Initialize litlogger.
    # checkpoint_name pins the registry name to logger_name. Without it the
    # checkpoint registers under the *experiment* name, which the platform
    # timestamps on creation (ml100k-best -> ml100k-best-2026-09-14T15-24-53.766+00-00),
    # so serving could never reconstruct the name it was stored under.
    logger = LightningLogger(
        name=args.logger_name,
        teamspace=teamspace_name,
        log_model=True,
        checkpoint_name=args.logger_name,
    )

    # Log metadata
    logger.log_metadata({
        "dataset": "MovieLens100K",
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "embedding_dim": args.embedding_dim,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "precision": args.precision,
        "data_dir": DATA_ROOT, 
        "smoke_test": args.smoke_test,
        "project": args.project,
        "workflow": args.workflow,
        "experiment_group": args.experiment_group,
        "experiment_name": args.experiment_name,
        "sweep_id": args.sweep_id,
    })

    # ── 3) DataModule ───────────────────────────────────────────────
    dm = MovieLens100K(data_dir=DATA_ROOT, batch_size=args.batch_size, val_split=args.val_split)
    dm.prepare_data()
    dm.setup()
    # cardinalities exposed by the DataModule (computed over the full dataset)
    num_users = dm.num_users
    num_items = dm.num_items

    # ── 3b) Model ────────────────────────────────────────────────────
    model = TwoTowerModel(
        num_users     = num_users,
        num_items     = num_items,
        embedding_dim = args.embedding_dim,
        lr            = args.lr
    )

    # ── 4) Checkpoint callback ──────────────────────────────────────
    # No dirpath: checkpoints stage under the logger's run dir (transient on the
    # job machine) and litlogger (log_model=True) uploads the best one to the
    # experiment manager, so it survives the ephemeral remote machine.
    ckpt_cb = ModelCheckpoint(
        filename     = "ml100k-{epoch:02d}-{val_ap:.2f}",
        monitor      = "val_ap",
        save_top_k   = 1,
        mode         = "max"
    )

    # ── 4b) Early stopping ──────────────────────────────────────────
    # Stop a run once val_loss stops improving for `patience` epochs
    # (i.e. it's been climbing/flat for ~10 epochs).
    early_stop_cb = EarlyStopping(
        monitor   = "val_loss",
        mode      = "min",
        patience  = 10,
    )

    # ── 4c) Mid-run checkpoint push (opt-in) ────────────────────────
    # log_model=True only registers at finalize(), so without this the weights
    # of a still-running remote job can't be pulled. See PushCheckpointMidRun.
    callbacks = [ckpt_cb, early_stop_cb]
    if args.push_mid_run_every > 0:
        live_model_name = f"{teamspace.owner.name}/{teamspace_name}/{args.logger_name}-live"
        callbacks.append(
            PushCheckpointMidRun(ckpt_cb, live_model_name, args.push_mid_run_every)
        )
        log(f"mid-run pushes enabled every {args.push_mid_run_every} epoch(s) -> {live_model_name}")

    # ── 5) Trainer ──────────────────────────────────────────────────
    # --smoke_test only caps scale (1 epoch, 2 batches); the logger, callbacks,
    # and everything else stay identical to a real run -- see the --smoke_test
    # help text for why fast_dev_run=True can't be used here instead.
    trainer_kwargs = dict(
        accelerator       = "auto",
        devices           = "auto",
        precision         = args.precision,
        callbacks         = callbacks,
        logger            = logger,
        log_every_n_steps = 20,
        check_val_every_n_epoch = 1,   # validate (and log val_* metrics) every epoch
    )
    if args.smoke_test:
        trainer_kwargs.update(max_epochs=1, limit_train_batches=2, limit_val_batches=2)
    else:
        trainer_kwargs.update(max_epochs=args.max_epochs)
    trainer = L.Trainer(**trainer_kwargs)

    # ── 6) Fit! ─────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)
    print("✅ Best checkpoint:", ckpt_cb.best_model_path)

    # ── 7) Finalize logger ─────────────────────────────────────────
    logger.finalize()

    # litlogger's auto-printed URL appends a broken "- vNone" suffix; print a
    # clean, working link to the experiment instead. logger_name can contain
    # "/" (see training/README.md, "Grouping experiments"), so it needs the
    # same URL-encoding litlogger's own link uses, or the link breaks.
    print(
        f"📊 View experiment: "
        f"https://lightning.ai/{teamspace.owner.name}/{teamspace_name}/experiments/"
        f"{quote(args.logger_name, safe='')}"
    )
    if args.smoke_test:
        print("✅ Smoke test passed -- litlogger experiment and checkpoint upload verified.")

if __name__ == "__main__":
    main()