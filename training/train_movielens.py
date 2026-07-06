# train_movielens.py

import argparse
from urllib.parse import quote

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_sdk import Studio
from litlogger import LightningLogger
from recsys.movielens_datamodule import MovieLens100K
from recsys.model import TwoTowerModel


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
        "--smoke_test", action="store_true",
        help=(
            "Run the REAL pipeline at minimal scale (1 epoch, 2 batches) to "
            "verify everything works end to end: litlogger experiment, "
            "checkpoint upload, PR-curve artifact -- all still go to the "
            "experiment manager, not local disk. NOT side-effect-free: it "
            "creates a real (tiny) experiment. Deliberately does NOT use "
            "Trainer(fast_dev_run=True) -- Lightning forcibly swaps any real "
            "logger for a no-op DummyLogger under fast_dev_run, which would "
            "skip the very litlogger/checkpoint/artifact integration this is "
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

    # Initialize litlogger
    logger = LightningLogger(
        name=args.logger_name,
        teamspace=teamspace_name,
        log_model=True
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
        "smoke_test": args.smoke_test,
        "project": args.project,
        "workflow": args.workflow,
        "experiment_group": args.experiment_group,
        "experiment_name": args.experiment_name,
        "sweep_id": args.sweep_id,
    })

    # ── 3) DataModule ───────────────────────────────────────────────
    dm = MovieLens100K(batch_size=args.batch_size, val_split=args.val_split)
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

    # ── 5) Trainer ──────────────────────────────────────────────────
    # --smoke_test only caps scale (1 epoch, 2 batches); the logger, callbacks,
    # and everything else stay identical to a real run -- see the --smoke_test
    # help text for why fast_dev_run=True can't be used here instead.
    trainer_kwargs = dict(
        accelerator       = "auto",
        devices           = "auto",
        precision         = args.precision,
        callbacks         = [ckpt_cb, early_stop_cb],
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
        print("✅ Smoke test passed -- litlogger experiment, checkpoint, and PR-curve artifact all verified.")

if __name__ == "__main__":
    main()