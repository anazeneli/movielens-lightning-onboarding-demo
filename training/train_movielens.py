# train_movielens.py

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_sdk import Studio
from litlogger import LightningLogger
# `recsys` resolves via the repo's editable install (`pip install -e .`) --
# works regardless of cwd, no sys.path hack needed.
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
        "--logger_name", type=str, default="ml100k-default",
        help=(
            "Experiment name in the experiment manager (the 'group'). Reuse the "
            "same name across multiple runs -- e.g. every combo in a hyperparameter "
            "sweep -- to group them as versions of ONE experiment instead of "
            "creating a separate experiment per run. Each run still gets its own "
            "timestamped version, so re-running never overwrites a prior one; "
            "distinguish versions via the logged metadata (lr, batch_size, ...)."
        ),
    )
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
            "meant to verify. --logger_name defaults to 'smoke-test' here "
            "unless you override it."
        ),
    )
    args = parser.parse_args()
    if args.smoke_test and args.logger_name == "ml100k-default":
        args.logger_name = "smoke-test"

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
    # clean, working link to the experiment instead.
    print(
        f"📊 View experiment: "
        f"https://lightning.ai/{teamspace.owner.name}/{teamspace_name}/experiments/{args.logger_name}"
    )
    if args.smoke_test:
        print("✅ Smoke test passed -- litlogger experiment, checkpoint, and PR-curve artifact all verified.")

if __name__ == "__main__":
    main()