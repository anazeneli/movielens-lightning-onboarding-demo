# train_movielens.py

import argparse
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root on path for `recsys`

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from litlogger import LightningLogger
from recsys.movielens_datamodule import MovieLens100K
from recsys.model import TwoTowerModel
from lightning_sdk import Studio

def main():
    # ── 1) CLI args ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train two-tower recommender on MovieLens 100K with LitLogger"
    )
    # Data & model hyperparameters
    parser.add_argument("--batch_size",    type=int,   default=256,  help="DataLoader batch size")
    parser.add_argument("--val_split",     type=float, default=0.1,  help="Validation split fraction")
    parser.add_argument("--embedding_dim", type=int,   default=64,   help="Size of embedding vectors")
    parser.add_argument("--lr",            type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--max_epochs",    type=int,   default=20,   help="Number of epochs")
    parser.add_argument("--precision",     type=int,   default=32,   choices=[16,32], help="Trainer precision")
    # Logger settings
    parser.add_argument("--logger_name",   type=str,   default="ml100k-default", help="LitLogger experiment name")
    parser.add_argument("--teamspace",     type=str,   default=Studio().teamspace.name,    help="LitLogger teamspace")
    args = parser.parse_args()

    # ── 2) LitLogger setup ─────────────────────────────────────────
    logger = LightningLogger(
        name      = args.logger_name,
        teamspace = args.teamspace,
        log_model = True
    )
    # Log any metadata you like
    logger.log_metadata({
        "dataset":      "MovieLens100K",
        "batch_size":   args.batch_size,
        "val_split":    args.val_split,
        "embedding_dim":args.embedding_dim,
        "lr":           args.lr,
    })

    # ── 3) DataModule ───────────────────────────────────────────────
    dm = MovieLens100K(batch_size=args.batch_size, val_split=args.val_split)
    dm.prepare_data()
    dm.setup()
    # cardinalities exposed by the DataModule (computed over the full dataset)
    num_users = dm.num_users
    num_items = dm.num_items

    # ── 4) Model ────────────────────────────────────────────────────
    model = TwoTowerModel(
        num_users     = num_users,
        num_items     = num_items,
        embedding_dim = args.embedding_dim,
        lr            = args.lr
    )

    # ── 5) Checkpoint callback ──────────────────────────────────────
    # No dirpath: checkpoints stage under the logger's run dir (transient) and
    # litlogger (log_model=True) uploads the best one to the experiment manager,
    # so nothing accumulates in the studio.
    ckpt_cb = ModelCheckpoint(
        filename     = "ml100k-{epoch:02d}-{val_acc:.2f}",
        monitor      = "val_acc",
        save_top_k   = 1,
        mode         = "max"
    )

    # ── 6) Trainer ──────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator       = "auto",
        devices           = "auto",
        max_epochs        = args.max_epochs,
        precision         = args.precision,
        callbacks         = [ckpt_cb],
        logger            = logger,
        log_every_n_steps = 20,
    )

    # ── 7) Fit! ─────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)
    print("✅ Best checkpoint:", ckpt_cb.best_model_path)

    # ── 8) Finalize logger ─────────────────────────────────────────
    logger.finalize()

if __name__ == "__main__":
    main()