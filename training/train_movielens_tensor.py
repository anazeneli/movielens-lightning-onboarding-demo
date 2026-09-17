# train_movielens.py

import argparse

import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from litlogger import LightningLogger
from litmodels import upload_model
from recsys.constants import registry_name
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
    parser.add_argument("--logger_name",   type=str,   default="ml-100k/train_movielens_tensor/ml100k-default",
                        help="LitLogger experiment name; '/' segments become folders")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Model-registry name (default: --logger_name flattened)")
    parser.add_argument("--teamspace",     type=str,   default=Studio().teamspace.name,    help="LitLogger teamspace")
    args = parser.parse_args()
    if args.checkpoint_name is None:
        args.checkpoint_name = registry_name(args.logger_name)

    # ── 2) LitLogger setup ─────────────────────────────────────────
    # log_model=False so logger_name can stay nested for folder hierarchy; the
    # checkpoint is uploaded by hand in step 8 under the flat checkpoint_name.
    # See train_movielens.py for the full reasoning (and why LightningLogger's
    # own checkpoint_name= argument does not do this).
    logger = LightningLogger(
        name            = args.logger_name,
        teamspace       = args.teamspace,
        log_model       = False,
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
    # step 8 uploads the best one to the model registry, so nothing accumulates
    # in the studio.
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

    # ── 8) Publish the best checkpoint, then finalize ──────────────
    teamspace = Studio().teamspace
    ckpt_model_name = f"{teamspace.owner.name}/{teamspace.name}/{args.checkpoint_name}"
    if ckpt_cb.best_model_path and os.path.isfile(ckpt_cb.best_model_path):
        upload_model(
            name=ckpt_model_name,
            model=ckpt_cb.best_model_path,
            progress_bar=False,
            verbose=0,
            metadata={"experiment": args.logger_name},
        )
        print(f"✅ Registered checkpoint: {ckpt_model_name}")

    logger.finalize()

if __name__ == "__main__":
    main()