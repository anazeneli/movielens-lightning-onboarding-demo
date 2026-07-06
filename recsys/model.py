# model.py

import os

import matplotlib

matplotlib.use("Agg")  # headless rendering for the PR-curve artifact
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
)


class TwoTowerModel(LightningModule):
    def __init__(self, num_users, num_items, embedding_dim=64, lr=1e-3):
        super().__init__()
        # Automatically save all __init__ args to self.hparams and checkpoint
        self.save_hyperparameters()
        # Now self.hparams.num_users, self.hparams.embedding_dim, etc. are available

        self.user_embedding = nn.Embedding(self.hparams.num_users, self.hparams.embedding_dim)
        self.item_embedding = nn.Embedding(self.hparams.num_items, self.hparams.embedding_dim)
        self.lr = self.hparams.lr

        # ── Metrics ─────────────────────────────────────────────────
        # Separate instances per stage so train/val state never mixes and is
        # aggregated + reset correctly each epoch.
        self.train_acc = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        # Average precision = area under the PR curve; the scalar used to
        # rank/compare runs. The curve itself is logged as a plot artifact.
        self.val_ap = BinaryAveragePrecision()
        self.val_pr_curve = BinaryPrecisionRecallCurve()
        self.best_val_ap = -1.0

    def forward(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=1)

    def training_step(self, batch, _):
        users, items, labels = batch
        logits = self(users, items)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.sigmoid(logits)
        targets = labels.long()
        self.train_acc(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        users, items, labels = batch
        logits = self(users, items)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.sigmoid(logits)
        targets = labels.long()
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_ap.update(preds, targets)
        self.val_pr_curve.update(preds, targets)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)

    def on_validation_epoch_end(self):
        val_ap = self.val_ap.compute()
        self.log("val_ap", val_ap, prog_bar=True)

        # Keep a single PR-curve artifact for the best-AP epoch, so it's easy
        # to compare the winning curve across runs.
        if not self.trainer.sanity_checking and val_ap.item() > self.best_val_ap:
            self.best_val_ap = val_ap.item()
            self._log_pr_curve()

        self.val_ap.reset()
        self.val_pr_curve.reset()

    def _log_pr_curve(self):
        precision, recall, _ = self.val_pr_curve.compute()
        precision = precision.detach().cpu().numpy()
        recall = recall.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, marker=".")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve (AP={self.best_val_ap:.4f}, epoch {self.current_epoch})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

        out_dir = getattr(self.logger, "log_dir", None) or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "pr_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Upload the plot as an experiment artifact when the logger supports it.
        experiment = getattr(self.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log_file"):
            experiment.log_file(path, remote_path="pr_curve.png", verbose=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
