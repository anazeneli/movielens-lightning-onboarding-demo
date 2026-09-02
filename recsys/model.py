# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
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
        # rank/compare runs (ModelCheckpoint monitors it).
        self.val_ap = BinaryAveragePrecision()

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

        # sync_dist=True on the plain tensors only: torchmetrics objects below
        # aggregate across ranks themselves, so syncing them again would warn.
        # Under DDP these two drive callbacks -- ModelCheckpoint monitors val_ap
        # and EarlyStopping monitors val_loss -- and per-rank values let ranks
        # disagree about the best epoch and when to stop.
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)

    def on_validation_epoch_end(self):
        val_ap = self.val_ap.compute()
        self.log("val_ap", val_ap, prog_bar=True, sync_dist=True)
        self.val_ap.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
