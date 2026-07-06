# movielens_datamodule.py

import os, json
import torch
from litdata import StreamingDataset, StreamingDataLoader, train_test_split
from lightning.pytorch import LightningDataModule

from recsys.constants import LITDATA_DIR

# The LitData-optimized dataset lives on the shared teamspace drive, so every
# machine in the teamspace (including remote job machines) streams it from the
# same mount. Build it once with `python training/optimize_data.py`.
DATA_ROOT = LITDATA_DIR


def _collate(samples):
    """Stack a list of {"user","item","label"} samples into model-ready tensors."""
    users  = torch.tensor([s["user"]  for s in samples], dtype=torch.long)
    items  = torch.tensor([s["item"]  for s in samples], dtype=torch.long)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
    return users, items, labels


class MovieLens100K(LightningDataModule):
    def __init__(self, data_dir=DATA_ROOT, batch_size=128, val_split=0.2, num_workers=0):
        super().__init__()
        self.data_dir, self.batch_size, self.val_split = data_dir, batch_size, val_split
        self.num_workers = num_workers
        self.num_users = self.num_items = None

    def prepare_data(self):
        # The LitData-optimized chunks must already be on the shared drive. Fail
        # fast rather than silently rebuilding on a remote job machine.
        if not os.path.exists(os.path.join(self.data_dir, "index.json")):
            raise FileNotFoundError(
                f"No LitData dataset at {self.data_dir!r}. Build it once with:\n"
                f"    python training/optimize_data.py"
            )

    def setup(self, stage=None):
        # Embedding cardinalities were computed during optimize; read them back
        # instead of re-scanning the raw ratings.
        with open(os.path.join(self.data_dir, "stats.json")) as f:
            stats = json.load(f)
        self.num_users = stats["num_users"]
        self.num_items = stats["num_items"]

        # Stream the optimized chunks and split into train/val.
        full = StreamingDataset(input_dir=self.data_dir, shuffle=True)
        self.train_ds, self.val_ds = train_test_split(full, splits=[1 - self.val_split, self.val_split])

    def train_dataloader(self):
        # num_workers defaults to 0 to avoid shared-memory IPC crashes in
        # constrained (small /dev/shm) containers; raise it on larger machines.
        return StreamingDataLoader(self.train_ds, batch_size=self.batch_size,
                                   num_workers=self.num_workers, collate_fn=_collate)

    def val_dataloader(self):
        return StreamingDataLoader(self.val_ds, batch_size=self.batch_size,
                                   num_workers=self.num_workers, collate_fn=_collate)
