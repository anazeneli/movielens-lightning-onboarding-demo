# training/optimize_data.py
#
# One-time step: convert the raw MovieLens 100K ratings into LitData's optimized,
# streamable chunk format, written to the shared teamspace drive. Training then
# streams these chunks with litdata.StreamingDataset instead of loading a
# DataFrame into memory — the pattern that lets data stay on the drive / in the
# cloud and scale far past what fits in RAM.
#
# Run once -- skips if the dataset is already built, like fetch_data.py:
#     python training/optimize_data.py
#     python training/optimize_data.py --force   # rebuild anyway

import argparse
import os, json

import pandas as pd
from litdata import optimize

from recsys.constants import RAW_DATA_DIR as RAW_DIR, LITDATA_DIR as OUT_DIR

# Gate on both files, not the directory: an interrupted run leaves index.json
# without stats.json, which would otherwise look complete and fail later in
# setup(). stats.json is written last, so it's the real completion marker.
REQUIRED = ("index.json", "stats.json")


def is_built(out_dir):
    """True only if every file a complete dataset needs is present."""
    return all(os.path.exists(os.path.join(out_dir, f)) for f in REQUIRED)


def to_sample(row):
    """Map one (user_idx, item_idx, label) tuple to a serializable LitData sample."""
    user_idx, item_idx, label = row
    return {"user": int(user_idx), "item": int(item_idx), "label": float(label)}


def main(force=False):
    if is_built(OUT_DIR) and not force:
        print(f"LitData dataset already present at {OUT_DIR} -- nothing to do. "
              f"(--force to rebuild)")
        return

    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(os.path.join(RAW_DIR, "u.data"), sep="\t", names=cols)

    # Implicit feedback + zero-based remapping (same as the old datamodule)
    df["label"] = (df.rating >= 4).astype("float32")
    df["user_idx"], _ = pd.factorize(df.user_id)
    df["item_idx"], _ = pd.factorize(df.item_id)

    inputs = list(zip(df.user_idx.tolist(), df.item_idx.tolist(), df.label.tolist()))
    print(f"Optimizing {len(inputs)} ratings → {OUT_DIR}")

    try:
        optimize(
            fn=to_sample,
            inputs=inputs,
            output_dir=OUT_DIR,
            chunk_bytes="64MB",
            mode="overwrite",
            num_workers=1,
        )
    except ImportError as e:
        # Safety net for litdata < 0.2.75 (pyproject floors it above this, so
        # normally dead code): it registered the result in the UI catalogue via
        # a lightning-sdk symbol that no longer exists. That runs *after* the
        # chunks are finalised, so the data is fine -- but only continue if it's
        # that exact import and index.json really landed.
        if "V1DatasetType" not in str(e):
            raise
        if not os.path.exists(os.path.join(OUT_DIR, "index.json")):
            raise RuntimeError(
                f"litdata failed on the V1DatasetType import *and* left no "
                f"index.json in {OUT_DIR} -- the chunks were not written, so "
                f"this is a real failure, not the cosmetic catalogue step."
            ) from e
        print(f"NOTE: litdata's dataset-catalogue registration was skipped "
              f"({e}). Chunks and index.json were written; continuing.")

    # Persist cardinalities so the datamodule never has to re-read the raw file
    # just to size the embedding tables.
    stats = {"num_users": int(df.user_idx.nunique()), "num_items": int(df.item_idx.nunique())}
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"Done. {stats}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if the dataset is already present.")
    main(force=p.parse_args().force)
