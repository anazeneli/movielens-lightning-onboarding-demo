# training/optimize_data.py
#
# One-time step: convert the raw MovieLens 100K ratings into LitData's optimized,
# streamable chunk format, written to the shared teamspace drive. Training then
# streams these chunks with litdata.StreamingDataset instead of loading a
# DataFrame into memory — the pattern that lets data stay on the drive / in the
# cloud and scale far past what fits in RAM.
#
# Run once (re-run to rebuild):
#     python training/optimize_data.py

import os, json

import pandas as pd
from litdata import optimize

RAW_DIR = os.environ.get("MOVIELENS_DATA_DIR", "/teamspace/lightning_storage/ml-100k")
OUT_DIR = os.environ.get("MOVIELENS_LITDATA_DIR", "/teamspace/lightning_storage/ml-100k/ml-100k-optimized")


def to_sample(row):
    """Map one (user_idx, item_idx, label) tuple to a serializable LitData sample."""
    user_idx, item_idx, label = row
    return {"user": int(user_idx), "item": int(item_idx), "label": float(label)}


def main():
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(os.path.join(RAW_DIR, "u.data"), sep="\t", names=cols)

    # Implicit feedback + zero-based remapping (same as the old datamodule)
    df["label"] = (df.rating >= 4).astype("float32")
    df["user_idx"], _ = pd.factorize(df.user_id)
    df["item_idx"], _ = pd.factorize(df.item_id)

    inputs = list(zip(df.user_idx.tolist(), df.item_idx.tolist(), df.label.tolist()))
    print(f"Optimizing {len(inputs)} ratings → {OUT_DIR}")

    optimize(
        fn=to_sample,
        inputs=inputs,
        output_dir=OUT_DIR,
        chunk_bytes="64MB",
        mode="overwrite",
        num_workers=1,
    )

    # Persist cardinalities so the datamodule never has to re-read the raw file
    # just to size the embedding tables.
    stats = {"num_users": int(df.user_idx.nunique()), "num_items": int(df.item_idx.nunique())}
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"Done. {stats}")


if __name__ == "__main__":
    main()
