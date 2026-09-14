# batch_inference.py
#
# Offline scoring for every user, written out as a Parquet artifact. This is the
# batch counterpart to server.py: same checkpoint, same resolution rules, but it
# runs to completion and exits rather than waiting for requests.
#
# Two-tower models are built for exactly this shape. Because user and item towers
# are independent, scoring every user against every item is one matmul of the two
# embedding tables -- not num_users forward passes. That's why this is cheap
# enough to run nightly on a small machine.
#
# Output goes to $LIGHTNING_ARTIFACTS_DIR, which on a Lightning job is the job's
# home and is collected as a job artifact. Anything written there is retrievable
# afterwards with `lightning cp lit://<owner>/<teamspace>/jobs/<job-name>/...`;
# anything written elsewhere is lost when the machine is released.

import os
from pathlib import Path

import pandas as pd
import torch

from recsys.constants import RAW_DATA_DIR
from recsys.model import TwoTowerModel

# Sibling module, not a package import -- serving/ isn't packaged, so this only
# resolves when serving/ is the cwd. Same constraint as recommender_demo.py.
from server import _resolve_checkpoint

TOP_K = int(os.environ.get("TOP_K", "10"))

ITEM_COLS = [
    "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def log(msg):
    print(f"[batch-inference] {msg}", flush=True)


def main():
    # $LIGHTNING_ARTIFACTS_DIR is set on a Lightning job; fall back to cwd so this
    # is still runnable locally for a quick check.
    out_dir = Path(os.environ.get("LIGHTNING_ARTIFACTS_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _resolve_checkpoint()
    log(f"checkpoint -> {ckpt}")
    # weights_only=False for the same reason as server.py: PyTorch >=2.6 otherwise
    # rejects this checkpoint's pickled numpy globals. It's our own registered
    # checkpoint, not an untrusted file.
    model = TwoTowerModel.load_from_checkpoint(str(ckpt), weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log(f"device -> {device}")

    # Read the table sizes off the model rather than the dataset: the checkpoint
    # is the source of truth here, and a mismatch would silently score garbage.
    num_users = model.user_embedding.num_embeddings
    num_items = model.item_embedding.num_embeddings
    log(f"scoring {num_users} users x {num_items} items")

    with torch.no_grad():
        user_embs = model.user_embedding(torch.arange(num_users, device=device))
        item_embs = model.item_embedding(torch.arange(num_items, device=device))
        # One matmul for the whole catalogue -- [num_users, num_items].
        scores = user_embs @ item_embs.T
        k = min(TOP_K, num_items)
        top_scores, top_items = torch.topk(scores, k, dim=1)

    top_scores = top_scores.cpu().numpy()
    top_items = top_items.cpu().numpy()

    # Titles are a nicety -- if the raw data isn't mounted, still emit the scores
    # rather than failing a nightly job over a display column.
    items_path = Path(RAW_DATA_DIR) / "u.item"
    if items_path.is_file():
        items = pd.read_csv(items_path, sep="|", names=ITEM_COLS, encoding="latin-1")
        # movie_id is 1-based, embedding index is 0-based.
        idx_to_title = dict(zip(items["movie_id"] - 1, items["title"]))
    else:
        log(f"WARNING: {items_path} not found -- titles will be null")
        idx_to_title = {}

    rows = []
    for user_idx in range(num_users):
        for rank in range(k):
            item_idx = int(top_items[user_idx, rank])
            rows.append({
                "user_idx": user_idx,
                "rank": rank + 1,
                "item_idx": item_idx,
                "movie_id": item_idx + 1,
                "title": idx_to_title.get(item_idx),
                "score": float(top_scores[user_idx, rank]),
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / "recommendations.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df)} rows ({num_users} users x top-{k}) -> {out_path}")
    print(df.head(10).to_string(index=False))

    # Writing to $LIGHTNING_ARTIFACTS_DIR is not enough on its own: automatic
    # artifact collection did not surface this file for a pipeline step (nothing
    # appeared under lit://<owner>/<teamspace>/jobs/ afterwards). Explicitly
    # uploading it is the reliable path -- it goes to the same model store the
    # checkpoints use, which is proven to round-trip, and makes the output
    # retrievable by NAME from anywhere:
    #
    #     lightning model download {owner}/{teamspace}/{OUTPUT_NAME}
    #
    # Set OUTPUT_NAME to disable/rename; each run adds a new version, so a
    # nightly schedule builds a history rather than overwriting.
    output_name = os.environ.get("OUTPUT_NAME")
    if not output_name:
        experiment = os.environ.get("EXPERIMENT_NAME")
        output_name = f"{experiment}-recommendations" if experiment else "recommendations"

    try:
        from lightning_sdk import Studio
        from litmodels import upload_model_files

        teamspace = Studio().teamspace
        full_name = f"{teamspace.owner.name}/{teamspace.name}/{output_name}"
        upload_model_files(
            name=full_name,
            path=out_path,
            progress_bar=False,
            verbose=0,
            metadata={
                "rows": str(len(df)),
                "users": str(num_users),
                "top_k": str(k),
            },
        )
        log(f"uploaded -> {full_name}")
    except Exception as e:
        # Don't fail a nightly job because the upload leg broke -- the Parquet is
        # still on disk under the artifacts dir. Log loudly enough to notice.
        log(f"WARNING: could not upload {out_path.name}: {e}")


if __name__ == "__main__":
    main()
