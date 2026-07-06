# server.py
import os, json, traceback, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path for `recsys`
import pandas as pd
import torch
import litserve as ls
from litmodels import download_model
from lightning_sdk import Studio
from recsys.model import TwoTowerModel

def log(msg):
    print(f"[RecSysAPI] {msg}", flush=True)

def _resolve_checkpoint():
    """Locate a checkpoint uploaded by litlogger (see training/README.md).

    litlogger uploads checkpoints under "{owner}/{teamspace}/{experiment_name}[:version]"
    (owner/teamspace auto-resolved from Studio().teamspace). Override with
    CHECKPOINT_NAME for a specific model/version, otherwise this resolves to
    the latest checkpoint of EXPERIMENT_NAME (default "ml100k-default").
    """
    model_name = os.environ.get("CHECKPOINT_NAME")
    if not model_name:
        teamspace = Studio().teamspace
        experiment_name = os.environ.get("EXPERIMENT_NAME", "ml100k-default")
        model_name = f"{teamspace.owner.name}/{teamspace.name}/{experiment_name}"

    download_dir = os.environ.get("CHECKPOINT_DOWNLOAD_DIR", "/tmp/recsys-checkpoints")
    os.makedirs(download_dir, exist_ok=True)
    log(f"downloading checkpoint '{model_name}' -> {download_dir}")
    result = download_model(model_name, download_dir=download_dir)
    # download_model's docstring promises an absolute path, but in practice it
    # returns just the filename(s) relative to download_dir -- resolve against
    # download_dir rather than trusting the value is already absolute.
    names = result if isinstance(result, list) else [result]
    path = Path(names[0])
    if not path.is_absolute():
        path = Path(download_dir) / path
    if path.is_dir():
        ckpts = sorted(path.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt file found in downloaded model dir {path}")
        path = ckpts[-1]
    return path

class RecSysAPI(ls.LitAPI):
    def setup(self, device):
        log("setup()")
        # 1) Load checkpoint (downloaded from the experiment manager, not local disk)
        ckpt = _resolve_checkpoint()
        log(f"ckpt path -> {ckpt}")
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing ckpt: {ckpt}")
        self.model = TwoTowerModel.load_from_checkpoint(str(ckpt))
        self.model.eval().to(device)
        log(f"model device -> {next(self.model.parameters()).device}")

        # 2) Load MovieLens item metadata (adjust path if needed)
        item_cols = [
            "movie_id","title","release_date","video_release_date","IMDb_URL",
            "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
            "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
            "Romance","Sci-Fi","Thriller","War","Western"
        ]
        items_path = Path(os.environ.get("MOVIELENS_DATA_DIR", "/teamspace/lightning_storage/data/ml-100k")) / "u.item"
        if not items_path.is_file():
            log(f"WARNING: {items_path} not found. Titles will be None.")
            self.item_df = None
        else:
            self.item_df = pd.read_csv(items_path, sep="|", names=item_cols, encoding="latin-1")
            # zero-based index alignment helper: movie_id (1..1682) → idx (0..1681)
            self.item_df["idx"] = self.item_df["movie_id"] - 1
            self.item_df.set_index("idx", inplace=True)
            log(f"Loaded {len(self.item_df)} item rows")

        log("setup() done")

    def decode_request(self, request):
        log("decode_request()")
        try:
            dev = next(self.model.parameters()).device
            # Required fields
            users = torch.tensor(request["user_ids"], dtype=torch.long, device=dev)
            items = torch.tensor(request["item_ids"], dtype=torch.long, device=dev)
            # Optional top_k provided by client
            top_k = request.get("top_k", None)

            log(f"users.shape={users.shape} items.shape={items.shape} top_k={top_k}")
            return [users, items, top_k]  # return as list for positional unpacking
        except Exception:
            log("Exception in decode_request:")
            traceback.print_exc()
            raise

    def predict(self, *args):
        log("predict()")
        try:
            # Expecting (user_ids, item_ids, top_k)
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                user_ids, item_ids, top_k = (*args[0], None) if len(args[0]) == 2 else args[0]
            else:
                # pad top_k if not given
                if len(args) == 2:
                    user_ids, item_ids = args
                    top_k = None
                else:
                    user_ids, item_ids, top_k = args

            log(f"predict got user_ids={user_ids.shape}, item_ids={item_ids.shape}, top_k={top_k}")

            with torch.no_grad():
                logits = self.model(user_ids, item_ids)
                probs  = torch.sigmoid(logits)  # [batch] here batch=len(item_ids)

            # Sort & slice if requested
            if top_k is not None:
                # item_ids is aligned 1:1 with probs
                scores, order = torch.sort(probs, descending=True)
                scores = scores[:top_k]
                sel_items = item_ids[order[:top_k]]
            else:
                scores = probs
                sel_items = item_ids

            # Build rich response dicts
            response_rows = []
            for score, idx in zip(scores.tolist(), sel_items.tolist()):
                row = {
                    "item_idx": int(idx),
                    "score": float(score)
                }
                if self.item_df is not None and idx in self.item_df.index:
                    row.update({
                        "movie_id": int(self.item_df.loc[idx, "movie_id"]),
                        "title":    self.item_df.loc[idx, "title"]
                    })
                response_rows.append(row)

            log(f"predict() returning {len(response_rows)} rows (first 3): {response_rows[:3]}")
            return response_rows  # pass list to encode_response
        except Exception:
            log("Exception in predict:")
            traceback.print_exc()
            raise

    def encode_response(self, output):
        log("encode_response()")
        # output is already JSON-serializable list of dicts
        return {"results": output}

if __name__ == "__main__":
    os.environ.setdefault("LIT_LOG_LEVEL", "DEBUG")
    max_batch = int(os.getenv("MAX_BATCH_SIZE", "1"))
    log(f"starting server, max_batch_size={max_batch}")
    api = RecSysAPI(max_batch_size=max_batch)
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8011)