import os
import torch
import pandas as pd
from recsys.model import TwoTowerModel
from recsys.constants import RAW_DATA_DIR as DATA_DIR

# 1) Load MovieLens item metadata from the shared teamspace drive
item_cols = ["movie_id","title","release_date","video_release_date","IMDb_URL",
             "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
             "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
             "Romance","Sci-Fi","Thriller","War","Western"]
items = pd.read_csv(os.path.join(DATA_DIR, "u.item"), sep="|", names=item_cols, encoding="latin-1")

# 2) Load trained Two-Tower model checkpoint
ckpt_path = "checkpoints/ml100k-epoch=19-val_acc=0.61.ckpt"
model = TwoTowerModel.load_from_checkpoint(ckpt_path)
model.eval()

# 3) Suppose we want recommendations for user_id = 42 (factorized index)
user_idx = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4) Compute scores for all item indices [0…1681]
item_ids_tensor = torch.arange(len(items), device=device)
with torch.no_grad():
    user_tensor = torch.tensor([user_idx], device=device)
    user_emb    = model.user_embedding(user_tensor).squeeze(0)      # [emb_dim]
    item_embs   = model.item_embedding(item_ids_tensor)            # [num_items, emb_dim]
    scores      = (item_embs @ user_emb).cpu()                     # [num_items]

# 5) Select top-5 items
topk = 5
top_indices = torch.topk(scores, topk).indices.tolist()

print(f"Top {topk} recommendations for user {user_idx}:")
for rank, idx in enumerate(top_indices, start=1):
    title = items.loc[items.movie_id == idx+1, "title"].values[0]  # movie_id in ML=idx+1
    score = scores[idx].item()
    print(f"{rank}. {title} (score {score:.4f})")