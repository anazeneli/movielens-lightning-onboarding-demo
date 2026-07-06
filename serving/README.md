# serving/

Inference and demo layer — takes a trained checkpoint and turns it into
recommendations. All three import the model from [`recsys/`](../recsys/).

| File | Purpose |
|---|---|
| `server.py` | **Inference API** built on [LitServe](https://github.com/Lightning-AI/litserve/tree/main). Loads a checkpoint, exposes `/predict` — given `user_ids` + `item_ids`, returns scored top-K items with movie titles (from `u.item` on the drive). Runs on `:8011`. |
| `app.py` | **Streamlit UI.** The human-facing client: pick a user + K, calls the API, renders a table. Point it at the API with `INFERENCE_API_URL`. |
| `reccomender_demo.py` | **Standalone demo** — loads a checkpoint and prints recommendations directly, no server. |

Run:

```bash
python serving/server.py          # start the API
streamlit run serving/app.py      # start the UI (in another terminal)
```

## ⚠️ Checkpoint source

`server.py` and `reccomender_demo.py` currently reference local checkpoint paths
that no longer exist (`~/my-models/...`, `checkpoints/...`). Since checkpoints
now live in the **experiment manager** (see [`training/README.md`](../training/README.md)),
these should be updated to download a checkpoint first, e.g.:

```python
from litmodels import download_model
ckpt = download_model("<teamspace>/<experiment-or-model-name>")
model = TwoTowerModel.load_from_checkpoint(ckpt)
```
