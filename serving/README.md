# serving/

Inference and demo layer — takes a trained checkpoint and turns it into
recommendations. All three import the model from [`recsys/`](../recsys/).

| File | Purpose |
|---|---|
| `server.py` | **Inference API** built on [LitServe](https://github.com/Lightning-AI/litserve/tree/main). Loads a checkpoint, exposes `/predict` — given `user_ids` + `item_ids`, returns scored top-K items with movie titles (from `u.item` on the drive). Runs on `:8011`. |
| `app.py` | **Streamlit UI.** The human-facing client: pick a user + K, calls the API, renders a table. Point it at the API with `INFERENCE_API_URL`. |
| `recommender_demo.py` | **Standalone demo** — loads a checkpoint and prints recommendations directly, no server. |

Run:

```bash
python serving/server.py          # start the API
streamlit run serving/app.py      # start the UI (in another terminal)
```

## Checkpoint source

`server.py` downloads its checkpoint via `litmodels.download_model` (see
[`training/README.md`](../training/README.md) for how checkpoints get
uploaded there in the first place). It resolves the model name in this order:

1. `CHECKPOINT_NAME` — a full `owner/teamspace/experiment_name[:version]`.
2. `EXPERIMENT_NAME` — just an experiment name, combined with the *current*
   teamspace (`Studio().teamspace`).
3. Neither set — falls back to `DEFAULT_CHECKPOINT_NAME` in `server.py`, a
   placeholder pointing at no real checkpoint. It'll fail with an explicit
   console error telling you to set one of the above.

Before picking a value for `CHECKPOINT_NAME` / `EXPERIMENT_NAME`, check your
teamspace's **Weights Registry** in the Lightning UI (Teamspace →
Weights Registry) to see what checkpoints/litmodels your team has already
uploaded, and test against one of those rather than guessing a name.
