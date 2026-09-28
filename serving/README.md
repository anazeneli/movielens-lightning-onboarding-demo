# serving/

Inference and demo layer — takes a trained checkpoint and turns it into
recommendations. All four import the model from [`recsys/`](../recsys/).

| File | Purpose |
|---|---|
| `server.py` | **Inference API** built on [LitServe](https://github.com/Lightning-AI/litserve/tree/main). Loads a checkpoint, exposes `/predict` — given `user_ids` + `item_ids`, returns scored top-K items with movie titles (from `u.item` on the drive). Runs on `:8011`. |
| `app.py` | **Streamlit UI.** The human-facing client: pick a user + K, calls the API, renders a table. Point it at the API with `INFERENCE_API_URL`. |
| `recommender_demo.py` | **Standalone demo** — loads a checkpoint and prints recommendations directly, no server. |
| `batch_inference.py` | **Offline scoring.** Scores every user against the catalogue in one matmul over the two embedding tables, writes `recommendations.parquet`, and **uploads it** (see below). Run on a schedule via [`pipelines/batch_inference_pipeline.py`](../pipelines/batch_inference_pipeline.py). |

## Output must be uploaded, or it's lost

`batch_inference.py` writes its Parquet to `$LIGHTNING_ARTIFACTS_DIR` **and then
explicitly uploads it** with `litmodels.upload_model_files`. The upload is the
part that matters.

Writing to the artifacts dir alone is not enough. A pipeline step did exactly
that, completed cleanly, and the file was then findable nowhere — not under
`lit://<owner>/<teamspace>/jobs/`, not under `/artifacts/`. Nothing errored; the
output was just gone with the machine. Automatic collection is not something to
depend on.

Retrieve a run's output by name:

```bash
lightning model download <owner>/<teamspace>/<experiment>-recommendations --download-dir ./out
```

The name defaults to `{EXPERIMENT_NAME}-recommendations`; set `OUTPUT_NAME` to
override it. Each run adds a **new version** rather than overwriting, so a
nightly schedule builds a history. A failed upload warns instead of failing the
job — the file is still on local disk at that point — so check the logs for
`WARNING: could not upload` if an expected version is missing.

Run:

```bash
python serving/server.py          # start the API
streamlit run serving/app.py      # start the UI (in another terminal)
```

## Deploy it: an endpoint anyone can call

`server.py` is also what runs behind a Lightning **Deployment**: a long-lived
HTTPS endpoint that callers reach with plain `curl` (or any HTTP client), with no
Lightning account, SDK or Studio on their side. Create it from this Studio:

```bash
TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")   # keep it: callers need it
lightning deployment create recsys-api --teamspace <owner>/<teamspace> \
  --studio <this-studio> --machine CPU --port 8011 \
  --min-replicas 0 --max-replicas 1 \
  --token-auth "$TOKEN" \
  -e CHECKPOINT_NAME=<owner>/<teamspace>/<logger_name> \
  --command "cd /teamspace/studios/this_studio/movielens-lightning-onboarding-demo/serving && python server.py"
lightning deployment inspect recsys-api --teamspace <owner>/<teamspace>   # endpoint URL
```

Then, from any machine **outside Lightning** (your laptop, another cloud):

```bash
URL=https://8011-dep-<id>-d.cloudspaces.litng.ai     # from `inspect`
curl -H "Authorization: Bearer $TOKEN" "$URL/health"                 # -> ok
curl -X POST "$URL/predict" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"user_ids":[0,0,0],"item_ids":[0,49,99],"top_k":3}'
# -> {"results":[{"item_idx":49,"score":0.32,"movie_id":50,"title":"Star Wars (1977)"}, ...]}
```

Verified 2026-09-28 from a laptop: `/health` returned `ok` and `/predict`
returned scored titles.

What the response means: each entry is one of the `item_ids` you sent, ranked by
`score`, the predicted probability (0–1) that the user rates it 4+ stars. It only
ranks the candidates you send; to recommend from the whole catalogue, send all
item ids `0..1681` with the `top_k` you want. `item_idx` is `movie_id - 1`.

Things that trip people up:

- **A browser shows "Unauthorized".** A browser can't send the bearer header, and
  `/predict` is a POST with a JSON body anyway. Use curl, Postman, or code.
  A request without the header returning 401 is the auth working.
- **The first call after idle is slow.** `--min-replicas 0` scales to zero, and
  bills nothing, when idle; the next request waits a few minutes while a replica
  boots, then answers normally. Before a demo, warm it with one call, or keep it
  up with `lightning deployment update recsys-api --min-replicas 1`, which bills
  continuously until you set it back to 0.
- **The URL and token are stable** across days, restarts and scale-to-zero. Only
  deleting the deployment or changing the token changes them.
- **Editing `server.py` does not update the deployment.** It keeps running what
  it was created with until a new release.
- **Swap the model** by updating `CHECKPOINT_NAME` to another flat registry name
  (e.g. a sweep's best run). URL and token stay the same.
- **The token is the only protection.** Share it like a password.
- **Tear down** with
  `lightning deployment delete recsys-api --teamspace <owner>/<teamspace> --yes`.

## Checkpoint source

`server.py` downloads its checkpoint via `litmodels.download_model` (see
[`training/README.md`](../training/README.md) for how checkpoints get
uploaded there in the first place). It resolves the model name in this order:

1. `CHECKPOINT_NAME` — a full `owner/teamspace/experiment_name[:version]`.
2. `EXPERIMENT_NAME` — just an experiment name, combined with the *current*
   teamspace (`Studio().teamspace`). Use the run's **flat** `--logger_name`
   (the leaf, e.g. `ml-100k-<sweep_id>-lr0.01-bs256`), not the foldered path the
   experiment manager shows (`ml-100k/<sweep_id>/...`): the registry has no
   folders, so the full path won't resolve.
3. Neither set — falls back to `DEFAULT_CHECKPOINT_NAME` in `server.py`, a
   placeholder pointing at no real checkpoint. It'll fail with an explicit
   console error telling you to set one of the above.

Before picking a value for `CHECKPOINT_NAME` / `EXPERIMENT_NAME`, check your
teamspace's **Weights Registry** in the Lightning UI (Teamspace →
Weights Registry) to see what checkpoints/litmodels your team has already
uploaded, and test against one of those rather than guessing a name.
