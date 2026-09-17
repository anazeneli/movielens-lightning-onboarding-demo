# CLAUDE.md

Orientation for an agent working in this repo. The per-directory READMEs carry
the detail; this file is the map plus the constraints that aren't guessable from
reading the code.

## What this is

A MovieLens 100K two-tower recommender, used end to end on Lightning AI:
train → track → register → serve. It is a **demonstration repo** — the model is
deliberately small so the platform mechanics stay visible.

| Directory | Role | README |
|---|---|---|
| `recsys/` | The model, datamodule, and shared path constants | — |
| `training/` | Training script + sweep launcher (fan-out) | [training/README.md](training/README.md) |
| `serving/` | LitServe API, Streamlit UI, standalone demo | [serving/README.md](serving/README.md) |
| `pipelines/` | Full lifecycle as one ordered, schedulable pipeline | [pipelines/README.md](pipelines/README.md) |

## First-time setup

Run in order. Every step is idempotent — re-running a satisfied step prints
"nothing to do" rather than redoing work.

```bash
pip install -e .                    # installs recsys + the deps the base image lacks
python training/fetch_data.py       # raw ml-100k -> shared drive (MD5-verified)
python training/optimize_data.py    # LitData chunks + stats.json (--force to rebuild)
python training/train_movielens.py --smoke_test    # end-to-end check, ~30s, CPU
```

**Prerequisite the scripts cannot do for you:** the data lands on a *teamspace
folder* mounted at `/teamspace/lightning_storage/<name>/`, and `.../data/` must
already exist. `/teamspace/lightning_storage/` itself is **not writable** — only
its mounted subfolders are — so if `ls /teamspace/lightning_storage/data` fails,
create the folder first (it mounts within ~60s, no Studio restart):

```python
from lightning_sdk import Teamspace
Teamspace(name="<teamspace>", org="<owner>").new_folder("data")
```

The mount root is `lightning_storage`, not the `folders` that `new_folder`'s
docstring claims. If your mount differs, override `MOVIELENS_DATA_DIR` /
`MOVIELENS_LITDATA_DIR` rather than editing `recsys/constants.py`.

Setup is correct when the smoke test prints `Smoke test passed` and the run
appears in the teamspace's experiments. Duplicating a Studio that is already set
up carries the installed environment with it, and the data folder is
teamspace-scoped, so a duplicate inside the same teamspace needs none of this.

## Constraints that will bite you

These are all load-bearing. Each one was learned by breaking it.

**The experiment name and the registry name are two different strings.**
Experiment names are nested — `{project}/{workflow}/{sweep_id}/{leaf}` — and each
`/` segment renders as a folder in the experiment manager. The **model registry**
cannot take that name: it parses `/` as its own `owner/teamspace/model_name`
delimiter, so a nested name arrives as 6 parts instead of 3 and upload fails with
`ValueError: Model name must be in the format 'organization/teamspace/model_name'`.

So checkpoints register under the **flattened** name (`/` → `-`), produced by
`recsys.constants.registry_name()`, which training and serving both call so they
can't disagree. That decoupling is only possible with **`log_model=False`**: with
`log_model=True` litlogger registers the checkpoint under `Experiment.name`
itself, and the nested name blows up the upload. `train_movielens.py` therefore
sets `log_model=False` and uploads the best checkpoint explicitly via
`litmodels.upload_model` after `fit()`.

`LightningLogger(checkpoint_name=...)` looks like it solves this and **does
not**. In litlogger 2026.8.28 it is assigned at `logger.py:435` and never read
again — the registry name always comes from `Experiment.name`. An earlier
revision of this file claimed it pinned the registry name and that dropping it
would break serving; both were wrong, and the argument has been removed.

**Remote jobs run with cwd = studio root, not the repo.** Any path handed to a
job must be absolute. Both `training/sweep_launcher.py` and
`pipelines/lifecycle_pipeline.py` resolve `REPO_ROOT` from `__file__` for this
reason. A relative path works locally and fails remotely.

**Checkpoints upload after `fit()`, not during the run.** Metrics stream as the
run proceeds, but the checkpoint is published once at the end, so a *running*
job's weights are not in the registry —
`lightning model download` on an in-progress run returns "Either the model
doesn't exist or you don't have access to it", and the job's Drive artifacts
directory is an empty placeholder until the job is terminal. `train_movielens.py`
has an opt-in `--push_mid_run_every N` callback that publishes the best-so-far
checkpoint under a `-live` name if you need mid-run retrieval.

**The platform may timestamp the displayed experiment name.** An experiment
created as `ml100k-best` has been observed stored as
`ml100k-best-2026-09-14T15-24-53.766+00-00` — the suffix is added server-side, so
no client change removes it. This does **not** affect the registry name: that is
built client-side from `--checkpoint_name`, which never leaves the process. To
find an experiment's real stored name, list it — don't reconstruct it.

**Jobs can't write into the live Studio filesystem.** A studio job's outputs go
to home (`$LIGHTNING_ARTIFACTS_DIR`) and surface afterwards under
`/teamspace/jobs/<name>/artifacts`, which is **read-only**. Writing there from
inside a job fails with `OSError: [Errno 30] Read-only file system`.

**If you don't upload it, it is gone.** This is the single most important thing
to internalise about writing code that runs here. Every job and pipeline step
runs on a machine that is destroyed when the step ends. A file written to local
disk — including `$LIGHTNING_ARTIFACTS_DIR` — is **not** guaranteed to survive.

Do not rely on automatic artifact collection. Verified the hard way: a pipeline
step wrote `recommendations.parquet` to `$LIGHTNING_ARTIFACTS_DIR`, completed
successfully, and the file appeared **nowhere** afterwards — not under
`lit://<owner>/<teamspace>/jobs/`, not under `/artifacts/`. Nothing errored. The
output was simply lost.

Anything you need after the run must be **explicitly pushed** to durable storage:

| What | How | Example |
|---|---|---|
| Model checkpoints | `litmodels.upload_model` (explicit, after `fit()`) | `train_movielens.py` |
| Arbitrary output files | `litmodels.upload_model_files` | `serving/batch_inference.py` |
| Metrics / params | litlogger — uploads as the run proceeds | `train_movielens.py` |
| Ad-hoc files | `lightning cp <file> lit://<owner>/<teamspace>/uploads/<path>` | — |

Both upload paths are versioned: re-uploading the same name adds a version rather
than overwriting, so scheduled jobs accumulate history for free.

Note litlogger's `log_file` artifact API is **not** a working route in this
teamspace — it returns `404` from the drive blob endpoint (see
[training/README.md](training/README.md), "File artifacts"). Use the model store.

Corollary for checkpoints: the checkpoint upload only happens after `fit()`
returns, so a job that dies mid-run leaves **nothing** behind. That is what
`--push_mid_run_every` exists for.

## Tooling

The `lightning` CLI on PATH is **2026.9.3**, noun-first (`lightning job run`,
`lightning model download`). `lai` is aliased to it. Earlier revisions of this
file warned about a stale `2026.04.23` verb-first build and told you to reach for
`uvx --from lightning-sdk` — that no longer applies, and neither does the old
"don't upgrade lightning-sdk" advice: the SDK is current and the training path
works on it.

Pipelines are **SDK-only**: `lightning pipeline` exposes only `logs` (and it
needs a STEP name, not just the pipeline). Build them in Python (`from
lightning_sdk.pipeline import Pipeline, JobStep, DeploymentReleaseStep,
Schedule`). Pipeline steps do **not** appear in `lightning job list` — read them
with `lightning pipeline logs <pipeline> <step>`.

### Skills

Lightning skills are installed for this repo (`.claude/skills/`, symlinked from
`.agents/skills/`). They load at session start — install one mid-session and you
must restart before `Skill` can invoke it.

| Skill | Use it for |
|---|---|
| `lightning-jobs` | Launch/monitor jobs and MMT, logs, SSH, artifacts. The one this repo leans on most. |
| `lightning-studios` | Create/start/stop Studios, switch machine types, upload files. |
| `lightning-deployments` | Long-lived autoscaled endpoints (what `DeploymentReleaseStep` cuts a release of). |
| `lightning-cost-estimation` | Live per-hour machine prices before committing to a sweep or a GPU tier. |
| `lightning-artifacts` | Publish a local file as a durable public `lightning.ai/artifacts/<id>` link. |
| `lightning-llm-gateway` | Hosted LLM calls + the teamspace model checkpoint registry. |
| `lightning-sandboxes` | Throwaway isolated VMs for untrusted or experimental code. |
| `lightning-blog` | Drafting/publishing on the Lightning blog (needs the blog-admin flag). |
| `find-skills` | Discover and install further skills. |

Two gotchas from `lightning-jobs` that bit this repo directly:
`lightning job inspect` does **not** emit parseable JSON (it wraps long values
mid-string) — use `lightning job list --json`; and `--query`/`--severity` on a
*finished* job can silently return zero lines, so fetch unfiltered and `grep`.

## Cost awareness

Machine time bills from **allocation** (`started_at`) to release — queueing and
Studio snapshotting beforehand are free, but machine boot, image pull and
environment setup are billed before your first line runs (measured: ~2 min on
T4). `job.total_cost` is provisional when a job first reports terminal and climbs
for ~2–3 minutes; a just-finished job can read `0.0`. Don't quote the first read.

Confirm before launching anything on A100/H100/H200/B200 or with a high machine
count, and prefer `Machine.T4` or `Machine.CPU` for smoke tests — this dataset
gains nothing from a large GPU.
