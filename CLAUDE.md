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

Setup is `pip install -e .` plus the one-time data steps in
[training/README.md](training/README.md).

## Constraints that will bite you

These are all load-bearing. Each one was learned by breaking it.

**Experiment names must be a single flat segment with no `/`.** With
`log_model=True`, litlogger registers the checkpoint under the experiment name,
recombined as `{owner}/{teamspace}/{name}`. The registry uses `/` only as its own
delimiter, so a slashed name is unparseable and the checkpoint upload fails with
`ValueError: Model name must be in the format 'organization/teamspace/model_name'`.
Group runs by shared **name prefix**, not by folder hierarchy.

**Remote jobs run with cwd = studio root, not the repo.** Any path handed to a
job must be absolute. Both `training/sweep_launcher.py` and
`pipelines/lifecycle_pipeline.py` resolve `REPO_ROOT` from `__file__` for this
reason. A relative path works locally and fails remotely.

**Checkpoints upload at `finalize()`, not during the run.** With `save_top_k=1`
the upload is deferred, so a *running* job's weights are not in the registry —
`lightning model download` on an in-progress run returns "Either the model
doesn't exist or you don't have access to it", and the job's Drive artifacts
directory is an empty placeholder until the job is terminal. `train_movielens.py`
has an opt-in `--push_mid_run_every N` callback that publishes the best-so-far
checkpoint under a `-live` name if you need mid-run retrieval.

**The platform timestamps experiment names; the registry name is pinned
separately.** An experiment created as `ml100k-best` is stored as
`ml100k-best-2026-09-14T15-24-53.766+00-00` — the suffix is added server-side, so
no client change removes it, and the bare name does not resolve. That used to
make the registered checkpoint unfindable by name. Both train scripts now pass
`checkpoint_name=args.logger_name` to `LightningLogger`, which pins the *model
registry* name to the flat name you chose, so `EXPERIMENT_NAME=ml100k-best`
resolves in serving. Drop that argument and serving breaks. To find an
experiment's real stored name, list it — don't reconstruct it.

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
| Model checkpoints | `log_model=True`, or `litmodels.upload_model` | `train_movielens.py` |
| Arbitrary output files | `litmodels.upload_model_files` | `serving/batch_inference.py` |
| Metrics / params | litlogger — uploads as the run proceeds | `train_movielens.py` |
| Ad-hoc files | `lightning cp <file> lit://<owner>/<teamspace>/uploads/<path>` | — |

Both upload paths are versioned: re-uploading the same name adds a version rather
than overwriting, so scheduled jobs accumulate history for free.

Note litlogger's `log_file` artifact API is **not** a working route in this
teamspace — it returns `404` from the drive blob endpoint (see
[training/README.md](training/README.md), "File artifacts"). Use the model store.

Corollary for checkpoints: `log_model=True` only publishes at `logger.finalize()`,
so a job that dies mid-run leaves **nothing** behind. That is what
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
