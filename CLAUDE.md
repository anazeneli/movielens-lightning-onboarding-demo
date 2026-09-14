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

**Nothing is ever overwritten, and names are get-or-create.** Reusing an exact
`--logger_name` reuses the *same* experiment, so metrics from separate runs
collide in it. Every job needs a unique name.

**Jobs can't write into the live Studio filesystem.** A studio job's outputs go
to home (`$LIGHTNING_ARTIFACTS_DIR`) and surface afterwards under
`/teamspace/jobs/<name>/artifacts`, which is **read-only**. Writing there from
inside a job fails with `OSError: [Errno 30] Read-only file system`.

## Tooling

**The `lightning` CLI on PATH in this Studio is stale** (`2026.04.23`, verb-first
`lightning run job`). Current is `2026.9.3`, noun-first (`lightning job run`,
`lightning deployment create`, `lightning model download`) — the old build
doesn't have those subcommands at all. Use the current CLI without touching the
environment:

```bash
uvx --from lightning-sdk lightning <command>
```

**Do not blind-upgrade `lightning-sdk` in this Studio.** The environment has a
pinned `litlogger` that the training path depends on. The *Python SDK* works fine
on the installed version — only the CLI is behind.

Pipelines are **SDK-only**: `lightning pipeline` exposes just `logs`, with no
`create`/`run`/`list`. Build them in Python (`from lightning_sdk.pipeline import
Pipeline, JobStep, DeploymentReleaseStep, Schedule`).

## Cost awareness

Machine time bills from **allocation** (`started_at`) to release — queueing and
Studio snapshotting beforehand are free, but machine boot, image pull and
environment setup are billed before your first line runs (measured: ~2 min on
T4). `job.total_cost` is provisional when a job first reports terminal and climbs
for ~2–3 minutes; a just-finished job can read `0.0`. Don't quote the first read.

Confirm before launching anything on A100/H100/H200/B200 or with a high machine
count, and prefer `Machine.T4` or `Machine.CPU` for smoke tests — this dataset
gains nothing from a large GPU.
