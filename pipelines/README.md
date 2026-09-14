# pipelines/

The full model lifecycle as **one dependency-ordered pipeline**, optionally on a
cron schedule:

```text
data-prep (CPU)  ->  train (GPU)  ->  eval (GPU)  ->  serve (deployment)
```

| File | Purpose |
|---|---|
| `lifecycle_pipeline.py` | Builds and launches the four-step pipeline. Each step picks its own machine; `--cron` attaches a native schedule; `--serve_studio` / `--serve_image` let the serving step come from a different repo or environment. |

Run it:

```bash
python pipelines/lifecycle_pipeline.py                          # on-demand, T4 throughout
python pipelines/lifecycle_pipeline.py --train_machine H100     # train on H100, rest on T4
python pipelines/lifecycle_pipeline.py --cron "0 6 * * *" --timezone America/New_York
```

Read step logs (the CLI *can* do this much — see "CLI is read-only" below):

```bash
lightning pipeline logs <pipeline-name> --teamspace lightning-ai/mle-demo
```

## Why a pipeline instead of `sweep_launcher.py`

They solve different problems and both belong in the repo:

| | `training/sweep_launcher.py` | `pipelines/lifecycle_pipeline.py` |
|---|---|---|
| Shape | **fan-out** — N independent jobs at once | **sequence** — ordered stages with dependencies |
| Use | hyperparameter search | prep → train → eval → ship |
| Machines | one `machine=` for the whole grid | a different machine per step |
| Scheduling | none | native cron via `Schedule` |

A sweep answers *"which config is best?"*. A pipeline answers *"take the thing we
decided on and get it into production, repeatedly."* Don't reach for the pipeline
to run a sweep — it executes its steps in order by design.

## How the steps chain

`JobStep(..., wait_for=...)` defaults to the sentinel `"DEFAULT"`, which
`lightning_sdk/pipeline/utils.py:prepare_steps` resolves to **a linear dependency
on the previous step**. So listing the steps in order is enough — `eval` cannot
start against a checkpoint `train` hasn't written yet.

To fan out instead, name the dependency explicitly:

```python
JobStep(name="eval-a", wait_for=["train"], ...)
JobStep(name="eval-b", wait_for=["train"], ...)   # runs alongside eval-a
```

**The first step must not have a `wait_for`** — `prepare_steps` raises
`ValueError: The first step isn't allowed to receive 'wait_for=...'`.

## Scheduling

`Pipeline.run()` takes `schedules=`, so cron is a platform primitive here — no
GitHub Action, no Airflow, no external cron box:

```python
Schedule(
    cron_expression="0 6 * * *",   # required
    name="ml-100k-nightly",
    timezone="America/New_York",   # IANA name; defaults to UTC
    parallel_runs=False,           # default: don't start a run while one is going
)
```

Leave `parallel_runs` off for anything that trains — two concurrent runs writing
the same experiment name is not what you want.

## The steps share a model, not a filesystem

This is the part worth understanding, because it's what makes the serving step
relocatable.

Each step runs on its **own machine**, which is gone when the step ends. Nothing
is handed between steps on disk. What connects them is the **model registry**:

1. `train` registers the checkpoint under `EXPERIMENT_NAME` (via
   `log_model=True` — see [`training/README.md`](../training/README.md), "How we
   log to litlogger").
2. `eval` and `serve` resolve that same name back to a checkpoint
   (`_resolve_checkpoint()` in [`serving/server.py`](../serving/server.py)).

Which means the serving step doesn't have to live in this repo at all:

```bash
python pipelines/lifecycle_pipeline.py \
  --serve_studio prod-serving \
  --serve_command "python /teamspace/studios/this_studio/server.py"
```

Training out of the research repo, serving out of the production repo, joined
only by a model name.

## Gotchas

Things that cost time to discover, in the spirit of the other READMEs here.

- **The CLI is read-only for pipelines.** `lightning pipeline` exposes only
  `logs` — there is no `pipeline create` / `run` / `list`. Everything that builds
  a pipeline is SDK-side (`from lightning_sdk.pipeline import ...`). If you go
  looking for a CLI equivalent of this script, there isn't one; that's why this
  is a Python file and not a shell one-liner.

- **`--serve_studio` / `--serve_image` require `--serve_command`.** The default
  command points at `serving/server.py` by **absolute path**, which exists only
  in *this* Studio. Pointing the step at a different Studio without changing the
  command launches something that dies on `No such file or directory`. The script
  refuses up front rather than letting that happen at runtime.

- **`experiment_name` is fixed when the pipeline is *defined*, not per run.** A
  scheduled pipeline re-executes the same command strings, so every nightly run
  trains under the same experiment name (litlogger versions it) and the
  deployment picks up the latest. That is usually what you want for "retrain
  nightly, serve the current model." If you need one distinct experiment per
  run, template the name inside the step command — `entrypoint` is `sh -c`, so
  `$(date +%Y%m%d)` works there.

- **`DeploymentReleaseStep` is not a job.** It cuts a new **release** of a
  long-lived deployment, so it has no terminal state the way the three `JobStep`s
  do. Re-running the pipeline updates the same `--deployment_name` in place
  rather than creating a second endpoint. It also keeps billing after the
  pipeline "finishes" — tear it down with
  `lightning deployment delete <name> --yes` when you're done.

- **Per-step machines are the cost lever.** Data prep on `Machine.CPU` and
  training on `Machine.H100` in the same pipeline is the normal case, not an
  optimization. Running the prep step on the training machine is the easiest way
  to pay H100 rates to read a CSV.

- **Scheduled vs. autoscaled are different things.** A `Schedule` runs a pipeline
  on a clock. A deployment's `AutoScaleConfig` reacts to *traffic* (and with
  `min_replicas=0` costs nothing at idle). Batch inference on a timetable is the
  first; a live endpoint is the second. Neither substitutes for the other.

## Prerequisites

The data must exist on the shared drive before the first run — the `data-prep`
step calls `MovieLens100K.prepare_data()`, which expects the LitData copy built
by [`training/optimize_data.py`](../training/optimize_data.py). See
[`training/README.md`](../training/README.md) for the one-time setup.
