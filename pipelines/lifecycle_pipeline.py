# pipeline_launcher.py
#
# The full lifecycle as ONE Lightning pipeline:
#
#     data-prep (CPU) -> train (GPU) -> eval (GPU) -> serve (deployment)
#
# Why a pipeline instead of the four separate launches in sweep_launcher.py:
#
#   * Steps are dependency-ordered. wait_for defaults to "DEFAULT", which the SDK
#     resolves to a linear dependency on the previous step (see
#     lightning_sdk/pipeline/utils.py:prepare_steps), so listing steps in order
#     is enough -- eval can't run against a checkpoint training hasn't written.
#     Pass wait_for=["step-name"] explicitly to fan out instead of chaining.
#   * Each step picks its OWN machine. Data prep has no business on an H100, and
#     paying H100 rates to score 1682 items is the cost mistake this avoids.
#   * Schedules are native: Pipeline.run(schedules=[Schedule(cron_expression=...)])
#     gives cron without an external scheduler. That is what makes "daily batch
#     inference" a platform primitive here rather than a GitHub Action.
#
# The first step is not allowed a wait_for -- prepare_steps raises on it.

import argparse
import pathlib
from datetime import datetime

from lightning_sdk import Machine, Studio
from typing import List, Union

from lightning_sdk.pipeline import (
    DeploymentReleaseStep,
    DeploymentStep,
    JobStep,
    MMTStep,
    Pipeline,
    Schedule,
)

PROJECT = "ml-100k"
WORKFLOW = "train_movielens"

# Jobs run with cwd = studio root, not this repo -- use an absolute path, same
# reasoning as sweep_launcher.py.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Launch the data-prep -> train -> eval -> serve pipeline."
)
parser.add_argument(
    "--train_machine", default="T4",
    help="Machine for the training step (T4, L4, A100, H100, ...). Default T4 to "
         "keep a demo run cheap; the point of the pipeline is that this is "
         "independent of every other step's machine.",
)
parser.add_argument(
    "--eval_machine", default="T4",
    help="Machine for the eval step. Default T4.",
)
parser.add_argument(
    "--serve_machine", default="T4",
    help="Machine for the deployment step. Default T4.",
)
parser.add_argument(
    "--max_epochs", type=int, default=25,
    help="Epochs for the training step.",
)
parser.add_argument(
    "--deployment_name", default="recsys-api",
    help="Name of the deployment the serve step releases to. Re-running the "
         "pipeline cuts a NEW RELEASE of this same deployment rather than "
         "creating a second endpoint.",
)
parser.add_argument(
    "--serve_studio", default=None,
    help="Run the serve step from a DIFFERENT Studio than the one training ran "
         "in -- i.e. a different repo/environment. Steps are independent: each "
         "carries its own studio= or image=, so 'training lives in repo A, "
         "serving lives in repo B' is the normal case, not a workaround. "
         "Defaults to this Studio.",
)
parser.add_argument(
    "--serve_image", default=None,
    help="Alternative to --serve_studio: run the serve step from a Docker image "
         "instead of a Studio snapshot. Mutually exclusive with --serve_studio.",
)
parser.add_argument(
    "--serve_command", default=None,
    help="Command for the serve step. Defaults to this repo's serving/server.py "
         "by ABSOLUTE path -- which only exists in this Studio, so you must pass "
         "this whenever you use --serve_studio or --serve_image.",
)
parser.add_argument(
    "--cron", default=None,
    help="Attach a schedule, e.g. '0 6 * * *' for 06:00 daily. Omit for a "
         "single on-demand run. NOTE: a scheduled pipeline re-runs this exact "
         "command string, so the experiment name is fixed at build time -- see "
         "the comment on experiment_name below.",
)
parser.add_argument(
    "--timezone", default="UTC",
    help="Timezone for --cron (IANA name, e.g. America/New_York). Default UTC.",
)
parser.add_argument(
    "--parallel_runs", action="store_true",
    help="Allow a scheduled pipeline to start a new run while the previous one "
         "is still going. Default is off, which is what you want for training.",
)
args = parser.parse_args()

studio = Studio()
teamspace = studio.teamspace

run_id = f"{datetime.now():%Y%m%d-%H%M%S}"

# Flat, slash-free name -- same constraint as sweep_launcher.py: log_model
# registers the checkpoint under this name, recombined as
# "{owner}/{teamspace}/{name}", and the registry uses "/" only as its own
# delimiter. Keep it one segment.
#
# This is fixed when the pipeline is DEFINED, not per execution. For a scheduled
# pipeline every run therefore trains under the same experiment name (litlogger
# versions it) and the deployment always picks up the latest. That is the right
# behaviour for "retrain nightly, serve the current model"; if you need one
# distinct experiment per nightly run, template the name inside the step command
# instead (entrypoint is `sh -c`, so $(date ...) is available there).
experiment_name = f"{PROJECT}-pipe-{run_id}"

train_cmd = (
    f"python {REPO_ROOT}/training/train_movielens.py "
    f"--max_epochs {args.max_epochs} "
    f"--precision 16 "
    f"--logger_name {experiment_name} "
    f"--project {PROJECT} --workflow {WORKFLOW} "
    f"--experiment_group pipeline-{run_id} --experiment_name {experiment_name} "
    f"--sweep_id {run_id}"
)

# recommender_demo.py does `from server import _resolve_checkpoint` -- a sibling
# import that only resolves when serving/ is the cwd, so cd first. It reads
# EXPERIMENT_NAME to find the checkpoint the train step just registered.
eval_cmd = (
    f"cd {REPO_ROOT}/serving && "
    f"EXPERIMENT_NAME={experiment_name} python recommender_demo.py"
)

steps: List[Union[JobStep, DeploymentStep, MMTStep]] = [
    # Materialise the dataset once, on a CPU box, and fail the whole pipeline
    # here if the drive mount or the data is wrong -- rather than discovering it
    # after paying for a GPU to start.
    JobStep(
        name="data-prep",
        machine=Machine.CPU,
        command=(
            f"cd {REPO_ROOT} && python -c "
            f"\"from recsys.movielens_datamodule import MovieLens100K; "
            f"dm = MovieLens100K(); dm.prepare_data(); dm.setup(); "
            f"print('data ready:', dm.num_users, 'users', dm.num_items, 'items')\""
        ),
    ),
    JobStep(
        name="train",
        machine=Machine.from_str(args.train_machine),
        command=train_cmd,
    ),
    JobStep(
        name="eval",
        machine=Machine.from_str(args.eval_machine),
        command=eval_cmd,
    ),
]

if args.serve_studio and args.serve_image:
    parser.error("--serve_studio and --serve_image are mutually exclusive.")

# The serve step's source is independent of where training ran. Pointing it at
# another Studio (or image) is how you show "training lives in repo A, serving
# lives in repo B" -- the steps only share the model, passed by name through
# EXPERIMENT_NAME, not a filesystem.
serve_source = {}
if args.serve_studio:
    serve_source["studio"] = args.serve_studio
elif args.serve_image:
    serve_source["image"] = args.serve_image

# REPO_ROOT is this Studio's path. It is meaningless in another Studio or image,
# so refuse rather than launch a serve step that will fail on `No such file`.
serve_command = args.serve_command or f"python {REPO_ROOT}/serving/server.py"
if (args.serve_studio or args.serve_image) and not args.serve_command:
    parser.error(
        "--serve_studio/--serve_image need --serve_command too: the default "
        f"command points at '{REPO_ROOT}/serving/server.py', an absolute path "
        "in THIS Studio that won't exist in the one you're serving from."
    )

# Releases into a long-lived deployment rather than running a job: this is
# the always-on endpoint, so it has no "finished" state the way the three
# JobSteps above do.
steps.append(
    DeploymentReleaseStep(
        name="serve",
        machine=Machine.from_str(args.serve_machine),
        deployment_name=args.deployment_name,
        command=serve_command,
        ports=[8011],
        env={"EXPERIMENT_NAME": experiment_name},
        **serve_source,
    )
)

schedules = None
if args.cron:
    schedules = [
        Schedule(
            name=f"{PROJECT}-nightly",
            cron_expression=args.cron,
            timezone=args.timezone,
            parallel_runs=args.parallel_runs,
        )
    ]

pipeline_name = f"{PROJECT}-lifecycle-{run_id}"
# shared_filesystem=False is REQUIRED on Lightning Cloud baremetal. Pipeline
# defaults it to True, and the API then resolves a shared filesystem backend from
# the step's cloud account -- it only implements AWS (s3_folder) and GCP
# (gcs_folder), so a baremetal cluster raises
# `NotImplementedError: This cluster isn't support yet` at create time.
# We don't need it regardless: these steps hand off a model by registry name, not
# files on a shared disk (see "The steps share a model, not a filesystem" in the
# README).
pipeline = Pipeline(name=pipeline_name, studio=studio, shared_filesystem=False)
pipeline.run(steps=steps, schedules=schedules)

print(f"Launched pipeline '{pipeline_name}'")
print(f"  data-prep (CPU) -> train ({args.train_machine}) -> "
      f"eval ({args.eval_machine}) -> serve ({args.serve_machine})")
print(f"  experiment name : {experiment_name}")
print(f"  deployment      : {args.deployment_name}")
if schedules:
    print(f"  schedule        : '{args.cron}' ({args.timezone}), "
          f"parallel_runs={args.parallel_runs}")
else:
    print("  schedule        : none (single on-demand run)")
print(f"\nStep logs: lightning pipeline logs {pipeline_name} --teamspace "
      f"{teamspace.owner.name}/{teamspace.name}")
