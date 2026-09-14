# batch_inference_pipeline.py
#
# Daily batch inference on a cron schedule: one step, one Schedule.
#
# This is deliberately separate from lifecycle_pipeline.py. That one is the
# training lifecycle (prep -> train -> eval -> serve) and runs when you ship a
# new model. This one takes whatever model is current and scores the catalogue on
# a timetable. Different cadence, different trigger, different machine size --
# so, different pipeline.
#
# Note what this is NOT: a deployment. A deployment is an always-on endpoint that
# waits for requests and autoscales with traffic. This wakes on a clock, scores
# everything, writes a Parquet artifact and exits, costing nothing in between.
# Both serve the same model from the registry; neither substitutes for the other.

import argparse
import pathlib
from datetime import datetime

from lightning_sdk import Machine, Studio
from typing import List, Union

from lightning_sdk.pipeline import DeploymentStep, JobStep, MMTStep, Pipeline, Schedule

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Schedule (or run once) batch inference over all users."
)
parser.add_argument(
    "--experiment_name", default=None,
    help="Experiment whose checkpoint to score with, resolved in THIS teamspace. "
         "Omit to let server.py's _resolve_checkpoint fall back to CHECKPOINT_NAME "
         "or its seeded default -- which won't exist in a fresh teamspace, so in "
         "practice you want to pass this.",
)
parser.add_argument(
    "--machine", default="T4",
    help="Machine for the inference step (default T4). Two-tower scoring is one "
         "matmul over the embedding tables, so this does not need a large GPU.",
)
parser.add_argument(
    "--top_k", type=int, default=10,
    help="How many recommendations to keep per user (default 10).",
)
parser.add_argument(
    "--cron", default=None,
    help="Cron expression, e.g. '0 6 * * *' for 06:00 daily. Omit to run the "
         "pipeline once now instead of scheduling it.",
)
parser.add_argument(
    "--timezone", default="UTC",
    help="IANA timezone for --cron, e.g. America/New_York. Default UTC.",
)
args = parser.parse_args()

studio = Studio()
teamspace = studio.teamspace

env = {"TOP_K": str(args.top_k)}
if args.experiment_name:
    env["EXPERIMENT_NAME"] = args.experiment_name

# cd into serving/: batch_inference.py does `from server import _resolve_checkpoint`,
# a sibling import that only resolves with serving/ as the cwd.
command = f"cd {REPO_ROOT}/serving && python batch_inference.py"

run_id = f"{datetime.now():%Y%m%d-%H%M%S}"
pipeline_name = f"ml-100k-batch-inference-{run_id}"

steps: List[Union[JobStep, DeploymentStep, MMTStep]] = [
    JobStep(
        name="batch-inference",
        machine=Machine.from_str(args.machine),
        command=command,
        env=env,
    )
]

schedules = None
if args.cron:
    schedules = [
        Schedule(
            name="ml-100k-nightly-inference",
            cron_expression=args.cron,
            timezone=args.timezone,
            # Inference is idempotent, but overlapping runs would still double the
            # bill for no extra output. Keep them serialized.
            parallel_runs=False,
        )
    ]

# shared_filesystem=False is REQUIRED on Lightning Cloud baremetal. Pipeline
# defaults it to True, and the API then resolves a shared filesystem backend from
# the step's cloud account -- it only implements AWS (s3_folder) and GCP
# (gcs_folder), so a baremetal cluster raises
# `NotImplementedError: This cluster isn't support yet` at create time.
# We don't need it regardless: these steps hand off a model by registry name, not
# files on a shared disk.
pipeline = Pipeline(name=pipeline_name, studio=studio, shared_filesystem=False)
pipeline.run(steps=steps, schedules=schedules)

print(f"Pipeline '{pipeline_name}' created")
print(f"  step     : batch-inference on {args.machine}")
print(f"  model    : {args.experiment_name or '(server.py default resolution)'}")
print(f"  top_k    : {args.top_k}")
if schedules:
    print(f"  schedule : '{args.cron}' ({args.timezone})")
    print("\nThis now runs on a recurring schedule and bills each run. Stop it with:")
    print(f"    python -c \"from lightning_sdk.pipeline import Pipeline; "
          f"Pipeline(name='{pipeline_name}').stop()\"")
else:
    print("  schedule : none (single run)")
print(f"\nLogs: lightning pipeline logs {pipeline_name} --teamspace "
      f"{teamspace.owner.name}/{teamspace.name}")
print("Output: recommendations.parquet, collected as a job artifact --")
print(f"    lightning ls -r lit://{teamspace.owner.name}/{teamspace.name}/jobs/")
