# sweep_launcher.py
#
# Launches a lr x batch_size grid as separate Lightning jobs, all logging into
# ONE experiment in the experiment manager so they can be compared side by side
# (rather than 9 unrelated experiments). See training/README.md, "Grouping
# experiments" for how this works.

from lightning_sdk import Studio, Machine

studio = Studio()
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes    = [128, 256, 512]

grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

# Shared across every job in this sweep: same --logger_name means all 9 runs
# become versions of ONE experiment, not 9 separate experiments. lr/batch_size
# are still logged as metadata per run (see train_movielens.py), so each
# version is distinguishable when comparing results.
EXPERIMENT_GROUP = "ml100k-sweep"

for idx, (lr, bs) in enumerate(grid):
    # Unique per-job name -- only used to tell jobs apart in the Jobs UI.
    job_name = f"ml100k-sweep-{idx}-lr{lr}-bs{bs}"

    cmd      = (
        f"python training/train_movielens.py "
        f"--lr {lr} "
        f"--batch_size {bs} "
        f"--precision 16 "
        f"--max_epochs 25 "
        f"--logger_name {EXPERIMENT_GROUP}"
    )

    # Launch from current machine as default
    # job_plugin.run(command=cmd, name=job_name)  # type: ignore[call-arg]

    # Specify machine to launch experiment jobs from
    job_plugin.run(command=cmd, machine=Machine.H100, name=job_name)  # type: ignore[call-arg]

    print(f"Launched {job_name} → `{cmd}`")

print(f"\nAll 9 runs grouped under experiment '{EXPERIMENT_GROUP}' -- compare their "
      f"versions in the experiment manager to pick the best config, then run that "
      f"config's full training with its own --logger_name.")