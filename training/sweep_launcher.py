# sweep_launcher.py

from lightning_sdk import Studio, Machine

studio = Studio()
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes    = [128, 256, 512]

grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

for idx, (lr, bs) in enumerate(grid):
    job_name = f"ml100k-sweep-{idx}-lr{lr}-bs{bs}"

    cmd      = (
        f"python training/train_movielens.py "
        f"--lr {lr} "
        f"--batch_size {bs} "
        f"--precision 16 "
        f"--max_epochs 25 "
        f"--logger_name {job_name}"
    )

    # Launch from current machine as default
    # job_plugin.run(command=cmd, name=job_name)  # type: ignore[call-arg]

    # Specify machine to launch experiment jobs from
    job_plugin.run(command=cmd, machine=Machine.H100, name=job_name)  # type: ignore[call-arg]

    print(f"Launched {job_name} → `{cmd}`")