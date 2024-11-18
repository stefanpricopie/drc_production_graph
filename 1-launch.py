#!/usr/bin/env python3
import datetime
import os
import platform
import subprocess

completed_process = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], check=True,
                                   stdout=subprocess.PIPE, universal_newlines=True)
# Strip newline character at the end
latest_git_hash = completed_process.stdout.strip()
if latest_git_hash is None:
    raise ValueError("Could not obtain the latest git hash")

# Assuming this remains constant as in your bash script
EMAIL = "stefan.pricopie@postgrad.manchester.ac.uk"
N_RUNS = 100


# Get the current date in YYYYMMDD format
current_date = datetime.datetime.now().strftime("%Y%m%d")

# Assuming BINDIR remains constant as in your bash script
BINDIR = os.path.dirname(os.path.abspath(__file__))
# Modify OUTDIR to include both the current date and the latest git hash
OUTDIR = f"results/{current_date}_{latest_git_hash}"
# Ensure the directory exists
os.makedirs(OUTDIR, exist_ok=True)


def qsub_job(runner, configs, jobname, memory, ncores):
    # Generate the Config array
    config_array = "configs=(" + " \\\n         \"" + "\" \\\n         \"".join(configs) + "\")"

    # Set memory flag based on the input
    if memory is None:
        memory_flag = ""
    elif memory == 512:
        # For 32GB per core
        memory_flag = "#$ -l mem512"
    elif memory == 1500:
        # 1.5TB RAM = 48GB per core, max 32 cores (Skylake CPU). 7 nodes.
        memory_flag = "#$ -l mem1500"
    elif memory == 2000:
        # 2TB RAM   = 64GB per core, max 32 cores (Icelake CPU), 8TB SSD /tmp. 10 nodes.
        memory_flag = "#$ -l mem2000"
    else:
        raise ValueError(f"Memory value {memory} not recognised")

    # Set the number of cores
    if ncores == 1:
        ncores_flag = ""
        set_threads_cmd = ""
    elif isinstance(ncores, int) and ncores > 1:
        ncores_flag = f"#$ -pe smp.pe {ncores}"
        set_threads_cmd = "export OMP_NUM_THREADS=$NSLOTS"
    else:
        raise ValueError(f"Number of cores {ncores} not recognised")

    cmd = f"""#!/bin/bash --login
#$ -t 1-{len(configs)}  # Using N_RUNS to specify task range
#$ -N {jobname}
{ncores_flag}
# -l s_rt=06:00:00
{memory_flag}
# -M {EMAIL}
# -m as
#$ -cwd
#$ -j y
#$ -o {OUTDIR}

{config_array}

# Use SGE_TASK_ID to access the specific configuration
CONFIG_INDEX=$(($SGE_TASK_ID - 1))  # Arrays are 0-indexed
CONFIG=${{configs[$CONFIG_INDEX]}}

# Set the number of threads to match the number of requested cores
{set_threads_cmd}

echo "{runner} $CONFIG"
echo "Job: $JOB_ID, Task: $SGE_TASK_ID, Config: $CONFIG"

{BINDIR}/{runner} $CONFIG
"""
    with subprocess.Popen(["qsub", "-v", "PATH"], stdin=subprocess.PIPE) as proc:
        proc.communicate(input=cmd.encode())


def add_config(configurations, problem, algo, b_exp, c_base, c_synth, outdir=OUTDIR):
    begin_seed = 0
    for seed in range(begin_seed, begin_seed + N_RUNS):
        # Initialize base config string
        config = (f"{problem} {algo} --b_exp {b_exp} --c_base {c_base} --c_synth {c_synth} --seed {seed} --output {outdir}")

        configurations.append(config)


def run_local(runner, configs):
    for i, config in enumerate(configs):
        print(f"\nRun #{i} {config}")
        cmd = [runner]
        cmd.extend(config.split())
        subprocess.run(cmd)


def run_job(job, job_name, memory=None, ncores=1):
    runner = "malaria.py"       # Your Python script for running a single experiment
    configurations = job()  # Generate the configurations for the job

    if platform.system() == "Linux":
        # assert N_RUNS == 50, "N_RUNS must be 50 for cluster runs"
        # split configurations into jobnames and configs
        qsub_job(runner=runner, configs=configurations,
                 jobname=f"{job_name}{memory if memory is not None else ''}",
                 memory=memory, ncores=ncores)
    elif platform.system() == "Darwin":  # macOS is identified as 'Darwin'
        run_local(runner=f"{os.getcwd()}/{runner}", configs=configurations)


def run(problems, algos, c_bases, c_synths):
    configurations = []

    for problem in problems:
        for algo in algos:
            for c_base in c_bases:
                for c_synth in c_synths:
                    add_config(configurations=configurations, problem=problem, algo=algo, b_exp=2,
                               c_base=c_base, c_synth=c_synth)

    return configurations


if __name__ == "__main__":
    problems = [
        'master_16087_graph_processed',
        # 'subgraph_1000_seed0',
        # 'subgraph_5000_seed1',
        # 'subgraph_10000_seed2',
    ]
    c_bases = [1, 10]
    c_synths = [1, 10]

    run_rs = lambda: run(problems=problems, algos=['rs', 'lc5', 'lc20'], c_bases=c_bases, c_synths=c_synths)
    run_bo = lambda: run(problems=problems, algos=['bo'], c_bases=c_bases, c_synths=c_synths)
    run_bopu = lambda: run(problems=problems, algos=['bopu'], c_bases=c_bases, c_synths=c_synths)

    job_mem = [
        {'job': run_rs, 'job_name': 'rs'},
        {'job': run_bo, 'job_name': 'bo', 'memory': 1500},
        {'job': run_bopu, 'job_name': 'bopu', 'memory': 2000},
    ]

    for job_kwargs in job_mem:
        run_job(**job_kwargs)
