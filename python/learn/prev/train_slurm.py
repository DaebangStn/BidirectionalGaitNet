import submitit
from eo.learn.nn.train import launch_experiment
from eo.plot.exp_config import add_predefined_exp

# Hardcoded list of experiments to train
EXP_NAMES = []
add_predefined_exp(EXP_NAMES)

CONFIG_PATH = "config/regression_soft.yaml"

def main():
    executor = submitit.AutoExecutor(folder="slurm_logs")
    executor.update_parameters(
        gpus_per_node=1,
        tasks_per_node=1,
        timeout_min=1000,
    )
    jobs = [executor.submit(launch_experiment, exp_name) for exp_name in EXP_NAMES]
    for job in jobs:
        print(f"Submitted job: {job}")
    
    for job in jobs:
        job.wait()
        print(f"Job {job} finished")
    
    
if __name__ == "__main__":
    main()
