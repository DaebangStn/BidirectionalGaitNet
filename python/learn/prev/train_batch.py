from eo.plot.exp_config import add_predefined_exp
from eo.learn.util import PROJECT_ROOT
from eo.learn.util import *
from eo.learn.nn.train import launch_experiment


NUM_WORKERS = 1

# Hardcoded list of experiments to train
EXP_NAMES = []
add_predefined_exp(EXP_NAMES)


def main():
    total_exp_num = len(EXP_NAMES)
    failed_exp = []
    
    if NUM_WORKERS > 1:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_exp = {
                executor.submit(launch_experiment, EXP_NAMES[exp_idx]): exp_idx
                for exp_idx in range(total_exp_num)
            }
            for future in as_completed(future_to_exp):
                try:
                    future.result()
                except Exception as e:
                    exp_idx = future_to_exp[future]
                    print(f"Experiment {exp_idx} ({EXP_NAMES[exp_idx]}) failed: {e}")
                    failed_exp.append(EXP_NAMES[exp_idx])
    else:
        for exp_idx in range(total_exp_num):
            launch_experiment(EXP_NAMES[exp_idx])

    if len(failed_exp) > 0:
        print(f"Failed experiments:")
        for exp_name in failed_exp:
            print(f"\'{exp_name}\', ")
        
    print("Playing alarm sound")
    os.system(f'aplay {PROJECT_ROOT / "data/res/alarm.wav"}')
    os.system(f'notify-send "Job has finished."')

if __name__ == "__main__":
    main()
