import wandb
import os
import argparse

def main(args):
    api = wandb.Api()
    print(args.sweep)
    sweep = api.sweep(args.sweep)
    if len(sweep.runs) >=  args.max_runs:
        command = f'wandb sweep --stop {sweep.sweep_id}'
        os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True)
    parser.add_argument('--max_runs', type=int, default=50)
    args  = parser.parse_args()
    main(args)
