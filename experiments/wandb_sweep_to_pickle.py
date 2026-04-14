"""
wandb_sweep_to_pickle.py

helper file to save out wandb sweeps to pickle files

Usage:

name of sweep: user/project/sweep_id
name of pickle file: a helpful name for the pickle file
"""

from tqdm import tqdm
import pandas as pd
import argparse
import wandb
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True, help="Wandb sweep id")
    parser.add_argument("--pickle_name", type=str, required=True, help="Name of pickle file")
    parser.add_argument("--date", type=str, required=True, help="Directory to save files")
    args = parser.parse_args()

    api = wandb.Api()
    project = "ictanoh_team/broydeer"
    saved_files = os.path.join(os.path.dirname(__file__), "saved_files")
    os.makedirs(saved_files, exist_ok=True)
    sweep = api.sweep(f"{project}/{args.sweep_id}")
    dataframes = []
    for run in tqdm(sweep.runs):
        # Debug: print config for every run
        print(f"\n=== run {run.name} (state={run.state}) ===")
        print(f"  config keys: {list(run.config.keys())}")
        print(f"  summary keys: {list(run.summary.keys())}")
        print(f"  raw config: {run.rawconfig}")
        history = []
        for row in run.scan_history():  # important to get all the run information out
            history.append(row)
        run_data = pd.DataFrame(history)
        # Flatten config from rawconfig — skip internal wandb keys
        for key, value in run.rawconfig.items():
            if key.startswith("_"):
                continue
            run_data[key] = value
        dataframes.append(run_data)

    df = pd.concat(dataframes, ignore_index=True)
    print(df.shape)

    df.to_pickle(f"{saved_files}/{args.date}_{args.pickle_name}_{args.sweep_id}.pkl")
