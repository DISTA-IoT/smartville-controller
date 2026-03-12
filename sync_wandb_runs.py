import subprocess
import os

def sync_wandb_folders(root_dir):
    # Find all folders starting with /pox/pox/smartController/wandb/run-XXXXX
    folders = [f.path for f in os.scandir(root_dir) if f.is_dir() and 'run-' in f.path]

    # Loop through each folder and sync it with wandb
    for folder in folders:
        print(f"Syncing {folder}")
        subprocess.run(['wandb', 'sync', folder])

sync_wandb_folders('/pox/pox/smartController/wandb')