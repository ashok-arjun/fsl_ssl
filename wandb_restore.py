# To restore the last and best model from a given wandb run
# It will save in the same location

import os
import shutil
import wandb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--path', type=str)

args = parser.parse_args()

wandb.init(id=args.id, project="fsl_ssl", resume=True)

wandb_path = wandb.restore(args.path)

if not os.path.exists(os.path.split(args.path)[0]):
    os.makedirs(os.path.split(args.path)[0], exist_ok=True)

shutil.move(wandb_path.name, args.path)

print("Checkpoint restored in the given path")