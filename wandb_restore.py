# To restore the last and best model from a given wandb run
# It will save in the same location

import os
import shutil
import wandb
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--path', type=str)

args = parser.parse_args()


wandb.init(id=args.id, project="fsl_ssl", resume=True)

wandb_path = wandb.restore(args.path)

checkpoint_dir = os.path.split(args.path)[0]

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

model_epoch = torch.load(wandb_path.name)["epoch"]

restore_path = os.path.join(checkpoint_dir, f"{model_epoch}.tar")

shutil.move(wandb_path.name, restore_path)

print("Checkpoint restored at", restore_path)

print("The model's epoch is", model_epoch)
