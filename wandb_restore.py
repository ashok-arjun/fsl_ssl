# This script will ask for a wandb run ID and restore path
import wandb

RUN_ID = "27wluxlz"
PATH = "ckpts/dogs/_resnet18_baseline_aug_tracking_lr0.0010/last_model.tar"

wandb.init(id=RUN_ID, project="fsl_ssl", resume=True)
wandb.restore(PATH)