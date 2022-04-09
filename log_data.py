import os
from tabulate import tabulate
import wandb



def model_configs(cfg):
    
    header = f"Training {os.path.basename(cfg.data_dir)} Data Set"
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    config_table = [["Model",cfg.model],
                    ["Batch Size",cfg.batch_size],
                    ["Image Size",cfg.image_size],
                    ["Epochs",cfg.epochs],
                    ["Learning Rate",cfg.learning_rate],
                    ["Use Pretrained",cfg.pretrained],
                    ["Momentum",cfg.momentum],
                    ["Weight Decay",cfg.weight_decay],
                    ["Loss Function",cfg.loss_criterion],
                    ["Optimizer",cfg.optimizer],
                    ["Learning Rate Decay",cfg.lr_scheduler],
                    ["Decay epochs ",cfg.steps],
                    ["Decay Learning Rate factor ",cfg.optimizer]]
    print(tabulate(config_table,tablefmt="fancy_grid"))



def wandb_logs(phase,epoch_loss,epoch_acc):

    if phase == 'train':
        wandb.log({"train_loss": epoch_loss,
                "train_accuracy" : epoch_acc
                        })

    elif phase == 'val':
        wandb.log({"val_loss": epoch_loss,
                    "val_accuracy" : epoch_acc
                                })


def get_accuracy(phase,epoch_loss,epoch_acc):
    header = f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}'
    print('-' * len(header))
    print(header)
    print('-' * len(header))
