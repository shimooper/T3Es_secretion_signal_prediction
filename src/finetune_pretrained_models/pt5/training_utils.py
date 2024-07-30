import sys
import os
import random

import torch
import numpy as np
from transformers import TrainingArguments, Trainer, set_seed
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from .classifier import PT5_classification_model
from .data_utils import prepare_dataset
from src.utils.consts import OUTPUTS_DIR
from src.finetune_pretrained_models.huggingface_utils import CalcMetricsOnTrainSetCallback, compute_metrics


WANDB_KEY = "64c3807b305e96e26550193f5860452b88d85999"
WANDB_PROJECT = "type3_secretion_signal_pt5"

# Deepspeed config for optimizer CPU offload
ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}


# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


# Main training function
def train_per_protein(
        train_df,  # training data
        valid_df,  # validation data
        num_labels=1,  # 1 for regression, >1 for classification

        # effective training batch size is batch * accum
        # we recommend an effective batch size of 8
        batch=4,  # for training
        accum=2,  # gradient accumulation

        val_batch=16,  # batch size for evaluation
        epochs=10,  # training epochs
        lr=3e-4,  # recommended learning rate
        seed=42,  # random seed
        deepspeed=True,  # if gpu is large enough disable deepspeed for training speedup
        half_precision=False,  # enable mixed precision training
):
    # Disable deepspeed if we run on windows
    deepspeed = deepspeed and os.name != 'nt'

    # Set gpu device
    print(f"$CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Set all random seeds
    set_seeds(seed)

    # load model
    checkpoint, model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=half_precision)

    # Set up Weights & Biases
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "end"  # Upload the final model to W&B at the end of training (after loading the best model)
    os.environ['WANDB_LOG_DIR'] = os.path.join(OUTPUTS_DIR, 'pt5_finetune')
    run_name = f"{checkpoint}_train_batch{batch * accum}_lr{lr}"
    run_output_dir = os.path.join(OUTPUTS_DIR, 'pt5_finetune', run_name)

    # Create Datasets
    train_set = prepare_dataset(train_df, tokenizer)
    valid_set = prepare_dataset(valid_df, tokenizer)

    # Huggingface Trainer arguments
    args = TrainingArguments(
        output_dir=run_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="matthews_correlation",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        report_to=["wandb"],
        run_name=run_name,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
        fp16=half_precision,
    )

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CalcMetricsOnTrainSetCallback(trainer))

    # Train model
    trainer.train()

    # Save model (only the finetuned parameters)
    save_model(model, os.path.join(run_output_dir, "PT5_GB1_finetuned.pth"))

    return tokenizer, model


def save_model(model, filepath):
    # Saves all parameters that were changed during finetuning

    # Create a dictionary to hold the non-frozen parameters
    non_frozen_params = {}

    # Iterate through all the model parameters
    for param_name, param in model.named_parameters():
        # If the parameter has requires_grad=True, add it to the dictionary
        if param.requires_grad:
            non_frozen_params[param_name] = param

    # Save only the finetuned parameters
    torch.save(non_frozen_params, filepath)
