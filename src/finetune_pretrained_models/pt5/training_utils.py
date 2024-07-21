
import os
import random

import torch
import numpy as np
from transformers import TrainingArguments, Trainer, set_seed
import wandb

from evaluate import load
from datasets import Dataset

from classifier import PT5_classification_model
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


# Dataset creation
def create_dataset(tokenizer, seqs, labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset


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
        mixed=False,  # enable mixed precision training
        gpu=1, # gpu selection (1 for first gpu)
):
    # Disable deepspeed if we run on windows
    deepspeed = deepspeed and os.name != 'nt'

    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu - 1)

    # Set all random seeds
    set_seeds(seed)

    # load model
    checkpoint, model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=torch.cuda.is_available())

    # Set up Weights & Biases
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "end"  # Upload the final model to W&B at the end of training (after loading the best model)
    run_name = f"{checkpoint}-train_batch{batch * accum}-lr{lr}"

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["sequence"] = train_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    valid_df["sequence"] = valid_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    # Add spaces between each amino acid for PT5 to correctly use them
    train_df['sequence'] = train_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
    valid_df['sequence'] = valid_df.apply(lambda row: " ".join(row["sequence"]), axis=1)

    # Create Datasets
    train_set = create_dataset(tokenizer, list(train_df['sequence']), list(train_df['label']))
    valid_set = create_dataset(tokenizer, list(valid_df['sequence']), list(valid_df['label']))

    # Huggingface Trainer arguments
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUTS_DIR, 'pt5_finetune_runs', run_name),
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
        fp16=mixed,
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
    save_model(model, os.path.join(OUTPUTS_DIR, 'pt5_finetune_runs', run_name, "PT5_GB1_finetuned.pth"))

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


def load_model(filepath, num_labels=1, mixed=False):
    # Creates a new PT5 model and loads the finetuned weights from a file

    # load a new model
    checkpoint, model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=mixed)

    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model
