
import os
import random

import torch
import numpy as np
from transformers import TrainingArguments, Trainer, set_seed

from evaluate import load
from datasets import Dataset

from classifier import PT5_classification_model


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


# Main training fuction
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
        gpu=1):  # gpu selection (1 for first gpu)

    # Disable deepspeed if we run on windows
    deepspeed = deepspeed and os.name != 'nt'

    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu - 1)

    # Set all random seeds
    set_seeds(seed)

    # load model
    model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=torch.cuda.is_available())

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
        "./",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
        fp16=mixed,
    )

    # Metric definition for validation data
    def compute_metrics(eval_pred):
        if num_labels > 1:  # for classification
            metric = load("accuracy")
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
        else:  # for regression
            metric = load("spearmanr")
            predictions, labels = eval_pred

        return metric.compute(predictions=predictions, references=labels)

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    return tokenizer, model, trainer.state.log_history


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
    model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=mixed)

    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model
