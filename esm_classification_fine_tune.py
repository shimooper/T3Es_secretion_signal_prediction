import os
import argparse
from copy import deepcopy
from timeit import default_timer as timer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import wandb
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from evaluate import load

from consts import (OUTPUTS_DIR, MODEL_ID_TO_MODEL_NAME,
                    FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                    FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                    FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE,
                    BATCH_SIZE)
from utils import read_fasta_file

NUMBER_OF_EPOCHS = 3
RANDOM_STATE = 42
WANDB_KEY = "64c3807b305e96e26550193f5860452b88d85999"
WANDB_PROJECT = "type3_secretion_signal"

mcc_metric = load("matthews_correlation")


# Used to log metrics on train set during training at the end of each epoch (by default metrics are calculated only on evaluation/validation set)
# Solution taken from - https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
# The calculated loss here appears in the logs as "train/train_loss" and is the loss of the train_set at the end of each epoch.
# This differs from the automatically logged "train/loss" which is the average loss of all steps during the epoch.
class CalcMetricsOnTrainSetCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_num_of_layers', help='The pretrained model number of layers', type=str, required=True)
    return parser.parse_args()


def read_train_data():
    positive_train = read_fasta_file(FIXED_POSITIVE_TRAIN_FILE)
    negative_train = read_fasta_file(FIXED_NEGATIVE_TRAIN_FILE)

    all_train_labels = [1] * len(positive_train) + [0] * len(negative_train)
    all_train_sequences = positive_train + negative_train

    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
        all_train_sequences, all_train_labels, test_size=0.25, random_state=RANDOM_STATE, shuffle=True)

    return train_sequences, validation_sequences, train_labels, validation_labels


def read_test_data(split):
    if split == 'test':
        positive_test = read_fasta_file(FIXED_POSITIVE_TEST_FILE)
        negative_test = read_fasta_file(FIXED_NEGATIVE_TEST_FILE)
    elif split == 'xantomonas':
        positive_test = read_fasta_file(FIXED_POSITIVE_XANTOMONAS_FILE)
        negative_test = read_fasta_file(FIXED_NEGATIVE_XANTOMONAS_FILE)
    else:
        raise ValueError(f"split must be one of ['test', 'xantomonas'], got {split}")

    test_labels = [1] * len(positive_test) + [0] * len(negative_test)
    test_sequences = positive_test + negative_test

    return test_sequences, test_labels


def create_dataset(tokenizer, sequences, labels=None):
    tokenized = tokenizer(sequences)
    dataset = Dataset.from_dict(tokenized)

    if labels:
        dataset = dataset.add_column("labels", labels)

    return dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions, axis=1)
    predictions_scores = predictions[:, 1]

    scores = mcc_metric.compute(predictions=predictions_labels, references=labels)
    scores['auprc'] = average_precision_score(y_true=labels, y_score=predictions_scores)
    return scores


def train_model(model_checkpoint, tokenizer, train_dataset, validation_dataset, output_dir, run_name: str):
    # Set up Weights & Biases
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "end"  # Upload the final model to W&B at the end of training (after loading the best model)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    training_args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,  # Only the best model is saved. Since load_best_model_at_end=True, it is possible that
                             # two checkpoints are saved: the last one and the best one (if they are different)
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUMBER_OF_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="matthews_correlation",
        # use_cpu=True,
        report_to=["wandb", "tensorboard"],  # reports to TensorBoardCallback + WandbCallback
        logging_dir=os.path.join(OUTPUTS_DIR, 'esm_finetune_tb_logs', run_name),  # TensorBoard log directory
        run_name=run_name,  # name of the W&B run
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=validation_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(CalcMetricsOnTrainSetCallback(trainer))

    train_results = trainer.train()

    # The train_loss metric logged here is the average loss across all steps during training.
    # The epoch logged here is the one that resulted in the best model according to metric_for_best_model (mcc) on the evaluation set.
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    return trainer


def test_on_test_data(trainer: Trainer, tokenizer, split):
    # First, estimate time of embedding and prediction
    start_test_time = timer()

    test_sequences, test_labels = read_test_data(split=split)
    test_dataset_without_labels = create_dataset(tokenizer, test_sequences)
    test_predictions = trainer.predict(test_dataset_without_labels, metric_key_prefix=split)

    end_test_time = timer()
    elapsed_time = end_test_time - start_test_time

    # Now, calculate and log the metrics
    test_dataset = create_dataset(tokenizer, test_sequences, test_labels)
    test_results = trainer.predict(test_dataset, metric_key_prefix=split)

    trainer.log_metrics(split, test_results.metrics)
    trainer.save_metrics(split, test_results.metrics)

    return test_results, elapsed_time


def main(model_num_of_layers):
    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()

    model_checkpoint = MODEL_ID_TO_MODEL_NAME[model_num_of_layers]
    run_name = f"{model_checkpoint.split('/')[-1]}-{NUMBER_OF_EPOCHS}-epochs"
    output_dir = os.path.join(OUTPUTS_DIR, 'esm_finetune_runs', run_name)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = create_dataset(tokenizer, train_sequences, train_labels)
    validation_dataset = create_dataset(tokenizer, validation_sequences, validation_labels)

    trainer = train_model(model_checkpoint, tokenizer, train_dataset, validation_dataset, output_dir, run_name)

    # Evaluate the best model on the train set and on the validation set
    train_set_results = trainer.predict(train_dataset, metric_key_prefix="train_set_predictions")
    trainer.log_metrics("train_set_predictions", train_set_results.metrics)
    trainer.save_metrics("train_set_predictions", train_set_results.metrics)

    validation_results = trainer.evaluate()
    trainer.log_metrics("eval", validation_results)
    trainer.save_metrics("eval", validation_results)

    # Evaluate the best model on the test set and on the Xantomonas set
    test_results, test_elapsed_time = test_on_test_data(trainer, tokenizer, 'test')
    xantomonas_results, xantomonas_elapsed_time = test_on_test_data(trainer, tokenizer, 'xantomonas')

    # Save the results
    results_df = pd.DataFrame({
        'best_epoch': [validation_results["epoch"]],
        'train_mcc': [train_set_results.metrics["train_set_predictions_matthews_correlation"]], 'train_auprc': [train_set_results.metrics["train_set_predictions_auprc"]],
        'validation_mcc': [validation_results["eval_matthews_correlation"]], 'validation_auprc': [validation_results["eval_auprc"]],
        'test_mcc': [test_results.metrics["test_matthews_correlation"]], 'test_auprc': [test_results.metrics["test_auprc"]], 'test_elapsed_time': [test_elapsed_time],
        'xantomonas_mcc': [xantomonas_results.metrics["xantomonas_matthews_correlation"]], 'xantomonas_auprc': [xantomonas_results.metrics["xantomonas_auprc"]], 'xantomonas_elapsed_time': [xantomonas_elapsed_time]
    })
    results_df.to_csv(os.path.join(output_dir, 'esm_finetune_results.csv'), index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_num_of_layers)
