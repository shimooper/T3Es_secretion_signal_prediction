import os

from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import wandb
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from evaluate import load

from consts import (OUTPUTS_DIR, FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE, FIXED_POSITIVE_TEST_FILE,
                    FIXED_NEGATIVE_TEST_FILE, FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE)
from utils import CustomCallback

# RUN_NAME = "large-model-with-train-metrics"
RUN_NAME = "small-model-test"
NUMBER_OF_EPOCHS = 10
RANDOM_STATE = 42
WANDB_KEY = "64c3807b305e96e26550193f5860452b88d85999"
WANDB_PROJECT = "type3_secretion_signal"

# choose model checkpoint (one fo the ESM-2 models)
# model_checkpoint = "facebook/esm2_t33_650M_UR50D"
model_checkpoint = "facebook/esm2_t6_8M_UR50D"
mcc_metric = load("matthews_correlation")


def read_fasta_file(file_path):
    sequences = []
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        sequences.append(str(seq_record.seq))
    return sequences


def preprocess_data():
    positive_train = read_fasta_file(FIXED_POSITIVE_TRAIN_FILE)
    negative_train = read_fasta_file(FIXED_NEGATIVE_TRAIN_FILE)
    positive_test = read_fasta_file(FIXED_POSITIVE_TEST_FILE)
    negative_test = read_fasta_file(FIXED_NEGATIVE_TEST_FILE)

    all_train_labels = [1] * len(positive_train) + [0] * len(negative_train)
    test_labels = [1] * len(positive_test) + [0] * len(negative_test)

    all_train_sequences = positive_train + negative_train
    test_sequences = positive_test + negative_test

    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
        all_train_sequences, all_train_labels, test_size=0.25, random_state=RANDOM_STATE, shuffle=True)

    return train_sequences, validation_sequences, train_labels, test_sequences, validation_labels, test_labels


def create_datasets(tokenizer, train_sequences, validation_sequences, test_sequences, train_labels, validation_labels, test_labels):
    train_tokenized = tokenizer(train_sequences)
    validation_tokenized = tokenizer(validation_sequences)
    test_tokenized = tokenizer(test_sequences)

    train_dataset = Dataset.from_dict(train_tokenized)
    validation_dataset = Dataset.from_dict(validation_tokenized)
    test_dataset = Dataset.from_dict(test_tokenized)

    train_dataset = train_dataset.add_column("labels", train_labels)
    validation_dataset = validation_dataset.add_column("labels", validation_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    return train_dataset, validation_dataset, test_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions, axis=1)
    predictions_scores = predictions[:, 1]

    scores = mcc_metric.compute(predictions=predictions_labels, references=labels)
    scores['auprc'] = average_precision_score(y_true=labels, y_score=predictions_scores)
    return scores


def train_model(train_dataset, validation_dataset, test_dataset, tokenizer):
    # Set up Weights & Biases
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "end"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    run_name = f"{model_checkpoint.split('/')[-1]}-{RUN_NAME}"
    batch_size = 8
    training_args = TrainingArguments(
        os.path.join(OUTPUTS_DIR, 'runs', run_name),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=NUMBER_OF_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="matthews_correlation",
        # use_cpu=True,
        report_to=["wandb", "tensorboard"],  # reports to TensorBoardCallback + WandbCallback
        logging_dir=os.path.join(OUTPUTS_DIR, 'logs', run_name),  # TensorBoard log directory
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
    trainer.add_callback(CustomCallback(trainer))

    train_results = trainer.train()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    validation_results = trainer.evaluate()
    trainer.log_metrics("eval", validation_results)
    trainer.save_metrics("eval", validation_results)

    test_results = trainer.predict(test_dataset)
    trainer.log_metrics("test", test_results.metrics)
    trainer.save_metrics("test", test_results.metrics)

    wandb.finish()


def main():
    train_sequences, validation_sequences, train_labels, test_sequences, validation_labels, test_labels = preprocess_data()

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, validation_dataset, test_dataset = create_datasets(tokenizer, train_sequences, validation_sequences,
                                                                      test_sequences, train_labels, validation_labels,
                                                                      test_labels)

    train_model(train_dataset, validation_dataset, test_dataset, tokenizer)


if __name__ == "__main__":
    main()
