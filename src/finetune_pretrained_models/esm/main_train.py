import os
import argparse
import wandb
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

from src.utils.consts import MODEL_ID_TO_MODEL_NAME, BATCH_SIZE, FINETUNED_MODELS_OUTPUT_DIR, FINETUNE_NUMBER_OF_EPOCHS, RANDOM_STATE
from src.utils.read_fasta_utils import read_train_data, read_test_data
from src.finetune_pretrained_models.huggingface_utils import CalcMetricsOnTrainSetCallback, compute_metrics

WANDB_KEY = "64c3807b305e96e26550193f5860452b88d85999"
WANDB_PROJECT = "type3_secretion_signal_esm"
LEARNING_RATE = 2e-5


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id (number of layers)', type=str, required=True)
    return parser.parse_args()


def create_dataset(tokenizer, sequences, labels=None):
    tokenized = tokenizer(sequences)
    dataset = Dataset.from_dict(tokenized)

    if labels:
        dataset = dataset.add_column("labels", labels)

    return dataset


def train_model(model_name, tokenizer, train_dataset, validation_dataset, output_dir, run_name: str):
    # Set up Weights & Biases
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "end"  # Upload the final model to W&B at the end of training (after loading the best model)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,  # Only the best model is saved. Since load_best_model_at_end=True, it is possible that
                             # two checkpoints are saved: the last one and the best one (if they are different)
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=FINETUNE_NUMBER_OF_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="matthews_correlation",
        # use_cpu=True,
        report_to=["wandb"],  # reports to WandbCallback
        run_name=run_name,  # name of the W&B run
        seed=RANDOM_STATE
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

    trainer.train()

    return model


def main(model_id):
    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()

    model_name = MODEL_ID_TO_MODEL_NAME[model_id]
    run_name = f"{model_id}_train_batch{BATCH_SIZE}_lr{LEARNING_RATE}"
    output_dir = os.path.join(FINETUNED_MODELS_OUTPUT_DIR, model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = create_dataset(tokenizer, train_sequences, train_labels)
    validation_dataset = create_dataset(tokenizer, validation_sequences, validation_labels)

    model = train_model(model_name, tokenizer, train_dataset, validation_dataset, output_dir, run_name)

    # Save model
    model.save_pretrained(Path(output_dir) / "best_model")


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_id)
