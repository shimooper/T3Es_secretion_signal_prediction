from datasets import load_dataset
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer, PreTrainedTokenizerFast

# Load a dataset
dataset = load_dataset("imdb")

# Define a simple character-level tokenizer
class CustomCharTokenizer:
    def __init__(self, vocab):
        self.vocab = {ch: i for i, ch in enumerate(vocab)}
        self.id2char = {i: ch for i, ch in enumerate(vocab)}

    def __call__(self, text):
        return {"input_ids": [self.vocab[char] for char in text if char in self.vocab]}

# Initialize the tokenizer
vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?. ")
tokenizer = PreTrainedTokenizerFast(tokenizer_object=CustomCharTokenizer(vocab))

# Define custom model configuration
config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=512
)

# Initialize the model
model = BertForSequenceClassification(config)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()