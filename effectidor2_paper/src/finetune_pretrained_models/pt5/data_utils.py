from datasets import Dataset


def create_dataset(tokenizer, seqs, labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset


def prepare_dataset(df, tokenizer):
    df["sequence"] = df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    df['sequence'] = df.apply(lambda row: " ".join(row["sequence"]), axis=1)
    dataset = create_dataset(tokenizer, list(df['sequence']), list(df['label']))
    return dataset
