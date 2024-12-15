import pandas as pd
import json

system_prompt = "You are an AI trained to determine the relationship between two sentences. The possible relationships are: entailment (the first sentence implies the second), contradiction (the sentences contradict each other), or neutral (neither entailment nor contradiction)."


def label_to_str(label: int) -> str:
    if label == 0:
        return "ENTAILMENT"
    elif label == 1:
        return "NEUTRAL"
    elif label == 2:
        return "CONTRADICTION"

    raise ValueError(f"Invalid label: {label}")


# Load SNLI dataset splits
splits = {
    "test": "plain_text/test-00000-of-00001.parquet",
    "validation": "plain_text/validation-00000-of-00001.parquet",
    "train": "plain_text/train-00000-of-00001.parquet",
}

# Load and sample training data
train_df = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["train"])
train_sample_df = train_df.sample(n=10000, random_state=42)

# exclude rows with label -1
train_sample_df = train_sample_df[train_sample_df["label"] != -1]


# Create training conversations


def create_conversation(row):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            },
            {"role": "assistant", "content": label_to_str(row["label"])},
        ]
    }


train_conversations = [
    create_conversation(row) for _, row in train_sample_df.iterrows()
]

# Save training conversations
with open("snli_finetune.jsonl", "w") as f:
    for conv in train_conversations:
        f.write(json.dumps(conv) + "\n")
print(f"Saved {len(train_conversations)} conversations to snli_finetune.jsonl")

# Load and sample validation data
valid_df = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["validation"])
valid_sample_df = valid_df.sample(n=1000, random_state=42)

# exclude rows with label -1
valid_sample_df = valid_sample_df[valid_sample_df["label"] != -1]

# Create and save validation conversations
valid_conversations = [
    create_conversation(row) for _, row in valid_sample_df.iterrows()
]

with open("snli_validation.jsonl", "w") as f:
    for conv in valid_conversations:
        f.write(json.dumps(conv) + "\n")

print(f"Saved {len(valid_conversations)} conversations to snli_validation.jsonl")
