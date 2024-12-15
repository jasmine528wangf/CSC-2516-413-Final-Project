from openai import OpenAI
import pandas as pd
import tqdm

OPENAI_API_KEY = "..." 

client = OpenAI(api_key=OPENAI_API_KEY)

def label_to_str(label: int) -> str:
    if label == 0:
        return "ENTAILMENT"
    elif label == 1:
        return "NEUTRAL"
    elif label == 2:
        return "CONTRADICTION"


def str_to_label(pred: str) -> int:
    if pred == "ENTAILMENT":
        return 0
    elif pred == "NEUTRAL":
        return 2
    elif pred == "CONTRADICTION":
        return 1


from openai import OpenAI
import pandas as pd
import tqdm

client = OpenAI(api_key=OPENAI_API_KEY)
# evaluate accuracy using finetuned model "ft:gpt-3.5-turbo-1106:personal::Adiry86W"

splits = {
    "test": "plain_text/test-00000-of-00001.parquet",
    "validation": "plain_text/validation-00000-of-00001.parquet",
    "train": "plain_text/train-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["test"])


test_sample_df = df
test_sample_df = test_sample_df[test_sample_df["label"] != -1].reset_index(drop=True)


# Create test prompts
def create_test_prompt(row):
    return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"

system_prompt = "You are an AI trained to determine the relationship between two sentences. The possible relationships are: entailment (the first sentence implies the second), contradiction (the sentences contradict each other), or neutral (neither entailment nor contradiction)."


def model_predict(premise: str, hypothesis: str) -> str:

    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::Adiry86W",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Premise: {premise}\nHypothesis: {hypothesis}.",
            },
        ],
        temperature=0,
    )

    parts = response.choices[0].message.content.split("\n")
    return parts[0].strip().upper(), parts[1]


df = pd.read_csv("selected_results_2.csv")

progress = tqdm.tqdm(df.iterrows(), total=len(df))
n_correct = 0
n_total = 0

for i, row in progress:
    idx = row["snli_id"]
    premise = test_sample_df.at[int(idx), "premise"]
    hypothesis = test_sample_df.at[int(idx), "hypothesis"]
    label = row["true_label"]

    try:
        pred, reason = model_predict(premise, hypothesis)
        pred_label = str_to_label(pred)

        if pred_label == None:
            print("Invalid prediction: ", pred)
            pred_label = 1
    except Exception as e:
        print(f"Error: {e}")
        pred_label = 1
        reason = "N/A"

    df.at[i, "predicted_label"] = pred_label
    df.at[i, "reason"] = reason

    if pred_label == label:
        n_correct += 1

    n_total += 1

    progress.set_description(f"test_acc: {n_correct / n_total * 100:.2f}%")

df.to_csv("selected_results_pred.csv", index=False)