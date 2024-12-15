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
        return 1
    elif pred == "CONTRADICTION":
        return 2


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


def model_predict(premise: str, hypothesis: str, *, few_shot: bool = False) -> str:
    prompt = ""

    if few_shot:
        prompt += """
Here are some examples:

Premise: A soccer game with multiple males playing.
Hypothesis: Some men are playing a sport.
Answer: ENTAILMENT.

Premise: A man inspects the uniform of a figure in some East Asian country.
Hypothesis: The man is sleeping.
Answer: CONTRADICTION

Premise: An older and younger man smiling.
Hypothesis: Two people are brothers.
Answer: NEUTRAL

Premise: A black race car starts up in front of a crowd of people.
Hypothesis: A car is about to begin moving.
Answer: ENTAILMENT

Premise: A man in a blue shirt is standing in a room with a television.
Hypothesis: The television is turned on.
Answer: NEUTRAL

===

"""

    prompt += f"""
    Premise: {premise}
    Hypothesis: {hypothesis}

    Respond with exactly one of:
    ENTAILMENT - if the hypothesis must be true given the premise
    NEUTRAL - if the hypothesis might be true given the premise
    CONTRADICTION - if the hypothesis cannot be true given the premise

    Answer:"""
    
    max_retries = 8
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip().upper()
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            # Calculate exponential backoff delay
            delay = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
            time.sleep(delay)
            continue


progress = tqdm.tqdm(df.iterrows(), total=len(df))
n_correct = 0
n_total = 0

for i, row in progress:
    premise = row["premise"]
    hypothesis = row["hypothesis"]
    true_label = row["label"]

    try:
        pred = model_predict(premise, hypothesis)
        pred_label = str_to_label(pred)

        if pred_label == None:
            print("Invalid prediction: ", pred)
            pred_label = 1
    except Exception as e:
        print(f"Error: {e}")
        pred_label = 1

    df.at[i, "predicted_label"] = pred_label

    if pred_label == true_label:
        n_correct += 1

    n_total += 1

    progress.set_description(f"test_acc: {n_correct / n_total * 100:.2f}%")

df.to_csv("selected_results_pred.csv", index=False)