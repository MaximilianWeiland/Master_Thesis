# import libraries
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# set path explicitly to the project root
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# initialize empty dataset list
training_data = []

with open("01_data/classification/training_validation_sets/stance/training_set.json", "r") as f:
    raw_training_data = json.load(f)

# extract only the original sentence without any augmentations
tag_to_label = {
    "pos": "positive",
    "neutral": "neutral",
    "neg": "negative"
}

for task in raw_training_data:
    sentence = task[0]["sentence"]
    social_group = task[0]["group"]
    stance = tag_to_label[task[0]["stance"]]
    training_data.append({"sentence": sentence,
                            "social_group": social_group,
                            "stance": stance})
    
# load the hyperparameter tuning results
with open("04_classification_evaluation/stance_classification/gen_llm/hyperparameter_tuning_results/ht_genllm_stance.json", "r") as f:
    ht_results = json.load(f)

# extract best temperature value (for 4o-mini) and best reasoning effort
best_temperature = float(max(ht_results[0], key=lambda t: ht_results[0][t]))
best_reasoning = max(ht_results[1], key=lambda r: ht_results[1][r])
if best_reasoning == 'null':
    best_reasoning = None

# compile prompt templates
medium = """
## Task Objective
Label the following sentence according to the stance the speaker expresses towards the highlighted social group. Return this label as a JSON object.
The stance can be either positive, neutral or negative. Label the sentence according to the following definition and criteria.

## Positive Stance
The text is positive towards the group if it expresses some sort of support or positive affect.
Especially within questions, a positive stance can also be expressed indirectly by raising the interests of the respective group or criticizing their disadvantage.

## Neutral Stance
The text is neutral towards the group if it references the group, but neither a positive nor negative stance are taken.
This happens mostly when the speaker mentions the group by stating a fact or when the group is part of a title or organization name.

## Negative Stance
The text is negative towards the group if it expresses any form of critical or negative feeling towards the mentioned group. Raising awareness of the disadvantage faced by a group is not a negative but a positive stance.
"""

short = """
## Task Objective
Label the following sentence according to the stance the speaker expresses towards the highlighted social group. Return this label as a JSON object.
The stance can be either positive, neutral or negative.

## Positive Stance
The sentence expesses support or positive affect towards the group. Especially within questions, a positive stance can also be expressed indirectly by raising the interests of the group.

## Neutral Stance
The sentence references the group, but neither a positive nor negative stance are taken. This happens mostly within factual statements.

## Negative Stance
The sentence expresses a critical or negative feeling towards the mentioned group. Raising awareness of the disadvantage faced by a group is not a negative but a positive stance.
"""


# compile manual few-shot examples in json format
positive_example = {"text": "Stance towards young people in: Our party stands for improving the job opportunities of young people.",
        "llm_text": '{"stance": "positive"}'}
        
subtle_positive_example_1 = {"text": "Stance towards pupils: What is the Government's approach on creating a good working environment for pupils and teachers in our schools?",
     "llm_text": '{"stance": "positive"}'}

subtle_positive_example_2 = {"text": "Stance towards women: Women experience discrimination repeatedly throughout their lives",
     "llm_text": '{"stance": "positive"}'}

neutral_example = {"text": "Stance towards GP: Each person in this country should have the chance to get an appointment at his or her GP within a couple of days.",
     "llm_text": '{"stance": "neutral"}'}

negative_example = {"text": "Stance towards criminal offenders: We must do everything we can to tackle violence on our streets and get criminal offenders into jail.",
     "llm_text": '{"stance": "negative"}'}

# collect all in a list and mix up the order
few_shot_examples_all = [positive_example, subtle_positive_example_1, subtle_positive_example_2, neutral_example, neutral_example]

# create prompt template
def compile_prompt_stance(system_prompt, few_shot_examples, num_positive_examples, test_sentence, social_group):

        chat = [
                {
                        "role": "system",
                        "content": system_prompt
                }
        ]

        # add all positive few-shot examples
        for i in range(num_positive_examples):
                chat.append({"role": "user", "content": f"Sentence: {few_shot_examples[i]['text']}"})
                chat.append({"role": "assistant", "content": few_shot_examples[i]["llm_text"]})

        # add the remaining few-shot examples
        chat.append({"role": "user", "content": f"Sentence: {few_shot_examples[-2]['text']}"})
        chat.append({"role": "assistant", "content": few_shot_examples[-2]["llm_text"]})
        chat.append({"role": "user", "content": f"Sentence: {few_shot_examples[-1]['text']}"})
        chat.append({"role": "assistant", "content": few_shot_examples[-1]["llm_text"]})    
        
        # add the test sentence
        chat.append({"role": "user", "content": f"Stance towards {social_group} in: {test_sentence}"})

        return chat


# function to send out the request
async def send_request(client, model_name, prompt, output_class, semaphore, temp=None, reasoning_effort=None):
    async with semaphore:

        if model_name == "gpt-5-nano":
            response = await client.responses.parse(
                model=model_name,
                input=prompt,
                text_format=output_class,
                reasoning={"effort": reasoning_effort}
            )
        elif model_name == "gpt-4o-mini":
            response = await client.responses.parse(
                model=model_name,
                input=prompt,
                text_format=output_class,
                temperature=temp
            )

        stance = response.output_parsed.stance
        try:
            if not isinstance(stance, str):
                return None
            return stance
        except Exception:
            return None
                                         
                                        
async def dispatch_all(client, model_name, system_message, few_shot_examples, num_positive_examples,
                       output_class, validation_data, temp, reasoning_effort, safe_interval, semaphore):
    tasks = []
    for row in validation_data:
        sentence = row["sentence"]
        social_group = row["social_group"]
        prompt = compile_prompt_stance(
            system_message,
            few_shot_examples,
            num_positive_examples,
            sentence,
            social_group
        )
        # create a task and fire it, do not wait
        task = asyncio.create_task(send_request(client, model_name, prompt, output_class, semaphore, temp, reasoning_effort))
        tasks.append(task)
        
        # wait before starting the next request
        await asyncio.sleep(safe_interval)

    # gather all results once everything is started
    return await asyncio.gather(*tasks)

async def run_dispatch(client, model_name, system_message, few_shot_examples, num_positive_examples,
                       output_class, validation_data, temp, reasoning_effort, safe_interval):
    semaphore = asyncio.Semaphore(8)
    llm_output = await dispatch_all(
        client, model_name, system_message, few_shot_examples, num_positive_examples,
        output_class, validation_data, temp, reasoning_effort, safe_interval, semaphore
    )
    return llm_output

# set a general seed
seed = 3

# define class for the output
class StanceJSON(BaseModel):
    stance: str

# create client for interacting with API
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# set the model names
model_names = ["gpt-4o-mini", "gpt-5-nano"]

# set model configurations
model_configs = {
    "gpt-4o-mini": {
        "system_message": short,
        "temperature": best_temperature,
        "reasoning_effort": None
    },
    "gpt-5-nano": {
        "system_message": medium,
        "temperature": None,
        "reasoning_effort": best_reasoning
    }
}

# set the number of positive examples to try out
num_fs = [1, 2, 3]

# set number of splits and create empty dictionary to store overall results in
k = 5
average_metrics = {}

# helper function to append to fold metrics
def extract_fold_metrics(cr, eval_class):
    precision = cr[eval_class]["precision"]
    recall = cr[eval_class]["recall"]
    f1 = cr[eval_class]["f1-score"]

    return {"precision": precision,
            "recall": recall,
            "f1": f1}

# helper function to calculate summary statistics
def summarize(values):
    mean = np.mean(values)
    sd = np.std(values)
    ci = 1.96 * (sd/np.sqrt(5))
    return {
        "mean": mean,
        "sd": sd,
        "lower": mean - ci,
        "upper": mean + ci
        }

for model_name in model_names:
    print(f"Cross-validation for model {model_name} starts")
    print("-"*100)

    # calculate a safe interval in which requests are sent
    if model_name == "gpt-4o-mini":
        safe_rpm = 1000
        safe_interval = 60.0 / safe_rpm
    else:
        safe_rpm = 750
        safe_interval = 60.0 / safe_rpm

    # initialize a dictionary for this model
    average_metrics[model_name] = {}

    # dictionary to save the individual fold evaluation metrics
    fold_metrics = {
        "1_few_shot": {
            "negative": [],
            "neutral": [],
            "positive": [],
            "macro": []
            },
        "2_few_shot": {
            "negative": [],
            "neutral": [],
            "positive": [],
            "macro": []
            },
        "3_few_shot": {
            "negative": [],
            "neutral": [],
            "positive": [],
            "macro": []
            }
        }
    
    # create cross-validation object for 5 folds
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    all_labels = [item["stance"] for item in training_data]

    # split by preserving the class proportions
    for fold, (train_idx, val_idx) in enumerate(kf.split(training_data, all_labels)):
        print(f"Fold number: {fold + 1}")

        # create validation fold based on the provided indices
        val_fold_data = [training_data[i] for i in val_idx]

        # extract model configuration
        system_message = model_configs[model_name]["system_message"]
        temperature = model_configs[model_name]["temperature"]
        reasoning_effort = model_configs[model_name]["reasoning_effort"]

        # loop over number of few shot examples and calculate
        for idx, fs in enumerate(num_fs):

            # prompt the LLM and process result
            llm_output = asyncio.run(run_dispatch(client, model_name, system_message, few_shot_examples_all, fs, StanceJSON, val_fold_data, temperature, reasoning_effort, safe_interval))
            ground_truth = [item["stance"] for item in val_fold_data]
            prediction = [stance for stance in llm_output]
            metrics = classification_report(ground_truth, prediction, output_dict=True)

            # update the fold metrics
            fs_key = f"{fs}_few_shot"
            fold_metrics[fs_key]["negative"].append(extract_fold_metrics(metrics, "negative"))
            fold_metrics[fs_key]["neutral"].append(extract_fold_metrics(metrics, "neutral"))
            fold_metrics[fs_key]["positive"].append(extract_fold_metrics(metrics, "positive"))
            fold_metrics[fs_key]["macro"].append(extract_fold_metrics(metrics, "macro avg"))


    for idx, fs in enumerate(num_fs):
        fs_key = f"{fs}_few_shot"

        negative_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["negative"]])
            for key in ["precision", "recall", "f1"]
        }
        neutral_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["neutral"]])
            for key in ["precision", "recall", "f1"]
        }
        positive_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["positive"]])
            for key in ["precision", "recall", "f1"]
        }
        macro_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["macro"]])
            for key in ["precision", "recall", "f1"]
        }
        average_metrics[model_name][fs_key] = {
            "negative": negative_metrics,
            "neutral": neutral_metrics,
            "positive": positive_metrics,
            "macro": macro_metrics
        }

# export the cross-validation results
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_gen_llm.json", "w") as f:
    json.dump(average_metrics, f)