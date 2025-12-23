# import libraries
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio
import numpy as np
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import json
import re
import sys
from sklearn.model_selection import KFold

# set path explicitly to the project root
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# import custom functions
from utils.evaluation import extract_spans, evaluate_seqeval, cross_span_evaluation, mention_detection_evaluation, sentence_level_evaluation

# create helper functions
def create_llm_annotations(text, spans):

    # sort the spans according to their start and end index
    spans = sorted(spans, key=lambda x: x["start"])

    # store the text and set index variable
    llm_text = ""
    last_idx = 0

    # loop through spans and add the span with custom characters
    for span in spans:
        llm_text += text[last_idx:span["start"]]
        llm_text += f"@@{text[span["start"]:span["end"]]}##"
        last_idx = span["end"]

    # add the rest of the text
    llm_text += text[last_idx:]

    return llm_text

def llm_output_to_bio(annotated_text):

    # split words via a regex
    words = re.findall(r"@@.*?##|\w+|'\w+|[^\w\s]", annotated_text)

    # empty list to store the bio tags
    bio_tags = []

    # loop through all words
    for word in words:

        # if it is an annotated span, split words and assign bio labels
        if word.startswith("@@") and word.endswith("##"):
            entity_text = word[2:-2]
            entity_words = re.findall(r"\w+|'\w+|[^\w\s]", entity_text)
            for i, t in enumerate(entity_words):
                tag = "B-sg" if i == 0 else "I-sg"
                bio_tags.append((t, tag))
        else:
            # otherwise assign O tag
            bio_tags.append((word, "O"))

    return bio_tags

def add_special_characters(original_sentence, llm_list):

    matches = []

    # reorder the entities list according to length of the entity
    llm_list = sorted(llm_list, key=len, reverse=True)

    for entity in llm_list:
        # full-word match but allow possessive 's or ’s
        pattern = r"(?i)(?<![A-Za-z])(" + re.escape(entity) + r")(?:['’]s)?(?![A-Za-z])"

        for match in re.finditer(pattern, original_sentence):
            matches.append(match.span(1))

    non_overlapping = []
    occupied = [False] * len(original_sentence)

    for start, end in matches:
        if not any(occupied[start:end]):
            non_overlapping.append((start, end))
            for i in range(start, end):
                occupied[i] = True 

    non_overlapping.sort(key=lambda x: x[0])

    result = original_sentence
    for start, end in reversed(non_overlapping):
        result = result[:start] + "@@" + result[start:end] + "##" + result[end:]

    return result


# initialize empty dataset list
training_data = []

with open("01_data/classification/training_validation_sets/ner/training_set.json", "r") as f:
    raw_training_data = json.load(f)

# loop through all sentences in the data
for task in raw_training_data:
    # get the sentence and all annotations
    text = task["sentence"]
    spans = task["annotations"]
    labels = [annotation["text"] for annotation in spans]
   
    llm_text = create_llm_annotations(text, spans)
    bio_tags = llm_output_to_bio(llm_text)

    # add everything to the dataset list
    training_data.append({
        "text": text,
        "labels": labels,
        "llm_text": llm_text,
        "bio_tags": bio_tags
    })

# load the hyperparameter tuning results
with open("04_classification_evaluation/group_mention_detection/gen_llm/hyperparameter_tuning_results/ht_genllm_ner.json", "r") as f:
    ht_results = json.load(f)

# extract best temperature value (for 4o-mini) and best reasoning effort
best_temperature = float(max(ht_results[0], key=lambda t: ht_results[0][t]["f1"]))
best_reasoning = max(ht_results[1], key=lambda r: ht_results[1][r]["f1"])

# overwrite best reasoning to None
best_reasoning = None

# define the prompts
medium = """
## Task Objective
Extract all qualifying social groups from the sentence as a JSON object. Do not paraphrase the mentions at all! If no social groups are present, return an empty JSON array.

## Definition of a Social Group
A social group is a collective of people sharing **socio-demographic attributes** (age, ethnicity, income, occupation etc.).

**Do not** extract:
- institutional groups, political groups and state authorities ("small businesses", "armed forces", "Government", "Ministers")
- individual persons or highly specific collectives ("family of a colleague")

**Do** extract:
- subgroups of institutions with shared socio-demographic traits ("business people", "police officers")
- singular forms **only** if it is a generalization to a broader group ("every woman")
- social group component within a composite term ("victim support" -> "victim") or title ("Society for Disabled People" -> "Disabled People")
- additional sociodemographic description of the group ("women with mental health problems")
- extract as **one span** if several groups are mentioned and their meaning cannot be understood separately ("veterans, their wifes and children")
- general terms ("communities", "people") **only** if particular group is specified ("local communities", "people with special needs")

## Positive Examples:
- families, the taxpayer, victims, young people

## Negative Examples:
- EU, business, Tories, Labour, people, the public
"""

short = """
## Task Objective
Extract all qualifying social groups from the sentence as a JSON object. Do not paraphrase the mentions at all! If no social groups are present, return an empty JSON array.

## Definition of a Social Group
A social group is a collective of people sharing **socio-demographic attributes** (age, ethnicity, income, occupation etc.).

**Do not extract** institutional groups, firms, state authorities, groups based on political opinion or very specific small collectives such as one single family.
**Do extract** groupings of people within an institution and singular forms if it is generalizing to a broader collective.
Extract only the social group component unless there is an additional description to it.

## Positive Examples:
- families, the taxpayer, victims, criminals

## Negative Examples:
- EU, business, Tories, Labour, people, the public
"""

# compile manual few-shot examples in json format
positive_examples = [
    {"text": "We need to do more for young people, especially those with physical disabilities.",
     "llm_text": '{"social_groups": ["young people", "those with physical disabilities"]}'},
    {"text": "We want to support every child and ensure that the most vulnerable in this country can lead happy lives.",
     "llm_text": '{"social_groups": ["every child", "the most vulnerable in this country"]}'},
     {"text": "I want to express my support to all families in this country.",
     "llm_text": '{"social_groups": ["all families in this country"]}'}
]
negative_examples = [
    {"text": "The Government is working in close cooperaton with the Labour party on this matter.",
     "llm_text": '{"social_groups": []}'},
    {"text": "Businesses such as small- and medium-sized firms are vital for the economy in this country.",
     "llm_text": '{"social_groups": []}'},
     {"text": "Investment in the armed forces was certainly neglected under the previous Government.",
     "llm_text": '{"social_groups": []}'}
]

# create prompt template for the entity recognition task
def compile_prompt_ner(system_prompt, positive_examples, negative_examples, test_sentence, num_few_shot=4):

        chat = [
                {
                        "role": "system",
                        "content": system_prompt
                }
        ]
        

        num_few_shot_each = num_few_shot//2

        for i in range(num_few_shot_each):

                # add a positive example
                chat.append({"role": "user", "content": f"Sentence: {positive_examples[i]['text']}"})
                chat.append({"role": "assistant", "content": positive_examples[i]["llm_text"]})

                # add a negative example
                chat.append({"role": "user", "content": f"Sentence: {negative_examples[i]['text']}"})
                chat.append({"role": "assistant", "content": negative_examples[i]["llm_text"]})
        
        # add the test sentence
        chat.append({"role": "user", "content": f"Sentence: {test_sentence}"})

        return chat

# function to send out the request
async def send_request(client, model_name, prompt, output_class, semaphore, temp=None, reasoning_effort=None):
    async with semaphore:
        if model_name == "gpt-5-nano" :                                   
            response = await client.responses.parse(model=model_name,
                                                    input=prompt,
                                                    text_format=output_class,
                                                    reasoning = {"effort": reasoning_effort}
                                                    )
        elif model_name == "gpt-4o-mini":
            response = await client.responses.parse(model=model_name,
                                                    input=prompt,
                                                    text_format=output_class,
                                                    temperature=temp
                                                    )

        response_list = response.output_parsed.social_group
        try:
            if not isinstance(response_list, list):
                return []
            if response_list == ['']:
                return []
            return response_list
        except Exception:
            return []
                                            
                                        
async def dispatch_all(client, model_name, system_message, positive_examples, negative_examples, num_few_shot_examples, output_class, validation_data, temp, reasoning_effort, safe_interval, semaphore):
    tasks = []
    for row in validation_data:
        sentence = row["text"]
        prompt = compile_prompt_ner(
            system_message,
            positive_examples,
            negative_examples,
            sentence,
            num_few_shot=num_few_shot_examples
        )
        # create a task and fire it, do not wait
        task = asyncio.create_task(send_request(client, model_name, prompt, output_class, semaphore, temp, reasoning_effort))
        tasks.append(task)
        
        # wait before starting the next request
        await asyncio.sleep(safe_interval)

    # gather all results once everything is started
    return await asyncio.gather(*tasks)

async def run_dispatch(client, model_name, system_message, positive_examples, negative_examples, num_few_shot_examples, output_class, validation_data, temp, reasoning_effort, safe_interval):
    semaphore = asyncio.Semaphore(16)
    llm_output = await dispatch_all(client, model_name, system_message, positive_examples, negative_examples, num_few_shot_examples, output_class, validation_data, temp, reasoning_effort, safe_interval, semaphore)
    return llm_output

# set a general seed
seed = 3

# define class for the output
class SocialGroupJSON(BaseModel):
    social_group: list[str]

# create client for interacting with API
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# set the model names
model_names = ["gpt-5-nano", "gpt-4o-mini"]

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

# set the number of few-shot examples to try out
num_fs = [2, 4, 6]

# set number of splits and create empty dictionary to store overall results in
k = 5
average_metrics = {}

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
        "2_few_shot": {
            "seqeval": [],
            "cross_span": [],
            "mention_detection": [],
            "sentence_level": []
            },
        "4_few_shot": {
            "seqeval": [],
            "cross_span": [],
            "mention_detection": [],
            "sentence_level": []
            },
        "6_few_shot": {
            "seqeval": [],
            "cross_span": [],
            "mention_detection": [],
            "sentence_level": []
            }
        }
    
    # create cross-validation object for 5 folds

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(training_data)):
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
            llm_output = asyncio.run(run_dispatch(client, model_name, system_message, positive_examples, negative_examples,
                                                  fs, SocialGroupJSON, val_fold_data, temperature, reasoning_effort, safe_interval))
            output_texts_special_characters = [add_special_characters(original["text"], llm_list) for original, llm_list in zip(val_fold_data, llm_output)]
            output_bios = [llm_output_to_bio(text) for text in output_texts_special_characters]
            ground_truth_bio = [[tag for (_, tag) in sent["bio_tags"]] for sent in val_fold_data]
            pred_bio = [[tag for (_, tag) in sent] for sent in output_bios]
            filtered_gt_bio = [tag_list_gt for (tag_list_gt, tag_list_pred) in zip(ground_truth_bio, pred_bio) if len(tag_list_gt) == len(tag_list_pred)]
            filtered_pred_bio = [tag_list_pred for (tag_list_gt, tag_list_pred) in zip(ground_truth_bio, pred_bio) if len(tag_list_gt) == len(tag_list_pred)]

            # compute seqeval score
            seqeval_results = evaluate_seqeval(filtered_gt_bio, filtered_pred_bio)

            # extract spans and compute cross-span and sentence level evaluation
            all_true_spans = [extract_spans(filtered_gt_bio[idx]) for idx in range(len(filtered_gt_bio))]
            all_predicted_spans = [extract_spans(filtered_pred_bio[idx]) for idx in range(len(filtered_pred_bio))]
            cross_span_results = cross_span_evaluation(all_true_spans, all_predicted_spans)
            mention_detection_results = mention_detection_evaluation(all_true_spans, all_predicted_spans)
            sentence_level_results = sentence_level_evaluation(filtered_gt_bio, filtered_pred_bio)
            
            # append to fold metrics
            fs_key = f"{fs}_few_shot"
            fold_metrics[fs_key]["seqeval"].append(seqeval_results)
            fold_metrics[fs_key]["cross_span"].append(cross_span_results)
            fold_metrics[fs_key]["mention_detection"].append(mention_detection_results)
            fold_metrics[fs_key]["sentence_level"].append(sentence_level_results)


    for idx, fs in enumerate(num_fs):
        fs_key = f"{fs}_few_shot"

        seqeval_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["seqeval"]])
            for key in ["precision", "recall", "f1"]
        }
        cross_span_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["cross_span"]])
            for key in ["precision", "recall", "f1"]
        }
        mention_detection_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["mention_detection"]])
            for key in ["precision", "recall", "f1"]
        }
        sentence_level_metrics = {
            key: summarize([m[key] for m in fold_metrics[fs_key]["sentence_level"]])
            for key in ["precision", "recall", "f1"]
        }
        average_metrics[model_name][fs_key] = {
            "seqeval": seqeval_metrics,
            "cross_span": cross_span_metrics,
            "mention_detection": mention_detection_metrics,
            "sentence_level": sentence_level_metrics
        }

# export the cross-validation results
with open("04_classification_evaluation/group_mention_detection/cross_val_results/evaluation_metrics_gen_llm.json", "w") as f:
    json.dump(average_metrics, f)