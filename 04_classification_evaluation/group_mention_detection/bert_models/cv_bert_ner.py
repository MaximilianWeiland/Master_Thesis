# libraries for data loading and system/path settings
import json
import sys
from pathlib import Path

# libraries for model building and training
import torch
from torch.utils.data import Dataset

# set path to project root and import custom classes and functions
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.evaluation import cv_ner

def tokenization_labelling(text, entities, tokenizer, tag2id, max_len):

    # get the encoding of the sentence
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True,
                         max_length=max_len, padding="max_length")
    
    # create preliminary list with O tags for all tokens
    tags = ["O"] * len(encoding.offset_mapping)

    # loop over annotations and extract start and end index as well as the given tag
    for ent in entities:
        start, end = ent["start"], ent["end"]
        ent_tag = ent["tag"][0:2]

        # loop over all tokens in the sentence and check for overlap
        for idx, (token_start, token_end) in enumerate(encoding.offset_mapping):
            # continue if it is a special token
            if token_start == token_end == 0:
                continue
            # check for overlap (overlap checking strictly necessary for deberta)
            if token_end > start and token_start < end:
                # assign B-tag if start is equal or smaller (smaller if deberta)
                if token_start <= start:
                    tags[idx] = f"B-{ent_tag}"
                # otherwise it is an inside token
                else:
                    tags[idx] = f"I-{ent_tag}"

    # extract the word ids
    word_ids = encoding.word_ids()

    # ensure propagate B-tags are propagated to all subwords of the same word (only actually relevant for deberta)
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        # if token has B-tag ensure that all other tokens of the same word get I-tag
        if tags[idx].startswith("B-"):
            for j, wid2 in enumerate(word_ids):
                if wid2 == wid and j != idx:
                    tags[j] = f"I-{ent_tag}"

    # convert tags to IDs, masking special tokens
    tag_ids = [-100 if wid is None else tag2id.get(tag, tag2id["O"])
               for tag, wid in zip(tags, encoding.word_ids())]

    return encoding["input_ids"], encoding["attention_mask"], tag_ids, word_ids


class TokenDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, max_len):
        self.dataset = []
        self.max_len = max_len

        for task in data:
            # get the sentence and all annotations
            text = task["sentence"]
            spans = task["annotations"]

            # tokenize and get all ids
            input_ids, attention_mask, tag_ids, word_ids = tokenization_labelling(text, spans, tokenizer, tag2id, self.max_len)

            # add everything to the dataset list
            self.dataset.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
                "word_ids": word_ids})
  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    tag_ids = torch.stack([item["tag_ids"] for item in batch])
    word_ids = [item["word_ids"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "tag_ids": tag_ids,
        "word_ids": word_ids
    }


# load the training data
with open("01_data/training_validation_sets/ner/training_set.json", "r") as f:
    data = json.load(f)

# initialize tag dictionary
tag_dict = {"O"}

# loop through all sentences
for task in data:
    if task["annotations"]:
        for annotation in task["annotations"]:
            label = annotation["tag"][0:2]
            tag_dict.add(f"B-{label}")
            tag_dict.add(f"I-{label}")

# sort the tag dictionary
tag_list = sorted(tag_dict)

# dictionaries that convert from id to tag and vice versa
tag_to_id = {tag: i for i, tag in enumerate(tag_list)}
id_to_tag = {id: label for label, id in tag_to_id.items()}

# load the results from hyperparameter tuning
with open("04_classification_evaluation/group_mention_detection/bert_models/hyperparameter_tuning_results/ht_bert_ner.json", "r") as f:
    hyperparameter_tuning_results = json.load(f)

# set all model names
model_names = ["roberta-base", "bert-base-cased", "distilbert-base-cased", "microsoft/deberta-v3-base"]

# apply cross validation to all models with their best hyperparameters
average_metrics = cv_ner(model_names=model_names, training_data=data, dataset_class=TokenDataset,
                         label2id=tag_to_id, id2label=id_to_tag, num_folds=5, optimal_configurations=hyperparameter_tuning_results,
                         classification_task="ner", custom_collate_fn=custom_collate_fn, seed=3)

# export the cross-validation metrics
with open("04_classification_evaluation/group_mention_detection/cross_val_results/evaluation_metrics_bert.json", "w") as f:
    json.dump(average_metrics, f, indent=4)