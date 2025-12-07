# libraries for data loading and system/path settings
import json
import sys
from pathlib import Path

# set path to project root and import custom classes and functions
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.classification import TokenDataset, collate_bert_ner
from utils.evaluation import cv_ner

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
                         classification_task="ner", custom_collate_fn=collate_bert_ner, seed=3)

# export the cross-validation metrics
with open("04_classification_evaluation/group_mention_detection/cross_val_results/evaluation_metrics_bert.json", "w") as f:
    json.dump(average_metrics, f, indent=4)