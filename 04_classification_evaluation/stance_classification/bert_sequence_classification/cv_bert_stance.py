# libraries for data loading and system/path settings
import json
import sys
from pathlib import Path

# set path to project root
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# import custom classes and functions
from utils.classification import StanceDataset
from utils.evaluation import cv_stance

# load training and validation data
with open("01_data/training_validation_sets/stance/training_set.json", "r") as f:
    data = json.load(f)

# initialize dictionary for the sentiment classes
sent_dict = set()

# loop through all sentences
for task in data:
    sent_dict.add(task[0]["stance"])

# sort the tag dictionary
label_list = sorted(sent_dict)

# dictionaries that convert from id to tag and vice versa
label_to_id = {tag: i for i, tag in enumerate(label_list)}
id_to_label = {id: label for label, id in label_to_id.items()}

# load the results from hyperparameter tuning
with open("04_classification_evaluation/stance_classification/bert_sequence_classification/hyperparameter_tuning_results/ht_bert_sequence.json", "r") as f:
    hyperparameter_tuning_results = json.load(f)

# set all model names
model_names = ["roberta-base", "bert-base-cased", "distilbert-base-cased", "microsoft/deberta-v3-base"]

# apply cross validation to all models with their best hyperparameters
average_metrics = cv_stance(model_names=model_names, training_data=data, dataset_class=StanceDataset, label2id=label_to_id,
                            id2label=id_to_label, num_folds=5, optimal_configurations=hyperparameter_tuning_results, seed=3)

# export the cross-validation metrics
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_bert_sequence.json", "w") as f:
    json.dump(average_metrics, f, indent=4)