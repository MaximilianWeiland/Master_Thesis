# import libraries
import pandas as pd
import json
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# import custom helper functions
from utils.preprocessing import create_regex_pattern
from utils.classification import text_to_bio, find_dictionary_matches
from utils.evaluation import extract_spans, evaluate_seqeval, cross_span_evaluation, mention_detection_evaluation, sentence_level_evaluation
from sklearn.model_selection import KFold

# import the annotations
with open("01_data/classification/training_validation_sets/ner/training_set.json", "r") as f:
    data = json.load(f)

# import the dictionary
group_dictionary_df = pd.read_csv("01_data/classification/dictionary/groups_dictionary.csv")

# add bio tags to the dataset
for task in data:
    bio_tags = text_to_bio(task)
    task["bio_tags"] = bio_tags

# create the regex pattern
combined_regex = create_regex_pattern(group_dictionary_df)

# empty dictionary to store the final average metrics
average_metrics = {}

 # dictionary to save the individual fold evaluation metrics
fold_metrics = {
    "seqeval": [],
    "cross_span": [],
    "mention_detection":[],
    "sentence_level": []
    }

k = 5
seed = 3
kf = KFold(n_splits=k, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):

    # create the training and validation fold based on the provided indices
    train_fold_data = [data[i] for i in train_idx]
    val_fold_data = [data[i] for i in val_idx]

    # store all bio tags in a list
    gt_bio = [sent["bio_tags"] for sent in val_fold_data]

    # empty list to store predicted tags
    pred_bio = []

    # loop through tasks, find dictionary matches and append bio tags to list
    for task in val_fold_data:
        sentence = task["sentence"]
        pred_tags = find_dictionary_matches(sentence, combined_regex)
        pred_bio.append(pred_tags)

    # evaluate at the entity level with seqeval
    metrics_seqeval = evaluate_seqeval(gt_bio, pred_bio)

    # evaluate on the entity level with custom cross-span metric
    all_true_spans = []
    all_predicted_spans = []

    # get spans of all true positives and predictions
    for idx in range(len(gt_bio)):
        all_true_spans.append(extract_spans(gt_bio[idx]))
        all_predicted_spans.append(extract_spans(pred_bio[idx]))
 
    # apply cross-span evaluation
    metrics_cross_span = cross_span_evaluation(all_true_spans, all_predicted_spans)

    # apply mention detection evaluation
    metrics_mention_detection = mention_detection_evaluation(all_true_spans, all_predicted_spans)

    # evaluate at the sentence level
    metrics_sentence_level = sentence_level_evaluation(gt_bio, pred_bio)

    # append all metrics to the dictionary
    fold_metrics["seqeval"].append(metrics_seqeval)
    fold_metrics["cross_span"].append(metrics_cross_span)
    fold_metrics["mention_detection"].append(metrics_mention_detection)
    fold_metrics["sentence_level"].append(metrics_sentence_level)

# helper function to calculate summary statistics
def summarize(values):
    mean = np.mean(values)
    sd = np.std(values)
    return {
        "mean": mean,
        "sd": sd,
        "lower": mean - 1.96 * sd,
        "upper": mean + 1.96 * sd
        }
            
seqeval_metrics = {
     key: summarize([m[key] for m in fold_metrics["seqeval"]])
     for key in ["precision", "recall", "f1"]
     }
cross_span_metrics = {
     key: summarize([m[key] for m in fold_metrics["cross_span"]])
     for key in ["precision", "recall", "f1"]
    }
mention_detection_metrics = {
     key: summarize([m[key] for m in fold_metrics["mention_detection"]])
     for key in ["precision", "recall", "f1"]
    }
sentence_level_metrics = {
     key: summarize([m[key] for m in fold_metrics["sentence_level"]])
     for key in ["precision", "recall", "f1"]
     }

average_metrics["dictionary"] = {
    "seqeval": seqeval_metrics,
    "cross_span": cross_span_metrics,
    "mention_detection": mention_detection_metrics,
    "sentence_level": sentence_level_metrics
    }

with open("04_classification_evaluation/group_mention_detection/cross_val_results/evaluation_metrics_dictionary.json", "w") as f:
    json.dump(average_metrics, f, indent=4)