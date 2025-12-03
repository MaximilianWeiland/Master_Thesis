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
from utils.evaluation import cv_stance_nli

# create the dataset class
class StanceNLIDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len, label2id):
        self.dataset = []

        for item in raw_data:
            sentence = item["sentence"]
            target = item["group"]
            gold_stance = item["stance"]
            
            hypotheses = {
                "pos": f"The text is positive towards {target}.",
                "neg": f"The text is negative towards {target}.",
                "neutral": f"The text is neutral, or contains no stance, towards {target}."
            }

            for stance, hypothesis in hypotheses.items():
                label_text = "entailment" if stance == gold_stance else "not_entailment"

                encoding = tokenizer(
                    sentence,
                    hypothesis,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt"
                    )
                self.dataset.append({
                    "gold_stance": gold_stance,
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "label": label2id[label_text]
                    })
                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# load training and validation data
with open("01_data/training_validation_sets/stance/training_set.json", "r") as f:
    data = json.load(f)

# load the results from hyperparameter tuning
with open("04_classification_evaluation/stance_classification/bert_nli/hyperparameter_tuning_results/ht_bert_nli_stance.json", "r") as f:
    hyperparameter_tuning_results = json.load(f)

# set all model names
model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

# apply cross validation to all models with their best hyperparameters
average_metrics = cv_stance_nli(model_names=model_name, training_data=data, dataset_class=StanceNLIDataset,
                                num_folds=5, optimal_configurations=hyperparameter_tuning_results, seed=3)

# export the cross-validation metrics
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_bert_nli.json", "w") as f:
    json.dump(average_metrics, f, indent=4)