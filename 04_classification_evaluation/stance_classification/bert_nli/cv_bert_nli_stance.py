# libraries for data loading and system/path settings
import json
import torch
import os
import sys
from pathlib import Path

# set path to project root and import custom classes and functions
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.classification import StanceNLIDataset
from utils.evaluation import cv_stance_nli

# choose to only use one GPU and set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load the training data
data_root = Path("/dataHDD1/max_weiland")
with open(data_root / "data/training_validation_sets/stance/training_set.json", "r") as f:
    data = json.load(f)


# load the results from hyperparameter tuning
with open(data_root / "ht_results/ht_bert_nli_stance.json", "r") as f:
    hyperparameter_tuning_results = json.load(f)

# set all model names
model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

# apply cross validation to all models with their best hyperparameters
average_metrics = cv_stance_nli(model_name=model_name, training_data=data, dataset_class=StanceNLIDataset,
                                num_folds=5, optimal_configurations=hyperparameter_tuning_results, device=device, seed=2)

# export the cross-validation metrics
with open(data_root / "cv_results/cv_metrics_bert_stance_nli.json", "w") as f:
    json.dump(average_metrics, f, indent=4)
