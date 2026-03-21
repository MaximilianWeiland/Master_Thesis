# libraries for data loading, manipulation and system/path settings
import json
import warnings
import os
import sys
from pathlib import Path

# libraries for model building and training
from transformers import AutoTokenizer, logging
import torch
import optuna
from functools import partial
from sklearn.utils import resample

# set path to project root and import custom classes and functions
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from utils.classification import EarlyStopping, tune_bert_nli_stance_optuna

# global settings to suppress unproblematic warning messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer")

# choose to only use one GPU and set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load training and validation data
data_root = Path("/dataHDD1/max_weiland")
with open(data_root / "data/training_validation_sets/stance/training_set.json", "r") as f:
    train_data = json.load(f)
with open(data_root / "data/training_validation_sets/stance/validation_set.json", "r") as f:
    val_data = json.load(f)

# oversample the minority classes

# unpack original training and validation data
train_data_original = [task[0] for task in train_data]
val_data = [task[0] for task in val_data]

# separate according to stance
pos_ann = [r for r in train_data if r[0]["stance"] == "pos"]
neu_ann = [r for r in train_data if r[0]["stance"] == "neutral"]
neg_ann = [r for r in train_data if r[0]["stance"] == "neg"]

# get the number of positive stances (which is the maximum size)
max_size = len(pos_ann)

# resample indices for the minority classes
neg_extra = resample(neg_ann, replace=False, n_samples=int(len(neg_ann)*0.5), random_state=0)

# compile augmentation datasets
train_data_neg_aug = [task[2] for task in neg_extra]

# hyperparameter tuning with TPE implemented via Optuna

# define the objective function whose output optuna tries to maximize
def objective(trial, model_name, train_data_original, train_data_neg_aug, val_dataset, device):

    # hyperparameter space in which optuna can search
    params = {
        "lr": trial.suggest_categorical("lr", [9e-6, 2e-5, 4e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.1, 0.3]),
    }

    # define a fresh early stopper
    early_stopper=EarlyStopping(patience=3, save_model=False, path=None, printoption=False)

    # train the model with these hyperparameters
    best_f1, best_epoch, f1_scores_train, f1_scores_val = tune_bert_nli_stance_optuna(
        train_data_original=train_data_original,
        train_data_neg_aug=train_data_neg_aug,
        val_dataset=val_dataset,
        model_name=model_name,
        device=device,
        early_stopper=early_stopper,
        params=params
    )

    # besides parameters, save losses and f1-scores in the trial object
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("f1_scores_train", f1_scores_train)
    trial.set_user_attr("f1_scores_val", f1_scores_val)

    return best_f1


# dictionary to save optimal paramaters
optimal_configs = {}

# number of trials, the device on which models should operate and the specific models to test
num_trials = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

print(f"\nStarting search for model: {model_name}")
print("-"*70)

# create a new optuna study, tokenizer fitting to the model as well as train and validation set
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
tokenizer = AutoTokenizer.from_pretrained(model_name)

# run the optimization loop
study.optimize(
    partial(objective,
            model_name=model_name,
            train_data_original=train_data_original,
            train_data_neg_aug=train_data_neg_aug,
            val_dataset=val_data,
            device=device),
    n_trials=num_trials
)

# get the best trial and save optimal parameters
best_trial = study.best_trial
optimal_configs[model_name] = {
    "best_f1": best_trial.value,
    "best_params": best_trial.params,
    "best_epoch": best_trial.user_attrs["best_epoch"],
    "f1_scores_train": best_trial.user_attrs["f1_scores_train"],
    "f1_scores_val": best_trial.user_attrs["f1_scores_val"]
}

# write results to output directory
with open(data_root / f"ht_results/ht_bert_nli_stance.json", "w") as f:
    json.dump(optimal_configs, f)