# libraries for data loading, manipulation and system/path settings
import json
import gc
import warnings
import os
import sys
from pathlib import Path

# libraries for model building and training
from transformers import AutoTokenizer, logging
import torch
from torch.utils.data import ConcatDataset
import optuna
from functools import partial
from sklearn.utils import resample

# set path to project root and import custom classes and functions
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.classification import (
    StanceDataset, EarlyStopping, tune_bert_stance_optuna
)

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

# initialize dictionary for the sentiment classes
sent_dict = set()

# loop through all sentences
for task in train_data:
    sent_dict.add(task[0]["stance"])

# sort the tag dictionary
label_list = sorted(sent_dict)

# dictionaries that convert from id to tag and vice versa
label_to_id = {tag: i for i, tag in enumerate(label_list)}
id_to_label = {id: label for label, id in label_to_id.items()}


# oversample the minority class

# unpack original training and validation data
train_data_original = [task[0] for task in train_data]
val_data = [task[0] for task in val_data]

# separate according to stance
pos_ann = [r for r in train_data if r[0]["stance"] == "pos"]
neu_ann = [r for r in train_data if r[0]["stance"] == "neutral"]
neg_ann = [r for r in train_data if r[0]["stance"] == "neg"]

# get the number of positive stances (which is the maximum size)
max_size = len(pos_ann)

# resample indices for the negative class
neg_extra = resample(neg_ann, replace=False, n_samples=int(len(neg_ann)*0.5), random_state=0)

# compile augmentation datasets
train_data_neg_aug = [task[2] for task in neg_extra]

# hyperparameter tuning with TPE implemented via Optuna

# define the objective function whose output optuna tries to maximize
def objective(trial, model_name, train_dataset, val_dataset, label2id, id2label, device):

    # hyperparameter space in which optuna can search
    params = {
        "lr": trial.suggest_categorical("lr", [9e-6, 2e-5, 4e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.1, 0.3]),
    }

    # define a fresh early stopper
    early_stopper=EarlyStopping(patience=3, save_model=False, path=None, printoption=False)

    # train the model with these hyperparameters
    best_f1, best_epoch, train_losses, val_losses, f1_scores_train, f1_scores_val = tune_bert_stance_optuna(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        label2id=label2id,
        id2label=id2label,
        device=device,
        early_stopper=early_stopper,
        params=params
    )

    # besides parameters, save losses and f1-scores in the trial object
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_losses", val_losses)
    trial.set_user_attr("f1_scores_train", f1_scores_train)
    trial.set_user_attr("f1_scores_val", f1_scores_val)

    return best_f1


# dictionary to save optimal paramaters
optimal_configs = {}

# number of trials, the device on which models should operate and the specific models to test
num_trials = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ["roberta-base", "bert-base-cased", "distilbert-base-cased", "microsoft/deberta-v3-base"]

# loop over the models
for model_name in model_names:

    print(f"\nStarting search for model: {model_name}")
    print("-"*50)

    # create a new optuna study, tokenizer fitting to the model as well as train and validation set
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create datasets with oversampled data
    train_dataset = StanceDataset(train_data_original, tokenizer, label_to_id, max_len=128)
    train_dataset_neg_aug = StanceDataset(train_data_neg_aug, tokenizer, label2id=label_to_id, max_len=128)
    train_dataset = ConcatDataset([train_dataset, train_dataset_neg_aug])
    val_dataset = StanceDataset(data=val_data, tokenizer=tokenizer, label2id=label_to_id, max_len=128)

    # run the optimization loop
    study.optimize(
        partial(objective,
                model_name=model_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                label2id=label_to_id,
                id2label=id_to_label,
                device=device),
        n_trials=num_trials
    )

    # get the best trial and save optimal parameters
    best_trial = study.best_trial
    optimal_configs[model_name] = {
        "best_f1": best_trial.value,
        "best_params": best_trial.params,
        "best_epoch": best_trial.user_attrs["best_epoch"],
        "train_losses": best_trial.user_attrs["train_losses"],
        "val_losses": best_trial.user_attrs["val_losses"],
        "f1_scores_train": best_trial.user_attrs["f1_scores_train"],
        "f1_scores_val": best_trial.user_attrs["f1_scores_val"]
    }

    # clean all objects and empty cache for the next model
    del study, tokenizer, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # write results to output directory
    with open(data_root / f"ht_results/ht_bert_sequence_stance.json", "w") as f:
        json.dump(optimal_configs, f)

    print("-"*50)