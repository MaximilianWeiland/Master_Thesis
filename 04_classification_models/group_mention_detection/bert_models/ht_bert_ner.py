# libraries for data loading, manipulation and system/path settings
import json
import random
import gc
import warnings
import os
import sys
from pathlib import Path

# libraries for model building and training
from transformers import AutoTokenizer, logging
import torch
import optuna
from functools import partial

# set path to project root and import custom classes and functions
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.classification import (
    TokenDataset, EarlyStopping, collate_bert_ner, tune_bert_ner_optuna
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

with open(data_root / "data/training_validation_sets/ner/training_set.json", "r") as f:
    train_data = json.load(f)
with open(data_root / "data/training_validation_sets/ner/validation_set.json", "r") as f:
    val_data = json.load(f)


# initialize tag dictionary
tag_dict = {"O"}

# loop through all sentences
for task in train_data:
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

# hyperparameter tuning with TPE implemented via Optuna

# define the objective function whose output optuna tries to maximize
def objective(trial, model_name, train_dataset, val_dataset, collate_fn, tag2id, id2tag, device):

    # hyperparameter space in which optuna can search
    params = {
        "lr": trial.suggest_categorical("lr", [9e-6, 2e-5, 4e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.1, 0.3])
    }

    # create a fresh early stopper for each trial
    early_stopper = EarlyStopping(
        patience=3,
        save_model=False,
        path=None,
        printoption=False
    )

    # train the model with these hyperparameters
    best_f1, best_epoch, train_losses, val_losses, f1_scores_train, f1_scores_val = tune_bert_ner_optuna(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        model_name=model_name,
        tag2id=tag2id,
        id2tag=id2tag,
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
model_names = ["roberta-base", "bert-base-cased", "distilbert-base-cased", "microsoft/deberta-v3-base"]

# loop over the models
for model_name in model_names:

    print(f"\nStarting search for model: {model_name}")
    print("-"*50)

    # create a new optuna study, tokenizer fitting to the model as well as train and validation set
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create datasets with augmentations added
    augmented_train_data_generative = []
    num_augmentations_to_add = int(len(train_data) * 0.25)
    candidates_for_augmentation = random.sample(
        train_data, k=min(num_augmentations_to_add, len(train_data)))
    for original_item in candidates_for_augmentation:
        if "augmentations" in original_item and len(original_item["augmentations"]) > 0:
            aug_gen = original_item["augmentations"][-1]
            augmented_train_data_generative.append({
                "id": f"{original_item['id']}_aug_{aug_gen['method']}",
                "sentence": aug_gen["sentence"],
                "annotations": aug_gen["annotations"]
            })
    train_data_gen_augmentations = train_data + augmented_train_data_generative
    train_dataset = TokenDataset(data=train_data_gen_augmentations, tokenizer=tokenizer, tag2id=tag_to_id, max_len=128)
    val_dataset = TokenDataset(data=val_data, tokenizer=tokenizer, tag2id=tag_to_id, max_len=128)

    # run the optimization loop
    study.optimize(
        partial(objective,
                model_name=model_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_bert_ner,
                tag2id=tag_to_id,
                id2tag=id_to_tag,
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
    with open(data_root / f"ht_results/ht_bert_ner.json", "w") as f:
        json.dump(optimal_configs, f)

    print("-"*50)