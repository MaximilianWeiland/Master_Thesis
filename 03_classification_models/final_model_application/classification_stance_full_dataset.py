# load libraries
import json
import ast
import pandas as pd
from pathlib import Path
import sys
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import AdamW
from sklearn.utils import resample

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils.classification import StanceNLIDataset, train_bert

# choose to only use one GPU and set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = Path("/dataHDD1/max_weiland")

# load the cross-validated evaluation results
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_bert_stance_nli.json", "r") as f:
    eval_bert_nli = json.load(f)
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_bert_stance_sequence.json", "r") as f:
    eval_bert_sequence = json.load(f)
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_dictionary_stance.json", "r") as f:
    eval_dictionary = json.load(f)
with open("04_classification_evaluation/stance_classification/cross_val_results/evaluation_metrics_gen_llm_stance.json", "r") as f:
    eval_gen_llm = json.load(f)

# load training set
with open(data_root / "data/training_validation_sets/stance/training_set.json", "r") as f:
    training_set = json.load(f)

# load sentence-level dataset with social groups already extracted
sentence_level_df = pd.read_csv(data_root / "data/speech_datasets/researchperiod_sentencelevel_socialgroups.csv")

# convert string lists to actual lists
sentence_level_df["sg_spans"] = sentence_level_df["sg_spans"].apply(ast.literal_eval)

# find the best performing method based on macro f1 score
best_gen_llm = {"gpt-5-nano": {},
                "gpt-4o-mini": {}}

for method in eval_gen_llm:
    best_macro_f1 = 0
    best_num_few_shot = None
    for num_few_shot in eval_gen_llm[method]:
        macro_f1_score = eval_gen_llm[method][num_few_shot]["macro"]["f1"]["mean"]
        if macro_f1_score > best_macro_f1:
            best_num_few_shot = num_few_shot
            best_macro_f1 = macro_f1_score
    best_gen_llm[method] = eval_gen_llm[method][best_num_few_shot]

all_methods = [eval_bert_nli, eval_bert_sequence, eval_dictionary, best_gen_llm]

best_performing_method = None
best_macro_f1 = 0

for method in all_methods:
    for sub_method in method:
        macro_f1_score = method[sub_method]["macro"]["f1"]["mean"]
        if macro_f1_score > best_macro_f1:
            best_performing_method = sub_method
            best_macro_f1 = macro_f1_score
        
print(f"Best method according to macro f1 score: {best_performing_method}")

# load hyperparameter tuning results
with open(project_root / "04_classification_evaluation/stance_classification/bert_nli/hyperparameter_tuning_results/ht_bert_nli_stance.json", "r") as f:
    ht_results = json.load(f)

# get the optimal hyperparameters for this model
epochs = ht_results[best_performing_method]["best_epoch"]
lr = ht_results[best_performing_method]["best_params"]["lr"]
batch_size = ht_results[best_performing_method]["best_params"]["batch_size"]
weight_decay = ht_results[best_performing_method]["best_params"]["weight_decay"]

# set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# define the tokenizer
tokenizer = AutoTokenizer.from_pretrained(best_performing_method)

model = AutoModelForSequenceClassification.from_pretrained(best_performing_method).to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
label_to_id = model.config.label2id

train_data_original = [task[0] for task in training_set]
train_dataset = StanceNLIDataset(train_data_original, tokenizer, max_len=128, label2id=label_to_id)

# oversample minority classes to a certain proportion and create datasets
neg_ann = [r for r in training_set if r[0]["stance"] == "neg"]
neg_extra = resample(neg_ann, replace=False, n_samples=int(len(neg_ann)*0.5), random_state=0)
train_data_neg_aug = [task[2] for task in neg_extra]
train_dataset_neg_aug = StanceNLIDataset(train_data_neg_aug, tokenizer, max_len=128, label2id=label_to_id)
train_dataset_balanced = ConcatDataset([train_dataset, train_dataset_neg_aug])

# create the data loader
train_dataloader = DataLoader(train_dataset_balanced, batch_size=batch_size, shuffle=True)

# train the model with the optimal configuration
train_bert(train_dataloader, model, optimizer, epochs, device, which_task="stance")

# save the model
torch.save(model.state_dict(), data_root / "final_models/stance_classification_model.pt")

model.eval()
all_preds = []
total_rows = len(sentence_level_df)
num_cuts = total_rows // 100
cut_reached = 0
start_time = time.time()

for idx, row in sentence_level_df.iterrows():

    if idx > 0 and idx % 100 == 0:
        cut_reached += 1
        cut_time = time.time() - start_time
        print(f"{cut_reached}/{num_cuts}, Time elapsed: {cut_time:.2f} seconds")

    sentence = row["sentence"]
    targets = row["sg_spans"]

    row_preds = []

    for target in targets:
        hypotheses = {
            "positive": f"The text is positive towards {target}.",
            "negative": f"The text is negative towards {target}.",
            "neutral": f"The text is neutral, or contains no stance, towards {target}."
        }

        inputs = tokenizer(
            [sentence] * 3,
            list(hypotheses.values()),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entail_probs = probs[:, 0].tolist()

        predicted_stance = list(hypotheses.keys())[entail_probs.index(max(entail_probs))]
        row_preds.append(predicted_stance)

    all_preds.append(row_preds)

sentence_level_df["stances"] = all_preds

# reorder the columns
cols_reorder = ["date", "days_until_election", "agenda", "text", "sentence", "sg_spans", "stances", "speech_id", "speechnumber", "speaker", "party", "chair", "age", "gender", "birth_date", "vulnerability", "backbencher", "constituency_name", "ons_id", "under_30", "over_65"]
sentence_level_df = sentence_level_df.loc[:, cols_reorder]

# export the df
sentence_level_df.to_csv(data_root / "data/speech_datasets/researchperiod_sentencelevel_socialgroups_stances.csv", index=False)