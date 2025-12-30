# load libraries
import json
import pandas as pd
import random
from pathlib import Path
import sys
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm 

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils.classification import TokenDataset, collate_bert_ner, train_bert
from utils.evaluation import labels_to_wordlevel_tags

# choose to only use one GPU and set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = Path("/dataHDD1/max_weiland")

# load the cross-validated evaluation results
with open(data_root / "../group_mention_detection/cross_val_results/evaluation_metrics_bert_ner.json", "r") as f:
    eval_bert = json.load(f)
with open(data_root / "../group_mention_detection/cross_val_results/evaluation_metrics_dictionary_ner.json", "r") as f:
    eval_dictionary = json.load(f)
with open(data_root / "../group_mention_detection/cross_val_results/evaluation_metrics_gen_llm_ner.json", "r") as f:
    eval_gen_llm = json.load(f)

# load training set
with open(data_root / "data/training_validation_sets/ner/training_set.json", "r") as f:
    training_set = json.load(f)

# load main dataset on the sentence level
sentence_level_df = pd.read_csv(data_root / "data/main_speech_datasets/researchperiod_sentencelevel.csv")

# find the best performing method based on seqeval
best_gen_llm = {"gpt-5-nano": {},
                "gpt-4o-mini": {}}

for method in eval_gen_llm:
    best_seqeval = 0
    best_num_few_shot = None
    for num_few_shot in eval_gen_llm[method]:
        seqeval_score = eval_gen_llm[method][num_few_shot]["seqeval"]["f1"]["mean"]
        if seqeval_score > best_seqeval:
            best_num_few_shot = num_few_shot
            best_seqeval = seqeval_score
    best_gen_llm[method] = eval_gen_llm[method][best_num_few_shot]

all_methods = [eval_bert, eval_dictionary, best_gen_llm]

best_performing_method_seqeval = None
best_performing_method_cross_span = None
best_seqeval = 0
best_cross_span = 0

for method in all_methods:
    for sub_method in method:
        seqeval_score = method[sub_method]["seqeval"]["f1"]["mean"]
        cross_span_score = method[sub_method]["cross_span"]["f1"]["mean"]
        if seqeval_score > best_seqeval:
            best_performing_method_seqeval = sub_method
            best_seqeval = seqeval_score
        if cross_span_score > best_cross_span:
            best_performing_method_cross_span = sub_method
            best_cross_span = cross_span_score

print(f"Best method according to seqeval: {best_performing_method_seqeval}")
print(f"Best method according to cross-span: {best_performing_method_cross_span}")

# initialize tag dictionary
tag_dict = {"O"}

# loop through all sentences
for task in training_set:
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

# load hyperparameter tuning results for the best model
with open(data_root / "../group_mention_detection/bert_models/hyperparameter_tuning_results/ht_bert_ner.json", "r") as f:
    ht_results = json.load(f)

# extract the optimal hyperparameter combination
epochs = ht_results[best_performing_method_seqeval]["best_epoch"]
lr = ht_results[best_performing_method_seqeval]["best_params"]["lr"]
batch_size = ht_results[best_performing_method_seqeval]["best_params"]["batch_size"]
weight_decay = ht_results[best_performing_method_seqeval]["best_params"]["weight_decay"]

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(best_performing_method_seqeval)

# create tensor dataset and respective data loaders
augmented_train_data_generative = []
num_augmentations_to_add = int(len(training_set) * 0.25)
candidates_for_augmentation = random.sample(
     training_set, k=min(num_augmentations_to_add, len(training_set)))
for original_item in candidates_for_augmentation:
    if "augmentations" in original_item and len(original_item["augmentations"]) > 0:
        aug_gen = original_item["augmentations"][-1]
        augmented_train_data_generative.append({
            "id": f"{original_item['id']}_aug_{aug_gen['method']}",
            "sentence": aug_gen["sentence"],
            "annotations": aug_gen["annotations"]})
train_data_gen_augmentations = training_set + augmented_train_data_generative

train_dataset = TokenDataset(train_data_gen_augmentations, tokenizer, tag_to_id, max_len=128)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_bert_ner)

# create the model and optimizer
model = AutoModelForTokenClassification.from_pretrained(
     best_performing_method_seqeval,
     num_labels=len(tag_to_id),
     id2label=id_to_tag,
     label2id=tag_to_id
     ).to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# train the model with the optimal configuration
train_bert(train_dataloader, model, optimizer, epochs, device, which_task="ner")

# save the model
torch.save(model.state_dict(), data_root / "final_models/social_group_detection_model.pt")

class InferenceTokenDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["sentence"]

        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "offset_mapping": encoding["offset_mapping"].squeeze(0),
            "word_ids": encoding.word_ids()
        }
    
def collate_bert_ner(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "offset_mapping": torch.stack([item["offset_mapping"] for item in batch]),
        "word_ids": [item["word_ids"] for item in batch]
    }

# turn the entire dataset to class object
sentence_level_dataset = InferenceTokenDataset(sentence_level_df, tokenizer, 128)

# turn it to a dataloader
sentence_level_dataloader = DataLoader(sentence_level_dataset, batch_size=128, shuffle=False, collate_fn=collate_bert_ner)

def bio_tags_to_spans(word_tags, word_offsets):
    spans = []
    start = None

    for i, tag in enumerate(word_tags):
        if tag == "B":
            if start is not None:
                spans.append((start, word_offsets[i - 1][1]))
            start = word_offsets[i][0]

        elif tag == "I":
            continue

        else:
            if start is not None:
                spans.append((start, word_offsets[i - 1][1]))
                start = None

    if start is not None:
        spans.append((start, word_offsets[len(word_tags) - 1][1]))

    return spans

model.eval()

all_results = []
   
with torch.no_grad():
      
      progress_bar = tqdm(sentence_level_dataloader, desc="Inference")
      for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_mask"].to(device)
        offset_mappings = batch["offset_mapping"]
        word_ids = batch["word_ids"]

        outputs = model(input_ids=input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

        for preds, sent_word_ids, offsets in zip(predictions, word_ids, offset_mappings):
            pred_np_array = preds.cpu().numpy()
            word_tags, _ = labels_to_wordlevel_tags(pred_np_array, id_to_tag, sent_word_ids)
            word_offsets = []
            prev_word_id = None
            for off, wid in zip(offsets, sent_word_ids):
                if wid is None or wid == prev_word_id:
                    continue
                word_offsets.append(off)
                prev_word_id = wid
            spans = bio_tags_to_spans(word_tags, word_offsets)
            all_results.append(spans)

all_text_spans = []

for spans, text in zip(all_results, sentence_level_df["sentence"]):
    sent_spans = [text[start:end] for start, end in spans]
    all_text_spans.append(sent_spans)

sentence_level_df["sg_spans"] = all_text_spans

# reorder the columns
cols_order = ["date", "agenda", "text", "sentence", "sg_spans", "speech_id", "speechnumber", "speaker", "party", "chair", "age", "gender", "birth_date", "vulnerability", "backbencher", "constituency_name", "ons_id", "under_30", "over_65"]
sentence_level_df = sentence_level_df.loc[:, cols_order]

# export the df
sentence_level_df.to_csv(data_root / "data//main_speech_datasets/researchperiod_sentencelevel_socialgroups.csv", index=False)