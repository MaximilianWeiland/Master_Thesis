import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from sklearn.metrics import classification_report as sklearn_classification_report
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import gc
from utils.evaluation import evaluate_seqeval, run_testset_ner, run_testset_stance, evaluate_nli_stance

# ----------------------------------------------------------------------
# Dictionary Baseline
# ----------------------------------------------------------------------

# function to convert the annotations to bio tags on the word level
def __tokenize_word_level(sentence):
    
    # get all words' start and end index
    word_spans = []
    char_idx = 0

    # split sentence via regex which ensures to also split at punctuation
    words = re.findall(r"\w+|'\w+|[^\w\s]", sentence)

    for word in words:
        start_idx = sentence.find(word, char_idx)
        end_idx = start_idx + len(word)
        word_spans.append((start_idx, end_idx))
        char_idx = end_idx
    
    return words, word_spans

def text_to_bio(task):

    # extract sentence and annotations
    sentence = task["sentence"]
    annotations = task["annotations"]

    words, word_spans = __tokenize_word_level(sentence)

    # initialize all tags as being O
    bio_tags = ["O"] * len(words)
    
    # loop through the annotations
    for annotation in annotations:
        start_ann, end_ann = annotation["start"], annotation["end"]
        for idx, (start_idx, end_idx) in enumerate(word_spans):
            if start_idx == start_ann:
                bio_tags[idx] = "B-sg"
            elif start_idx > start_ann and end_idx <= end_ann:
                bio_tags[idx] = "I-sg"

    return bio_tags

def find_dictionary_matches(sentence, dictionary_regex):
    
    # first tokenize the sentence
    words, word_spans = __tokenize_word_level(sentence)

    bio_tags = ["O"] * len(words)

    for match in re.finditer(dictionary_regex, sentence, re.IGNORECASE):
        start_match, end_match = match.span()

        for idx, (start_idx, end_idx) in enumerate(word_spans):
            if start_idx == start_match:
                bio_tags[idx] = "B-sg"
            elif start_idx > start_match and end_idx <= end_match:
                bio_tags[idx] = "I-sg"
    
    return bio_tags


# ----------------------------------------------------------------------
# BERT-Based Models for NER
# ----------------------------------------------------------------------

def tokenization_labelling(text, entities, tokenizer, tag2id, max_len):

    # get the encoding of the sentence
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True,
                         max_length=max_len, padding="max_length")
    
    # create preliminary list with O tags for all tokens
    tags = ["O"] * len(encoding.offset_mapping)

    # loop over annotations and extract start and end index as well as the given tag
    for ent in entities:
        start, end = ent["start"], ent["end"]
        ent_tag = ent["tag"][0:2]

        # loop over all tokens in the sentence and check for overlap
        for idx, (token_start, token_end) in enumerate(encoding.offset_mapping):
            # continue if it is a special token
            if token_start == token_end == 0:
                continue
            # check for overlap (overlap checking strictly necessary for deberta)
            if token_end > start and token_start < end:
                # assign B-tag if start is equal or smaller (smaller if deberta)
                if token_start <= start:
                    tags[idx] = f"B-{ent_tag}"
                # otherwise it is an inside token
                else:
                    tags[idx] = f"I-{ent_tag}"

    # extract the word ids
    word_ids = encoding.word_ids()

    # ensure propagate B-tags are propagated to all subwords of the same word (only actually relevant for deberta)
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        # if token has B-tag ensure that all other tokens of the same word get I-tag
        if tags[idx].startswith("B-"):
            for j, wid2 in enumerate(word_ids):
                if wid2 == wid and j != idx:
                    tags[j] = f"I-{ent_tag}"

    # convert tags to IDs, masking special tokens
    tag_ids = [-100 if wid is None else tag2id.get(tag, tag2id["O"])
               for tag, wid in zip(tags, encoding.word_ids())]

    return encoding["input_ids"], encoding["attention_mask"], tag_ids, word_ids

class TokenDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, max_len):
        self.dataset = []
        self.max_len = max_len

        for task in data:
            # get the sentence and all annotations
            text = task["sentence"]
            spans = task["annotations"]

            # tokenize and get all ids
            input_ids, attention_mask, tag_ids, word_ids = tokenization_labelling(text, spans, tokenizer, tag2id, self.max_len)

            # add everything to the dataset list
            self.dataset.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
                "word_ids": word_ids})
  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_bert_ner(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    tag_ids = torch.stack([item["tag_ids"] for item in batch])
    word_ids = [item["word_ids"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "tag_ids": tag_ids,
        "word_ids": word_ids
    }


def __clear_cache_models(model, optimizer, device):
    del model
    del optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

class EarlyStopping:
    def __init__(self, patience, min_delta=0.0001, save_model=True, path='checkpoint.pt', printoption=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = None
        self.best_epoch = None
        self.early_stop = False
        self.path = path
        self.printoption = printoption
        self.save_model = save_model

    def __call__(self, current_f1, model, epoch):
        if self.best_f1 is None:
            self.best_f1 = current_f1
            self.best_epoch = epoch+1
            self.save_checkpoint(current_f1, model)
        elif current_f1 < self.best_f1 - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if current_f1 > self.best_f1:
                self.save_checkpoint(current_f1, model)
                self.best_f1 = current_f1
                self.best_epoch = epoch+1
                self.counter = 0

    def save_checkpoint(self, current_f1, model):
        if self.save_model:
            torch.save(model.state_dict(), self.path)
        if self.printoption:
            print(f'Validation F1 increased ({self.best_f1:.6f} --> {current_f1:.6f}).  Saving model ...')

# define functions needed for training and hyperparameter tuning
def train_bert(train_dataloader, model, optimizer, epochs, device, which_task):
    
    # set the model to training mode
    model.train()

    # loop through epochs
    for epoch in range(epochs):

        # print the epoch number
        print(f"Epoch {epoch + 1}/{epochs}")

        # initialize training loss for the epoch
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        # loop through each batch
        for batch in progress_bar:

            # move all batch data to respective device
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            if which_task == "ner":
                labels = batch["tag_ids"].to(device)
            elif which_task == "stance":
                labels = batch["label"].to(device)

            # clear the old gradient
            optimizer.zero_grad()

            # run data through the model and save the outputs, use mps with mixed precision (16 bit floating point for forward pass)
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                # save the loss and add to the total loss for the epoch
                loss = outputs.loss
                total_loss += loss.item()

            # compute gradients by backpropagation, cap gradients to prevent gradient explosion
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            # update the model weights based on the gradient and update the progress bar
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        # get the average training loss per batch and print
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.4f}")


def tune_bert_ner_optuna(train_dataset, val_dataset, collate_fn, model_name, tag2id, id2tag, device, early_stopper, params):
    
    print(f"\nTrial with params: {params}")

    # extract the parameters with which to run this trial
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    # run always for 10 epochs
    epochs = 10

    # instantiate empty lists to save the development of losses and F1 score
    train_losses = []
    val_losses = []
    f1_scores_train = []
    f1_scores_val = []

    # set up a new model instance and optimizer
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
        ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # create new data loaders for this trial
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # set model to training mode and run the training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=tag_ids)
                loss = outputs.loss
                total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.4f}")

        # run model in inference mode to get loss and seqeval f1-score for the training dataset
        all_true, all_pred, train_loss = run_testset_ner(
            model=model, test_dataloader=train_dataloader, id2tag=id2tag, device=device, for_metric="seqeval"
            )
        train_losses.append(train_loss)
        metrics = evaluate_seqeval(all_true, all_pred)
        train_f1 = metrics["f1"]
        f1_scores_train.append(train_f1)

        # do the same to get loss and seqeval f1-score for the validation set
        all_true, all_pred, val_loss = run_testset_ner(
            model=model, test_dataloader=val_dataloader, id2tag=id2tag, device=device, for_metric="seqeval"
            )
        val_losses.append(val_loss)
        metrics = evaluate_seqeval(all_true, all_pred)
        val_f1 = metrics["f1"]
        f1_scores_val.append(val_f1)

        # feed validation loss into early stopping object to save and stop model training if necessary
        early_stopper(val_f1, model, epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    # delete all objects and empty the cache to free up space
    __clear_cache_models(model=model, optimizer=optimizer, device=device)

    return early_stopper.best_f1, early_stopper.best_epoch, train_losses, val_losses, f1_scores_train, f1_scores_val

# ----------------------------------------------------------------------
# BERT-Based Models for Stance Detection
# ----------------------------------------------------------------------
class StanceDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128):
        self.dataset = []
        for item in data:
            sentence = item["sentence"]
            target = item["group"]
            stance = item["stance"]
            label = label2id[stance]
            
            encoded = tokenizer(
                sentence,
                target,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
                )
            self.dataset.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)
                })
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

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
    

def tune_bert_stance_optuna(train_dataset, val_dataset, model_name, label2id, id2label, device, early_stopper, params):
    
    print(f"\nTrial with params: {params}")

    num_labels = len(label2id)

    # extract the parameters with which to run this trial
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    # set epochs to 10 for all trials
    epochs = 10

    # instantiate empty lists to save the development of losses and F1 score
    train_losses = []
    val_losses = []
    f1_scores_train = []
    f1_scores_val = []

    # set up a new model instance and optimizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # create new data loaders for this trial
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # set model to training mode and run the training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.4f}")

        # run model in inference mode to get loss and seqeval f1-score for the training dataset
        true_labels, pred_labels, train_loss = run_testset_stance(
            model=model, test_dataloader=train_dataloader, device=device
            )
        train_losses.append(train_loss)
        metrics = sklearn_classification_report(
            [id2label[i] for i in true_labels],
            [id2label[i] for i in pred_labels],
            output_dict=True
            )
        train_f1 = metrics["macro avg"]["f1-score"]
        f1_scores_train.append(train_f1)

        # do the same to get loss and seqeval f1-score for the validation set
        true_labels, pred_labels, val_loss = run_testset_stance(
            model=model, test_dataloader=val_dataloader, device=device
            )
        metrics = sklearn_classification_report(
            [id2label[i] for i in true_labels],
            [id2label[i] for i in pred_labels],
            output_dict=True
            )
        val_f1 = metrics["macro avg"]["f1-score"]
        f1_scores_val.append(val_f1)

        # feed validation loss into early stopping object to save and stop model training if necessary
        early_stopper(val_f1, model, epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    # delete all objects and empty the cache to free up space
    __clear_cache_models(model=model, optimizer=optimizer, device=device)

    return early_stopper.best_f1, early_stopper.best_epoch, train_losses, val_losses, f1_scores_train, f1_scores_val


def tune_bert_nli_stance_optuna(train_data_original, train_data_neu_aug, train_data_neg_aug, val_dataset, model_name, device, early_stopper, params):
    
    print(f"\nTrial with params: {params}")

    # extract the parameters with which to run this trial
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    # set epochs to 8 for all trials
    epochs = 8

    # instantiate empty lists to save the development of losses and F1 score
    f1_scores_train = []
    f1_scores_val = []

    # set up a new model instance and optimizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    label_to_id = model.config.label2id

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create datasets with oversampled data
    train_dataset = StanceNLIDataset(train_data_original, tokenizer, max_len=128, label2id=label_to_id)
    train_dataset_neu_aug = StanceNLIDataset(train_data_neu_aug, tokenizer, max_len=128, label2id=label_to_id)
    train_dataset_neg_aug = StanceNLIDataset(train_data_neg_aug, tokenizer, max_len=128, label2id=label_to_id)
    train_dataset = ConcatDataset([train_dataset, train_dataset_neu_aug, train_dataset_neg_aug])
    
    # create new data loader for this trial
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set model to training mode and run the training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.4f}")

        # run model in inference mode to get loss and seqeval f1-score for the training dataset
        true_labels, pred_labels = evaluate_nli_stance(
            model=model, data=train_data_original, tokenizer=tokenizer, device=device
            )
        metrics = sklearn_classification_report(true_labels, pred_labels, output_dict=True)
        train_f1 = metrics["macro avg"]["f1-score"]
        f1_scores_train.append(train_f1)

        # do the same to get loss and seqeval f1-score for the validation set
        true_labels, pred_labels = evaluate_nli_stance(
            model=model, data=val_dataset, tokenizer=tokenizer, device=device
            )
        metrics = sklearn_classification_report(true_labels, pred_labels, output_dict=True)
        val_f1 = metrics["macro avg"]["f1-score"]
        f1_scores_val.append(val_f1)

        # feed validation loss into early stopping object to save and stop model training if necessary
        early_stopper(val_f1, model, epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    # delete all objects and empty the cache to free up space
    __clear_cache_models(model=model, optimizer=optimizer, device=device)

    return early_stopper.best_f1, early_stopper.best_epoch, f1_scores_train, f1_scores_val