import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from custom_evaluation import evaluate_seqeval

# ----------------------------------------------------------------------
# Dictionary Baseline
# ----------------------------------------------------------------------

# function to convert the annotations to bio tags on the word level
def tokenize_word_level(sentence):
    
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

    words, word_spans = tokenize_word_level(sentence)

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


# ----------------------------------------------------------------------
# BERT-Based Models
# ----------------------------------------------------------------------

# define functions needed for training and hyperparameter tuning
def train_bert_ner(train_dataloader, model, optimizer, epochs, device):
    
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
            tag_ids = batch["tag_ids"].to(device)

            # clear the old gradient
            optimizer.zero_grad()

            # run data through the model and save the outputs, use mps with mixed precision (16 bit floating point for forward pass)
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=tag_ids)
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

def sample_hyperparams(search_space):
    return {k: random.choice(v) for k, v in search_space.items()}
    
def tune_bert_ner(num_trials, search_space, train_dataset, val_dataset, collate_fn, model_name, tag2id, id2tag, device):

    print(f"\nStarting search for model {model_name}")
    print("-"*100)

    best_f1 = 0
    best_params = None

    for trial in range(num_trials):

        best_epoch_f1 = 0
        patience_counter = 0
        patience = 3
        params = sample_hyperparams(search_space)
        print(f"\nTrial {trial+1} with params: {params}")

        # get all sampled hyperparameters explicitly
        lr = params["lr"]
        weight_decay = params["weight_decay"]
        batch_size = params["batch_size"]
        epochs = params["epochs"]

        model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=len(tag2id),
                                                        id2label=id2tag,
                                                        label2id=tag2id).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

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
                tag_ids = batch["tag_ids"].to(device)

                # clear the old gradient
                optimizer.zero_grad()

                # run data through the model and save the outputs, use mps with mixed precision (16 bit floating point for forward pass)
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=tag_ids)
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

            # After training, evaluate F1 on the validation set
            metrics = evaluate_seqeval(model, val_dataloader, id2tag, device)
            val_f1 = metrics["f1"]
        
            if val_f1 > best_epoch_f1:
                best_epoch_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_params = params

        print("-"*100)

    print(f"Best F1: {best_f1:.4f} with params: {best_params}")

    return best_f1, best_params