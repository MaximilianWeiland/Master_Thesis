import pandas as pd
import sys
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

# set directory to the project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# import custom classes and functions
from utils.clustering import (
    DatasetMaskHN, ModelMask,
    generate_triplets_hn,
    train_triplet_mask_hard
)

# load the dictionary terms
groups_dictionary = pd.read_csv("01_data/classification/dictionary/groups_dictionary_cl.csv")

# create a single list of all dictionary entries paired with their category
mention_list = []
category_list = []
categories = groups_dictionary.columns
cat2id = {c: i for i, c in enumerate(categories)}
for category in categories:
    for idx, row in groups_dictionary[[category]].iterrows():
        if not pd.isna(row.values):
            entry = row.values[0]
            mention_list.append(entry.strip())
            category_list.append(category)

category_list = [cat for cat in category_list]

df = pd.DataFrame({'mention': mention_list, 'category': category_list})

mention_list = df["mention"].to_list()
category_list = df["category"].to_list()

# define early stopping class measuring loss development
class EarlyStopping:
    def __init__(self, patience, min_delta=0.0001, save_model=True, path='checkpoint.pt', printoption=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.printoption = printoption
        self.save_model = save_model

    def __call__(self, current_loss, model):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(current_loss, model)
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if current_loss < self.best_loss:
                self.save_checkpoint(current_loss, model)
                self.best_loss = current_loss
                self.counter = 0

    def save_checkpoint(self, current_loss, model):
        if self.save_model:
            torch.save(model.state_dict(), self.path)
        if self.printoption:
            print(f'Loss decreased ({self.best_loss:.6f} --> {current_loss:.6f}).  Saving model ...')

# run training loop
device = 'mps' if torch.mps.is_available() else 'cpu'
print(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
output_path = "model_checkpoint/checkpoint.pt"
early_stopper = EarlyStopping(patience=10, save_model=True, path=output_path, printoption=True)
model = ModelMask(tokenizer=tokenizer, pretrained_model_name="bert-base-uncased").to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 30
for epoch in range(num_epochs):
    triplets = generate_triplets_hn(df)
    dataset = DatasetMaskHN(triplets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    loss = train_triplet_mask_hard(model, dataloader, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    early_stopper(loss, model)
    if early_stopper.early_stop:
      print("Early stopping triggered")
      break