# import necessary libraries
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_len=32):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_token = tokenizer.mask_token

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        anchor_text = f"Social group of {anchor} is: {self.mask_token}."
        positive_text = f"Social group of {positive} is: {self.mask_token}."
        negative_text = f"Social group of {negative} is: {self.mask_token}."


        anchor_enc = self.tokenizer(anchor_text, padding='max_length', truncation=True,
                                    max_length=self.max_len, return_tensors='pt')
        positive_enc = self.tokenizer(positive_text, padding='max_length', truncation=True,
                                      max_length=self.max_len, return_tensors='pt')
        negative_enc = self.tokenizer(negative_text, padding='max_length', truncation=True,
                                      max_length=self.max_len, return_tensors='pt')
        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_input_ids': positive_enc['input_ids'].squeeze(0),
            'positive_attention_mask': positive_enc['attention_mask'].squeeze(0),
            'negative_input_ids': negative_enc['input_ids'].squeeze(0),
            'negative_attention_mask': negative_enc['attention_mask'].squeeze(0),
        }
    
class TripletDatasetHN(Dataset):
    def __init__(self, triplets, tokenizer, max_length=128):
        self.data = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_text, positive_text, category = self.data[idx]

        anchor_enc = self.tokenizer(
            anchor_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        positive_enc = self.tokenizer(
            positive_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Squeeze batch dimension
        anchor_input_ids = anchor_enc['input_ids'].squeeze(0)
        anchor_attention_mask = anchor_enc['attention_mask'].squeeze(0)
        positive_input_ids = positive_enc['input_ids'].squeeze(0)
        positive_attention_mask = positive_enc['attention_mask'].squeeze(0)

        return {
            "anchor_input_ids": anchor_input_ids,
            "anchor_attention_mask": anchor_attention_mask,
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "category": category
        }
    
class ModelMask(nn.Module):
    def __init__(self, tokenizer, pretrained_model_name='bert-base-uncased', proj_dim=128):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model_name)
        self.mask_id = tokenizer.mask_token_id
        self.proj_dim = proj_dim
        self.hidden_size = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, proj_dim))

    def extract_mask_embedding(self, input_ids, hidden_states):
        mask_positions = (input_ids == self.mask_id)
        batch_size = input_ids.size(0)

        outputs = []
        for i in range(batch_size):
            positions = mask_positions[i]
            if positions.any():
                emb = hidden_states[i][positions].mean(dim=0)
            else:
                emb = hidden_states[i][0]
            outputs.append(emb)

        return torch.stack(outputs)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.extract_mask_embedding(input_ids, outputs.last_hidden_state)
        z = self.projector(h)
        z = F.normalize(z, p=2, dim=1)
        return z
    
def generate_triplets(df):

    categories = df['category'].unique()
    cat2mentions = {c: df[df['category']==c]['mention'].tolist() for c in categories}

    triplets = []

    for category, mentions in cat2mentions.items():
        for anchor in mentions:
            pos_candidates = [m for m in mentions if m != anchor]
            positive = random.choice(pos_candidates)
            negative_category = random.choice([c for c in categories if c != category])
            negative = random.choice(cat2mentions[negative_category])
            triplets.append((anchor, positive, negative))
    return triplets

def generate_triplets_heuristic_hn(df, close_map, hard_ratio=0.5):

    df = df.copy()
    df['category_lower'] = df['category'].str.lower()
    categories = df['category_lower'].unique()
    cat2mentions = {
        c: df[df['category_lower'] == c]['mention'].tolist()
        for c in categories
    }

    triplets = []

    for category, mentions in cat2mentions.items():

        for anchor in mentions:
            pos_candidates = [m for m in mentions if m != anchor]
            positive = random.choice(pos_candidates)
            if category in close_map and random.random() < hard_ratio:
                negative_category = random.choice(close_map[category])
            else:
                negative_category = random.choice([c for c in categories if c != category])
            negative = random.choice(cat2mentions[negative_category])
            triplets.append((anchor, positive, negative))

    return triplets


def generate_triplets_hn(df):
    df = df.copy()
    df['category_lower'] = df['category'].str.lower()
    categories = df['category_lower'].unique()
    cat2mentions = {c: df[df['category_lower']==c]['mention'].tolist() for c in categories}
    triplets = []
    for category, mentions in cat2mentions.items():
        for anchor in mentions:
            pos_candidates = [m for m in mentions if m != anchor]
            if not pos_candidates:
                continue
            positive = random.choice(pos_candidates)
            triplets.append((anchor, positive, category))
    return triplets


def mine_batch_hard_negatives(anchor_emb, pos_emb, labels):
    B = len(labels)
    dists = torch.cdist(anchor_emb, anchor_emb, p=2)
    hard_neg_indices = []

    for i in range(B):
        label = labels[i]
        mask = torch.tensor([l != label for l in labels], device=anchor_emb.device)
        mask[i] = False

        if mask.sum() == 0:
            hard_neg_indices.append(i)
            continue

        neg_dists = dists[i][mask]
        neg_indices = torch.where(mask)[0]
        hard_neg_idx = neg_indices[neg_dists.argmin()]
        hard_neg_indices.append(int(hard_neg_idx))

    return hard_neg_indices


def train_triplet(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        # encode anchor, positive and negative phrase
        anchor_emb = model.encode(batch["anchor_input_ids"], batch["anchor_attention_mask"])
        pos_emb = model.encode(batch["positive_input_ids"], batch["positive_attention_mask"])
        neg_emb = model.encode(batch["negative_input_ids"], batch["negative_attention_mask"])

        # compute triplet loss, backpropagate and update weights
        loss = loss_fn(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_triplet_hn(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        # Move tensors to device
        anchor_input_ids = batch['anchor_input_ids'].to(device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        labels = batch['category']  # labels can stay as text/strings

        # Encode anchors and positives
        anchor_emb = model.encode(anchor_input_ids, anchor_attention_mask)
        pos_emb = model.encode(positive_input_ids, positive_attention_mask)

        # Mine hard negatives within batch
        neg_indices = mine_batch_hard_negatives(anchor_emb, pos_emb, labels)

        # Select negative embeddings using mined indices
        neg_input_ids = anchor_input_ids[neg_indices]
        neg_attention_mask = anchor_attention_mask[neg_indices]
        neg_emb = model.encode(neg_input_ids, neg_attention_mask)

        # Compute triplet loss
        loss = loss_fn(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)