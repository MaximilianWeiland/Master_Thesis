import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Sequence

class DatasetMaskHN(Dataset):
    """
    PyTorch Dataset class to store triplet data used to train a constrastive learning model via triplet margin loss.
    """
    def __init__(
            self,
            triplets: List[Tuple[str, str, str]],
            tokenizer: Any,
            max_length: int=128
    ) -> None:
        """
        Initialization of class object.

        Args:
            triplets (List[Tuple[str, str, str]]): List of tuples containing the positive examples for the triplet.
            tokenizer (Any): BERT-compatible tokenizer.
            max_length (int): Maximum sequence length the tokenizer will encode.

        Returns:
            None
        """
        self.data: List[Tuple[str, str, str]] = triplets
        self.tokenizer: Any = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        """
        Returns the number of triplets in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns item of the dataset specified by index.
        """
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

        # squeeze batch dimension
        anchor_input_ids: Tensor = anchor_enc['input_ids'].squeeze(0)
        anchor_attention_mask: Tensor = anchor_enc['attention_mask'].squeeze(0)
        positive_input_ids: Tensor = positive_enc['input_ids'].squeeze(0)
        positive_attention_mask: Tensor = positive_enc['attention_mask'].squeeze(0)

        return {
            "anchor_input_ids": anchor_input_ids,
            "anchor_attention_mask": anchor_attention_mask,
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "category": category
        }
    
class ModelMask(nn.Module):
    """
    Encoder model for contrastive learning that extracts representations
    from [MASK] token positions in a pretrained Transformer model.

    The model encodes input sequences, averages hidden states at all [MASK]
    positions (if present), projects them into a lower-dimensional space,
    and applies L2 normalization.
    """
    def __init__(
            self,
            tokenizer: Any,
            pretrained_model_name: str='bert-base-uncased',
            proj_dim: int=128
    ) -> None:
        """
        Initialization of the class object.
        
        Args:
            tokenizer (Any): BERT-compatible tokenizer.
            pretrained_model_name (str): Model name of the BERT model to use for finetuning.
            proj_dim (int): Dimensionality of the hidden space that should be used for final projection.

        Returns:
            None
        """
        super().__init__()
        self.encoder: nn.Module = AutoModel.from_pretrained(pretrained_model_name)
        self.mask_id: int = tokenizer.mask_token_id
        self.proj_dim: int = proj_dim
        self.hidden_size: int = self.encoder.config.hidden_size
        self.projector: nn.Module = nn.Sequential(
            nn.Linear(self.hidden_size, proj_dim))

    def extract_mask_embedding(
            self,
            input_ids: Tensor,
            hidden_states: Tensor
    ) -> Tensor:
        """
        Extracts a sentence-level embedding by averaging hidden states at all [MASK] token positions.

        Args:
            input_ids (Tensor): Token IDs of shape (batch_size, seq_len).
            hidden_states (Tensor): Hidden states of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tensor: Mask-based embeddings of shape (batch_size, hidden_size).
        """
        # get the positions of all MASK tokens (should only be one)
        mask_positions = (input_ids == self.mask_id)
        # get the batch size
        batch_size = input_ids.size(0)

        # create empty list to store embedding
        outputs: List[Tensor] = []
        for i in range(batch_size):
            positions = mask_positions[i]
            if positions.any():
                emb = hidden_states[i][positions].mean(dim=0)
            else:
                emb = hidden_states[i][0]
            outputs.append(emb)

        return torch.stack(outputs)

    def encode(
            self,
            input_ids: Tensor,
            attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encodes input sequences and produces both raw and projected embeddings.

        Args:
            input_ids (Tensor): Token IDs of shape (batch_size, seq_len).
            attention_mask (Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            Tuple[Tensor, Tensor]:
                - h: Raw mask-based embeddings of shape (batch_size, hidden_size)
                - z: L2-normalized projected embeddings of shape (batch_size, proj_dim)
        """
        # run input through the model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # extract raw embedding at the mask token
        h: Tensor = self.extract_mask_embedding(input_ids, outputs.last_hidden_state)
        # run MASK embedding through linear projection layer and normalize
        z: Tensor = self.projector(h)
        z: Tensor = F.normalize(z, p=2, dim=1)

        return h, z
    
def generate_triplets_hn(
        df: pd.DataFrame
) -> List[Tuple[str, str, str]]:
    """
    Generates anchor and positive example pairs for constrastive learning with triplet loss.
    The negative example needs to be added with hard negative mining.

    Args:
        df (pd.DataFrame): Dataframe containing all categories with respective terms.
    
    Returns:
        List[Tuple[str, str, str]]: List of tuples containing anchor term, positive example and category.

    """
    # create a copy of the df and lowercase the category entries
    df: pd.DataFrame = df.copy()
    df['category_lower'] = df['category'].str.lower()

    # get all unique terms
    categories = df['category_lower'].unique()

    # create mapping from category header to all its entries
    cat2mentions: Dict[str, List[str]] = {c: df[df['category_lower']==c]['mention'].tolist() for c in categories}

    # create empty list to store triplet data in
    triplets: List[Tuple[str, str, str]] = []

    # loop over all categories
    for category, mentions in cat2mentions.items():
        # loop over all entries and treat each entry as anchor
        for anchor in mentions:
            # randomly pick another entry as positive example and append to the list
            pos_candidates: List[str] = [m for m in mentions if m != anchor]
            if not pos_candidates:
                continue
            positive: str = random.choice(pos_candidates)
            triplets.append((anchor, positive, category))
    return triplets

def mine_batch_hard_negatives(
        anchor_emb: Tensor,
        labels: Sequence[str]
) -> List[int]:
    """
    Performs batch-hard negative mining based on Euclidean distance.

    For each anchor embedding in the batch, the function selects the index of the
    closest embedding that belongs to a different class.

    Args:
        anchor_emb (Tensor): Tensor of shape (B, D) containing anchor embeddings.
        labels (Sequence[int]): Sequence of length B containing class labels for each embedding.

    Returns:
        List[int]: List of length B with indices for hard negative examples for the corresponding anchor.
    """

    # get the number of samples in the batch
    B: int = len(labels)
    # pairwise Euclidean distances between all embeddings
    dists: Tensor = torch.cdist(anchor_emb, anchor_emb, p=2)
    # create empty list to store indices for the hard negative examples in
    hard_neg_indices: List[int] = []

    # loop over indices of all samples in the batch
    for i in range(B):
        # get the category label of the current sample
        label: str = labels[i]
        # create mask and set mask for the current sample to False
        mask: Tensor = torch.tensor([l != label for l in labels], device=anchor_emb.device)
        mask[i] = False

        # if no negatives are available
        if mask.sum() == 0:
            hard_neg_indices.append(i)
            continue
        
        # get all distances and indices between current sample and all valid negatives
        neg_dists: Tensor = dists[i][mask]
        neg_indices: Tensor = torch.where(mask)[0]
        # select the negative example which is closest to the anchor and append to list
        hard_neg_idx: Tensor = neg_indices[neg_dists.argmin()]
        hard_neg_indices.append(int(hard_neg_idx))

    return hard_neg_indices

def train_triplet_mask_hard(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device
) -> float:
    """
    Finetunes an encoder BERT model on contrastive learning objective for one epoch.
    Weights are updated to maximize distance between anchor and negative example and
    minimize distance between anchor and the positive example.

    Args:
        model (nn.Module): BERT model to finetune.
        dataloader (DataLoader): PyTorch DataLoader containing the triplet data.
        optimizer (optim.Optimizer): Optimizer to update the weights.
        device (torch.device): Device (cuda/mps/cpu) on which the model should get finetuned.

    Returns:
        float: Average batch loss for the epoch.
    """
    # set the model to training mode
    model.train()
    # track loss with variable which is set to 0
    total_loss: float = 0.0
    # set the loss function
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    progress_bar = tqdm(dataloader, desc="Training")

    # loop over all batches
    for batch in progress_bar:

        # zero the gradient
        optimizer.zero_grad()

        # move all tensors to device
        anchor_input_ids: Tensor = batch['anchor_input_ids'].to(device)
        anchor_attention_mask: Tensor = batch['anchor_attention_mask'].to(device)
        positive_input_ids: Tensor = batch['positive_input_ids'].to(device)
        positive_attention_mask: Tensor = batch['positive_attention_mask'].to(device)
        labels = batch['category']

        # encode anchors and positives
        _, anchor_emb = model.encode(anchor_input_ids, anchor_attention_mask)
        _, pos_emb = model.encode(positive_input_ids, positive_attention_mask)

        # mine hard negatives within batch
        neg_indices: List[int] = mine_batch_hard_negatives(anchor_emb, pos_emb, labels)

        # select negative embeddings using mined indices
        neg_input_ids = anchor_input_ids[neg_indices]
        neg_attention_mask = anchor_attention_mask[neg_indices]
        _, neg_emb = model.encode(neg_input_ids, neg_attention_mask)

        # compute triplet loss and update the weights
        loss: Tensor = loss_fn(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        # update the total loss based on batch loss
        total_loss += loss.item()

    return total_loss / len(dataloader)