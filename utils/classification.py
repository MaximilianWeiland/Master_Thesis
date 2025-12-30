import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from sklearn.metrics import classification_report as sklearn_classification_report
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import Tensor, nn, optim
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import gc
from typing import List, Tuple, Dict, Any, Pattern, Callable
from utils.evaluation import evaluate_seqeval, run_testset_ner, run_testset_stance, evaluate_nli_stance

# ----------------------------------------------------------------------
# Dictionary Baseline
# ----------------------------------------------------------------------

def __tokenize_word_level(sentence: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Tokenizes a sentence into words and punctuation and computes character-level
    start and end indices for each token.

    The sentence is split using a regular expression that preserves punctuation
    as separate tokens. For each token, the function determines its start and end
    position within the original sentence.

    Args:
        sentence (str): The input sentence to tokenize.

    Returns:
        Tuple[List[str], List[Tuple[int, int]]]:
            - A list of tokens (words and punctuation).
            - A list of (start_index, end_index) tuples corresponding to each token.
    """
    # empty list to store all word indices and tracking variable
    word_spans: List[Tuple[int, int]] = []
    char_idx: int = 0

    # split sentence via regex which ensures to also split at punctuation
    words: List[str] = re.findall(r"\w+|'\w+|[^\w\s]", sentence)

    # loop over words and determine start and end index
    for word in words:
        start_idx: int = sentence.find(word, char_idx)
        end_idx: int = start_idx + len(word)
        word_spans.append((start_idx, end_idx))
        char_idx = end_idx
    
    return words, word_spans

def text_to_bio(task: Dict[str, Any]) -> List[str]:
    """
    Converts span-based annotations into BIO-tags at the word level.

    Tokenizes the sentence into words. Then, loops over all social group annoations
    and converts the words into BIO-tags using the annotation span information.

    Args:
        task (Dict[str, Any]): A dictionary containing:
            - "sentence" (str): The input text.
            - "annotations" (List[Dict[str, int]]): Annotation spans with "start" and "end" character indices.
    
    Returns:
        List[str]: List of BIO-tags on the word level for the given sentence.
    """
    # extract sentence and annotations
    sentence: str = task["sentence"]
    annotations: List[Dict[str, Any]] = task["annotations"]

    # tokenize the sentence and extract word-level span indices
    words, word_spans = __tokenize_word_level(sentence)

    # initialize all tags as being O
    bio_tags: List[str] = ["O"] * len(words)
    
    # loop through the annotations and label each word
    for annotation in annotations:
        start_ann: int = annotation["start"]
        end_ann: int = annotation["end"]
        for idx, (start_idx, end_idx) in enumerate(word_spans):
            if start_idx == start_ann:
                bio_tags[idx] = "B-sg"
            elif start_idx > start_ann and end_idx <= end_ann:
                bio_tags[idx] = "I-sg"

    return bio_tags

def find_dictionary_matches(
        sentence: str,
        dictionary_regex: Pattern[str]
) -> List[str]:
    """
    Finds word matches between input sentence and dictionary containing social group terms and returns sentence as BIO-tags.

    Tokenizes input sentence. Then, applies a regular expression containing all dictionary terms.
    Converts words into BIO-tags on the word level based on the matches.

    Args:
        sentence (str): Input sentence to process.
        dictionary_regex (Pattern[str]): Regular expression containing all dictionary terms.

    Returns:
        List[str]: BIO-tags based on the dictionary matches on the word level.
    """
    # tokenize the sentence
    words, word_spans = __tokenize_word_level(sentence)

    # instantiate list of O-tags with length of the sentence
    bio_tags: List[str] = ["O"] * len(words)

    # iterate over all dictionary matches and replace BIO-tags for each of them
    for match in re.finditer(dictionary_regex, sentence, re.IGNORECASE):
        start_match: int
        end_match: int
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

def tokenization_labelling(
    text: str,
    entities: List[Dict[str, Any]],
    tokenizer: Any,
    tag2id: Dict[str, int],
    max_len: int
) -> Tuple[List[int], List[int], List[int], List[int | None]]:
    """
    Encodes an input sentence using a (BERT) tokenizer and maps each token to a BIO-tag and word id.

    An input sentence gets tokenized using a (usually) BERT tokenizer. Based on annotation data about social groups
    for all of these tokens a BIO-tag is constructed. Also returns the word id of each token.

    Args:
        text (str): Input sentence to process.
        entities (List[Dict[str, Any]]): List of social group annotations.
            - "start" (int): Start character index.
            - "end" (int): End character index.
            - "tag" (str): Entity label.
        tokenizer (Any): Tokenizer object which is usally a BERT tokenizer from transformers package
        tag2id (Dict[str, int]): Dictionary to convert BIO-tags to ids that can get processed by the language model.
        max_len (int): Maximum length of the string the tokenizer will process.

    Returns:
        Tuple[List[int], List[int], List[int], List[int | None]]:
            - Token ids of the encoded sentence.
            - Attention mask of the encoded sentence.
            - Ids of the BIO-tags specifying entity information.
            - Word ids for each token.
    """
    # get the encoding of the sentence
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True,
                         max_length=max_len, padding="max_length")
    
    # create preliminary list with O tags for all tokens
    tags: List[str] = ["O"] * len(encoding.offset_mapping)

    # loop over annotations and extract start and end index as well as the given tag
    for ent in entities:
        start: int = ent["start"]
        end: int = ent["end"]
        ent_tag: str = ent["tag"][:2]

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
    word_ids: List[int | None] = encoding.word_ids()

    # ensure that I-tags are propagated to all subwords of the same word if it already has B-tag (only actually relevant for deberta)
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        # if token has B-tag ensure that all other tokens of the same word get I-tag
        if tags[idx].startswith("B-"):
            for j, wid2 in enumerate(word_ids):
                if wid2 == wid and j != idx:
                    tags[j] = f"I-{ent_tag}"

    # convert tags to IDs, masking special tokens
    tag_ids: List[int] = [-100 if wid is None else tag2id.get(tag, tag2id["O"])
               for tag, wid in zip(tags, encoding.word_ids())]

    return encoding["input_ids"], encoding["attention_mask"], tag_ids, word_ids

class TokenDataset(Dataset):
    """
    PyTorch Dataset for token-level sequence labeling with BIO tags.

    Each item in the dataset corresponds to a single tokenized sentence
    and contains input IDs, attention mask, tag IDs, and word ID mappings.
    """

    def __init__(
            self,
            data: List[Dict[str, Any]],
            tokenizer: Any,
            tag2id: Dict[str, int],
            max_len: int
    ) -> None:
        """
        Initialization of the class object.

        Args:
            data (List[Dict[str, Any]]): List of dictionaries for each sentence/task.
                - "sentence" (str): The sentence to process.
                - "annotations" (List[Dict[str, Any]]): List of dictionaries specifying the social groups present in the sentence.
            tokenizer (Any): BERT tokenizer compatible with 'tokenization_labelling'.
            tag2id (Dict[str, int]): Dictionary to convert BIO-tags to ids that can get processed by the language model.
            max_len (int): Maximum length of the string the tokenizer will process.

        Returns:
            None
        """
        # create empty list in which all data will be stored
        self.dataset: List[Dict[str, Any]] = []
        self.max_len: int = max_len

        # loop over all sentence/task data
        for task in data:

            # get the sentence and all annotations
            text: str = task["sentence"]
            spans: List[Dict[str, Any]] = task["annotations"]

            # tokenize and get all ids
            input_ids, attention_mask, tag_ids, word_ids = tokenization_labelling(text, spans, tokenizer, tag2id, self.max_len)

            # add everything to the dataset list
            self.dataset.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
                "word_ids": word_ids})
  
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Retrieves specific sentence data based on index.
        """
        return self.dataset[idx]

def collate_bert_ner(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for a BERT NER dataset.

    Args:
        batch (List[Dict[str, Any]]): Batch on which collate function is applied.
            - "input_ids" (Tensor): Tensor of token IDs.
            - "attention_mask" (Tensor): Tensor of attention mask.
            - "tag_ids" (Tensor): Tensor of BIO labels.
            - "word_ids" (List[int | None]): List of word ids.

    Returns:
        Dict[str, Any]: Dictionary with the batched data.
            - "input_ids" (Tensor): Tensor of shape (batch_size, seq_len).
            - "attention_mask" (Tensor): Tensor of shape (batch_size, seq_len).
            - "tag_ids" (Tensor): Tensor of shape (batch_size, seq_len).
            - "word_ids" (List[List[int | None]]): List of word ids.
    """
    input_ids: Tensor = torch.stack([item["input_ids"] for item in batch])
    attention_masks: Tensor = torch.stack([item["attention_mask"] for item in batch])
    tag_ids: Tensor = torch.stack([item["tag_ids"] for item in batch])
    word_ids: List[List[int | None]] = [item["word_ids"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "tag_ids": tag_ids,
        "word_ids": word_ids
    }


def __clear_cache_models(
        model: nn.Module,
        optimizer:optim.Optimizer,
        device: torch.device
) -> None:
    """
    Deletes previous model artifacts clears the cache.

    Args:
        model (nn.Module): PyTorch model to delete.
        optimizer (optim.Optimizer): Optimizer associated with the model.
        device (torch.device): Device on which the model got executed (mps/cuda/cpu).
    
    Returns:
        None
    """
    # delete model, optimzer and collect cache
    del model
    del optimizer
    gc.collect()

    # clear gpu memory
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.tpye == "cuda":
        torch.cuda.empty_cache()

class EarlyStopping:
    """
    Class to track and store model performance during training.
    Can save the best model and trigger early stopping after defined number of epochs.
    """
    def __init__(
            self,
            patience: int,
            min_delta: float=0.0001,
            save_model: bool=True,
            path: str='checkpoint.pt',
            printoption: bool=False
    ) -> None:
        """
        Initialization of the class object.

        Args:
            patience (int): Number of epochs to wait until early stopping gets triggered.
            min_delta (float): Minimum change in the metric to qualify as improvement.
            save_model (bool): Boolean specifying if the model should get saved.
            path (str): The path under which to save the model.
            printoption (bool): Boolean specifying if the tracked model performance should get printed.
        """
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_f1: bool = None
        self.best_epoch: int = None
        self.early_stop: bool = False
        self.path: str = path
        self.printoption: bool = printoption
        self.save_model: bool = save_model

    def __call__(
            self,
            current_f1: float,
            model: nn.Module,
            epoch:int
    ) -> None:
        """
        Call method to update early stopping state based on the current F1 score.

        Args:
            current_f1 (float): F1-score for the current epoch.
            model (nn.Module): PyTorch model being trained.
            epoch (int): Current epoch in training.
        
        Returns:
            None
        """
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

    def save_checkpoint(
            self,
            current_f1: float,
            model: nn.Module
    ) -> None:
        """
        Saves the current model instance and prints current validation F1-score.

        Args:
            current_f1 (float): F1-score for the current epoch.
            model (nn.Module): PyTorch model being trained.

        Returns:
            None
        """
        if self.save_model:
            torch.save(model.state_dict(), self.path)
        if self.printoption:
            print(f'Validation F1 increased ({self.best_f1:.6f} --> {current_f1:.6f}).  Saving model ...')

def train_bert(
        train_dataloader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        device: torch.device,
        which_task: str
) -> None:
    """
    Trains a BERT model for specified number of epochs for either a token or sequence classification task.

    Takes in a dataloader, BERT model instance and optimizer. Trains the model for the specified number of epochs
    and prints the current loss development as well as the general training time. The model can be either a
    token or sequence classification model which gets specified under 'which_task'.

    Args:
        train_dataloader (DataLoader): PyTorch DataLoader holding the training data.
        model (nn.Module): BERT model to train.
        optimizer (optim.Optimizer): Optimizer used to train the model.
        epochs (int): Number of epochs to train the model.
        device (torch.device): Device (mps/cuda/cpu) on which the model should be trained.
        which_task (str): Specifying if the model should be trained for token classification (which_task = 'ner') or sequence classification (which_task = 'stance').

    Returns:
        None
    """
    # set the model to training mode
    model.train()

    # loop through epochs
    for epoch in range(epochs):

        # print the epoch number
        print(f"Epoch {epoch + 1}/{epochs}")

        # initialize training loss for the epoch
        total_loss: float = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        # loop through each batch
        for batch in progress_bar:

            # move all batch data to respective device
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_masks: Tensor = batch["attention_mask"].to(device)
            if which_task == "ner":
                labels: Tensor = batch["tag_ids"].to(device)
            elif which_task == "stance":
                labels: Tensor = batch["label"].to(device)

            # clear the old gradient
            optimizer.zero_grad()

            # run data through the model and save the outputs, use mps with mixed precision (16 bit floating point for forward pass)
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                # save the loss and add to the total loss for the epoch
                loss: Tensor = outputs.loss
                total_loss += loss.item()

            # compute gradients by backpropagation, cap gradients to prevent gradient explosion
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            # update the model weights based on the gradient and update the progress bar
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        # get the average training loss per batch and print
        avg_loss: float = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.4f}")


def tune_bert_ner_optuna(
        train_dataset: Dataset,
        val_dataset: Dataset,
        collate_fn: Callable,
        model_name: str,
        tag2id: Dict[str, int],
        id2tag: Dict[int, str],
        device: torch.device,
        early_stopper: Any,
        params: Dict[str, Any]
) -> Tuple[float, int, List[float],  List[float],  List[float],  List[float]]:
    """
    Runs one trial of optuna's hyperparameter optimization search for a BERT-based token classification model.

    Receives a hyperparameter combination and runs one trial with a fixed number of 10 epochs.
    If the model does not improve for a certain number of epochs, early stopping gets triggered and
    the epoch number with which the best F1-score got achieved gets returned.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        collate_fn (Callable): Collate function for the DataLoader.
        model_name (str): Name of the BERT model.
        tag2id (Dict[str, int]): Dictionary to map tag strings with corresponding ids.
        id2tag (Dict[int, str]): Dictionary to map ids with corresponding tag strings.
        device (torch.device): Device (cuda/mps/cuda) on which the model should get trained.
        early_stopper (Any): Early Stopping class which takes in current model performance and can trigger early stopping.
        params (Dict[str, Any]): Hyperparameters for this trial.
            - "lr" (float): learning rate
            - "weight_decay" (float): optimizer weight decay
            - "batch_size" (int): training batch size
        
    Returns:
        Tuple[float, int, List[float],  List[float],  List[float],  List[float]]:
            - Best validation F1-score.
            - The number of epochs. Can be less than 10 if early stopping got triggered.
            - List of training loss for each epoch.
            - List of validation loss for each epoch.
            - List of F1-scores for each epoch measured on training set.
            - List of F1-scores for each epoch measured on validation set.
    """
    
    print(f"\nTrial with params: {params}")

    # extract the parameters with which to run this trial
    lr: float = params["lr"]
    weight_decay: float = params["weight_decay"]
    batch_size: int = params["batch_size"]

    # run always for 10 epochs
    epochs: int = 10

    # instantiate empty lists to save the development of losses and F1 score
    train_losses: List[float] = []
    val_losses: List[float] = []
    f1_scores_train: List[float] = []
    f1_scores_val: List[float] = []

    # set up a new model instance and optimizer
    model: nn.Module = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
        ).to(device)
    optimizer: optim.Optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # create new data loaders for this trial
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # set model to training mode and run the training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss: float = 0.0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_masks: Tensor = batch["attention_mask"].to(device)
            tag_ids: Tensor = batch["tag_ids"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=tag_ids)
                loss: Tensor = outputs.loss
                total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss: float = total_loss / len(train_dataloader)
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
    """
    PyTorch dataset class for a BERT sequence classification model.
    The sequence is a sentence which gets classified regarding the stance it takes towards a
    social group which gets separated by the SEP token.
    """
    def __init__(
            self,
            data: List[Dict[str, str]],
            tokenizer: Any,
            label2id: Dict[str, int],
            max_len: int=128
    ) -> None:
        """
        Initialization of the class object.

        Args:
            data (List[Dict[str, Any]]): Input data each item contains.
                - "sentence" (str): Input sentence.
                - "group" (str): Social group that gets mentioned in the sentence.
                - "stance" (str): Stance which is taken towards the social group.
            tokenizer (Any): BERT-compatible tokenizer.
            label2id (Dict[str, int]): Dictionary mapping string labels to integer ids.
            max_len (int): Maximum sequence length to tokenize.

        Returns:
            None
        """

        # initialize empty list to store data in
        self.dataset: List[Dict[str, Tensor]] = []

        # loop over all items in the data
        for item in data:
            # extract sentence, target, stance and stance id
            sentence: str = item["sentence"]
            target: str = item["group"]
            stance: str = item["stance"]
            label: str = label2id[stance]
            # encode the sentence and the target and append to the dataset
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
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Returns single dataset item by index.
        """
        return self.dataset[idx]

class StanceNLIDataset(Dataset):
    """
    PyTorch dataset class to store data used for training a BERT sequence classification model
    used for Natural Language Inference tasks.

    For each sentence, three hypothesis versions are produced, for which only one of them is actually true.
    The information about which of these hypotheses entails the premise gets encoded by this class.
    """
    def __init__(
            self,
            raw_data: List[Dict[str, str]],
            tokenizer: Any,
            max_len: int,
            label2id: Dict[str, int]
    ) -> None:
        """
        Initialization of the class object.

        Args:
            raw_data (List[Dict[str, Any]]): Raw input data.
                - "sentence" (str): Input sentence.
                - "group" (str): Social group which gets mentioned in the sentence.
                - "stance" (str): Stance the sentence takes towards the social group.
            tokenizer (Any): BERT-compatible tokenizer.
            max_len (int): Maximum sequence length the tokenizer will handle.
            label2id (Dict[str, int]): Dictionary mapping from NLI label ("entailment" | "not_entailment") to numeric ids.

        Returns:
            None
        """

        # initialize empty list to store the data in
        self.dataset: List[Dict[str, Any]] = []

        # loop over all items in the raw dataset
        for item in raw_data:
            # extract sentence, target and stance towards target
            sentence: str = item["sentence"]
            target: str = item["group"]
            gold_stance: str = item["stance"]
            
            # construct all possible hypotheses
            hypotheses: Dict[str, str] = {
                "pos": f"The text is positive towards {target}.",
                "neg": f"The text is negative towards {target}.",
                "neutral": f"The text is neutral, or contains no stance, towards {target}."
            }

            # loop over all hypotheses
            for stance, hypothesis in hypotheses.items():
                # label is entailment if the hypothesis is true, otherwise no entailment
                label_text: str = "entailment" if stance == gold_stance else "not_entailment"
                # encode the sentence with the hypothesis and append to the dataset
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
                

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single dataset item by index.
        """
        return self.dataset[idx]
    

def tune_bert_stance_optuna(
        train_dataset: Dataset,
        val_dataset: Dataset,
        model_name: str,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        device: torch.device,
        early_stopper: Any,
        params: Dict[str, Any]
) -> Tuple[float, int, List[float],  List[float],  List[float],  List[float]]:
    """
    Runs one trial of optuna's hyperparameter optimization search for a BERT-based sequence classification model.

    Receives a hyperparameter combination and runs one trial with a fixed number of 10 epochs.
    If the model does not improve for a certain number of epochs, early stopping gets triggered and
    the epoch number with which the best F1-score got achieved gets returned.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        model_name (str): Name of the BERT model.
        label2id (Dict[str, int]): Dictionary to map label strings with corresponding ids.
        id2label (Dict[int, str]): Dictionary to map ids with corresponding label strings.
        device (torch.device): Device (cuda/mps/cuda) on which the model should get trained.
        early_stopper (Any): Early Stopping class which takes in current model performance and can trigger early stopping.
        params (Dict[str, Any]): Hyperparameters for this trial.
            - "lr" (float): learning rate
            - "weight_decay" (float): optimizer weight decay
            - "batch_size" (int): training batch size
        
    Returns:
        Tuple[float, int, List[float],  List[float],  List[float],  List[float]]:
            - Best validation F1-score.
            - The number of epochs. Can be less than 10 if early stopping got triggered.
            - List of training loss for each epoch.
            - List of validation loss for each epoch.
            - List of F1-scores for each epoch measured on training set.
            - List of F1-scores for each epoch measured on validation set.
    """
    # print the current trial number with parameters
    print(f"\nTrial with params: {params}")

    # get the number of possible labels
    num_labels: int = len(label2id)

    # extract the parameters with which to run this trial
    lr: float = params["lr"]
    weight_decay: float = params["weight_decay"]
    batch_size: int = params["batch_size"]

    # set epochs to 10 for all trials
    epochs: int = 10

    # instantiate empty lists to save the development of losses and F1 score
    train_losses: List[float] = []
    val_losses: List[float] = []
    f1_scores_train: List[float] = []
    f1_scores_val: List[float] = []

    # set up a new model instance and optimizer
    model: nn.Module = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    optimizer: optim.Optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # create new data loaders for this trial
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # set model to training mode and run the training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss: float = 0.0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_masks: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["label"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss: Tensor = outputs.loss
                total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss: float = total_loss / len(train_dataloader)
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
        train_f1: float = metrics["macro avg"]["f1-score"]
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
        val_f1: float = metrics["macro avg"]["f1-score"]
        f1_scores_val.append(val_f1)

        # feed validation loss into early stopping object to save and stop model training if necessary
        early_stopper(val_f1, model, epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    # delete all objects and empty the cache to free up space
    __clear_cache_models(model=model, optimizer=optimizer, device=device)

    return early_stopper.best_f1, early_stopper.best_epoch, train_losses, val_losses, f1_scores_train, f1_scores_val


def tune_bert_nli_stance_optuna(
        train_data_original: List[Dict[str, str]],
        train_data_neg_aug: List[Dict[str, str]],
        val_dataset: List[Dict[str, str]],
        model_name: str,
        device: torch.device,
        early_stopper: Any,
        params: Dict[str, Any]
) -> Tuple[float, int, List[float],  List[float]]:
    """
    Runs one trial of optuna's hyperparameter optimization search for a BERT-based sequence classification model
    used for a Natural Language Inference task.

    Receives a hyperparameter combination and runs one trial with a fixed number of 8 epochs.
    If the model does not improve for a certain number of epochs, early stopping gets triggered and
    the epoch number with which the best F1-score got achieved gets returned.

    Args:
        train_data_original (List[Dict[str, str]]): Original training data.
        train_data_neg_aug (List[Dict[str, str]]): Augmented training data of the negative class only.
        val_dataset (List[Dict[str, str]]): Original validation data.
        model_name (str): Name of the BERT model.
        device (torch.device): Device (cuda/mps/cuda) on which the model should get trained.
        early_stopper (Any): Early Stopping class which takes in current model performance and can trigger early stopping.
        params (Dict[str, Any]): Hyperparameters for this trial.
            - "lr" (float): learning rate
            - "weight_decay" (float): optimizer weight decay
            - "batch_size" (int): training batch size
        
    Returns:
        Tuple[float, int, List[float],  List[float]]:
            - Best validation F1-score.
            - The number of epochs. Can be less than 8 if early stopping got triggered.
            - List of F1-scores for each epoch measured on training set.
            - List of F1-scores for each epoch measured on validation set.
    """
    # print the current trial with the hyperparameters chose
    print(f"\nTrial with params: {params}")

    # extract the parameters with which to run this trial
    lr: float = params["lr"]
    weight_decay: float = params["weight_decay"]
    batch_size: int = params["batch_size"]

    # set epochs to 8 for all trials
    epochs: int = 8

    # instantiate empty lists to save the development of losses and F1 score
    f1_scores_train: List[float] = []
    f1_scores_val: List[float] = []

    # set up a new model instance and optimizer
    model: nn.Module = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    optimizer: optim.Optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    label_to_id: Dict[str, int] = model.config.label2id

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create datasets with oversampled data
    train_dataset = StanceNLIDataset(train_data_original, tokenizer, max_len=128, label2id=label_to_id)
    train_dataset_neg_aug = StanceNLIDataset(train_data_neg_aug, tokenizer, max_len=128, label2id=label_to_id)
    train_dataset = ConcatDataset([train_dataset, train_dataset_neg_aug])
    
    # create new data loader for this trial
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set model to training mode and run the training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss: float = 0.0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_masks: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["label"].to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss: Tensor = outputs.loss
                total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss: float = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_loss:.4f}")

        # run model in inference mode to get loss and seqeval f1-score for the training dataset
        true_labels, pred_labels = evaluate_nli_stance(
            model=model, data=train_data_original, tokenizer=tokenizer, device=device
            )
        metrics = sklearn_classification_report(true_labels, pred_labels, output_dict=True)
        train_f1: float = metrics["macro avg"]["f1-score"]
        f1_scores_train.append(train_f1)

        # do the same to get loss and seqeval f1-score for the validation set
        true_labels, pred_labels = evaluate_nli_stance(
            model=model, data=val_dataset, tokenizer=tokenizer, device=device
            )
        metrics = sklearn_classification_report(true_labels, pred_labels, output_dict=True)
        val_f1: float = metrics["macro avg"]["f1-score"]
        f1_scores_val.append(val_f1)

        # feed model performance into early stopping object to save and stop model training if necessary
        early_stopper(val_f1, model, epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
    # delete all objects and empty the cache to free up space
    __clear_cache_models(model=model, optimizer=optimizer, device=device)

    return early_stopper.best_f1, early_stopper.best_epoch, f1_scores_train, f1_scores_val