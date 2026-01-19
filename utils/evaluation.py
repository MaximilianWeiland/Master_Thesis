import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import random
from typing import List, Tuple, Dict, Any, Union, Callable, Set

# ----------------------------------------------------------------------
# Run Test Set Through The Models For Evaluation
# ----------------------------------------------------------------------

NERTagSeq = List[List[str]]
SpanSeq = List[List[List[int]]]
def run_testset_ner(
        model: nn.Module,
        test_dataloader: DataLoader,
        id2tag: Dict[int, str],
        device: torch.device,
        for_metric: str
) -> Tuple[Union[NERTagSeq, SpanSeq], Union[NERTagSeq, SpanSeq], float] :
    """
    Evaluates a NER model on a test set and returns predictions and loss.

    Args:
        model (nn.Module): Trained NER model.
        test_dataloader (DataLoader): DataLoader for the evaluation set.
        id2tag (Dict[int, str]): Mapping from tag IDs to tag strings.
        device (torch.device): Device used for inference.
        for_metric (str): Evaluation mode. One of:
            - "seqeval"
            - "sentence_level"
            - "cross_span"
            - "mention_detection"

    Returns:
        Tuple[data_true, data_pred, avg_loss]:
            - If for_metric in {"seqeval", "sentence_level"}:
                data_true, data_pred are List[List[str]]
            - If for_metric in {"cross_span", "mention_detection"}:
                data_true, data_pred are List[List[List[int]]]
            - avg_loss (float): Mean loss over all batches
    """
    # set model to evaluation mode
    model.eval()

    # instantiate empty lists and set loss and batch tracking variables to 0
    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []
    all_true_spans: List[List[List[int]]] = []
    all_pred_spans: List[List[List[int]]] = []
    total_loss: float = 0.0
    num_batches: int = 0

    with torch.no_grad():
        # loop over all batches of the dataloader and extract content
        for batch in test_dataloader:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_masks: Tensor = batch["attention_mask"].to(device)
            tag_ids: Tensor = batch["tag_ids"].to(device)
            batch_word_ids = batch["word_ids"]
            # run through the model, compute loss and get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=tag_ids)
            logits: Tensor = outputs.logits
            loss: Tensor = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            predictions: Tensor = torch.argmax(logits, dim=2)

            # convert predictions and ground truth to format dependent on evaluation metric

            # if seqeval just work with the true and predicted tags
            if for_metric == "seqeval":
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()

                    true_tags: List[str] = []
                    pred_tags: List[str] = []
                    for t, p in zip(true_seq, pred_seq):
                        if t != -100:
                            true_tags.append(id2tag[t])
                            pred_tags.append(id2tag[p])

                    all_true_tags.append(true_tags)
                    all_pred_tags.append(pred_tags)

            # if cross_span or mention_detection get the span indices
            elif (for_metric == "cross_span") or (for_metric == "mention_detection"):
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    word_ids = batch_word_ids[i]

                    word_level_tags, _ = labels_to_wordlevel_tags(true_seq, id2tag, word_ids)
                    all_true_spans.append(extract_spans(word_level_tags))

                    word_level_tags, _ = labels_to_wordlevel_tags(pred_seq, id2tag, word_ids)
                    all_pred_spans.append(extract_spans(word_level_tags))

            # if sentence_level convert to word level and append word level tags
            elif for_metric == "sentence_level":
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    word_ids = batch_word_ids[i]

                    true_word_tags, _ = labels_to_wordlevel_tags(true_seq, id2tag, word_ids)
                    pred_word_tags, _ = labels_to_wordlevel_tags(pred_seq, id2tag, word_ids)

                    all_true_tags.append(true_word_tags)
                    all_pred_tags.append(pred_word_tags)

    # calculate average batch loss
    avg_loss: float = total_loss / num_batches

    if for_metric == "seqeval":
        return all_true_tags, all_pred_tags, avg_loss
    elif for_metric == "cross_span":
        return all_true_spans, all_pred_spans, avg_loss
    elif for_metric == "mention_detection":
        return all_true_spans, all_pred_spans, avg_loss
    elif for_metric == "sentence_level":
        return all_true_tags, all_pred_tags, avg_loss
    
def run_testset_stance(
        model: nn.Module,
        test_dataloader: DataLoader,
        device: torch.device
) -> Tuple[List[int], List[int], float]:
    """
    Evaluates a sequence classification model for stance classification and returns true labels, predicted labels and loss.

    Args:
        model (nn.Module): BERT sequence classification model.
        test_dataloader (DataLoader): Test dataset to evaluate on.
        device (torch.device): Device (cuda/mps/cpu) the model should run on.
    
    Returns:
        Tuple[List[int], List[int], float]:
            - A list of true labels.
            - A list of predicted labels.
            - Average loss of the test model evaluated on test dataset.
    """
    # set model to evaluation mode
    model.eval()

    # create empty lists to store labels in as well as variables to track loss and number of batches
    true_labels: List[int] = []
    pred_labels: List[int] = []
    total_loss: float = 0.0
    num_batches: int = 0

    with torch.no_grad():
        # loop over all batches of the dataloader
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # run inputs through the model, compute loss and get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=1)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            # extend label lists with ground truth and the predictions
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
    
    # calculate average batch loss
    avg_loss = total_loss / num_batches
    
    return true_labels, pred_labels, avg_loss

def evaluate_nli_stance(
        model: nn.Module,
        data: List[Dict[str, str]],
        tokenizer: Any,
        device: torch.device
) -> Tuple[List[str], List[str]]:
    """
    Runs test data through a BERT sequence classification model in Natural Language Inference (NLI) style.
    Extracts the ground truth as well as the model predictions and returns these values.

    Args:
        model (nn.Module): BERT sequence classification model trained on NLI.
        data (List[Dict[str, str]]): Test dataset containing:
            - "sentence" (str): Original sentence.
            - "group" (str): Social group mentioned in the sentence.
            - "stance" (str): Stance that is taken towards the social group.
        tokenizer (Any): BERT-compatible tokenizer.
        device (torch.device): Device (cuda/mps/cpu) the model should run on.

    Returns:
        Tuple[List[str], List[str]]:
            - Ground truth stance labels.
            - Predicted stance labels.
    """
    # set model to evaluation mode
    model.eval()

    # create empty lists to store ground truth and predicted labels in
    all_labels: List[str] = []
    all_preds: List[str] = []

    # loop over all test data items
    for item in data:
        # extract original sentence, social group target and stance towards it
        sentence: str = item["sentence"]
        target: str = item["group"]
        gold_stance: str = item["stance"]
        
        # construct hypotheses to evaluate
        hypotheses: Dict[str, str] = {
            "pos": f"The text is positive towards {target}.",
            "neg": f"The text is negative towards {target}.",
            "neutral": f"The text is neutral, or contains no stance, towards {target}."
            }

        # tokenize all 3 hypotheses as a batch and move all inputs to device
        inputs = tokenizer(
            [sentence]*3,
            list(hypotheses.values()),
            return_tensors="pt",
            padding=True,
            truncation=True
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # run inputs through the model and get probabilities by applying softmax activation
        with torch.no_grad():
            outputs = model(**inputs)
            probs: Tensor = torch.softmax(outputs.logits, dim=-1)
            entail_probs: List[float] = probs[:, 0].tolist() # 0 is the entailment index

        # choose hypothesis with highest entailment probability and append to lists
        predicted_stance: str = list(hypotheses.keys())[entail_probs.index(max(entail_probs))]
        all_labels.append(gold_stance)
        all_preds.append(predicted_stance)

    return all_labels, all_preds

# ----------------------------------------------------------------------
# Cross-Validation Loops
# ----------------------------------------------------------------------

def cv_ner(
        model_names: List[str],
        training_data: List[Dict[str, Any]],
        dataset_class: Any,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        num_folds: int,
        optimal_configurations: Dict[str, Dict[str, Any]],
        custom_collate_fn: Callable,
        device: torch.device,
        seed: int = 3
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Performs cross-validation for multiple NER models using precomputed optimal hyperparameters.
    These hyperparameter have been found via tuning on a held-out validation set.
    Augmentations are added to 25% of the training fold. Evaluation is done on several metrics and always
    contains uncertainty estimates. All results are saved and returned in a dictionary.

    Args:
        model_names (List[str]): List of pretrained model names.
        training_data (List[Dict[str, Any]]): Full training dataset.
        dataset_class (Any): Dataset class to wrap training/validation data.
        label2id (Dict[str, int]): Label to id mapping for NER tags.
        id2label (Dict[int, str]): Id to label mapping for NER tags.
        num_folds (int): Number of cross-validation folds.
        optimal_configurations (Dict[str, Dict[str, Any]]): Optimal hyperparameters for each model.
        custom_collate_fn (Callable): Collate function to use in DataLoader.
        device (torch.device): Device (cuda/mps/cpu) to train/evaluate models on.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Average metrics across folds per model.
    """
    # load training function within function to avoid cycle
    from utils.classification import train_bert

    # empty dictionary to store the final average metrics
    average_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    # loop over all individual models
    for model_name in model_names:

        print(f"\nCross-validation for model {model_name} starts")
        print("-"*100)

        # dictionary to save the individual fold evaluation metrics
        fold_metrics: Dict[str, List[Dict[str, float]]] = {
            "seqeval": [],
            "cross_span": [],
            "mention_detection": [],
            "sentence_level": []
            }
       
        # get the optimal hyperparameters for this model
        epochs: int = optimal_configurations[model_name]["best_epoch"]
        lr:float = optimal_configurations[model_name]["best_params"]["lr"]
        batch_size: int = optimal_configurations[model_name]["best_params"]["batch_size"]
        weight_decay: float = optimal_configurations[model_name]["best_params"]["weight_decay"]

        # define the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # create cross-validation object for 5 folds
        k: int = num_folds
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        # randomly split the training data into 5 folds and loop over them
        for fold, (train_idx, val_idx) in enumerate(kf.split(training_data)):

            print(f"\nFold number: {fold + 1}")

            # create the training and validation fold based on the provided indices
            train_fold_data = [training_data[i] for i in train_idx]
            val_fold_data = [training_data[i] for i in val_idx]

            # create tensor dataset and respective data loaders
            augmented_train_data_generative: List[Dict[str, Any]] = []
            num_augmentations_to_add: int = int(len(train_fold_data) * 0.25)
            candidates_for_augmentation = random.sample(
                train_fold_data, k=min(num_augmentations_to_add, len(train_fold_data))
                )
            for original_item in candidates_for_augmentation:
                if "augmentations" in original_item and len(original_item["augmentations"]) > 0:
                    aug_gen = original_item["augmentations"][-1]
                    augmented_train_data_generative.append({
                        "id": f"{original_item['id']}_aug_{aug_gen['method']}",
                        "sentence": aug_gen["sentence"],
                        "annotations": aug_gen["annotations"]
                    })
            train_fold_data_gen_augmentations = train_fold_data + augmented_train_data_generative

            train_dataset = dataset_class(train_fold_data_gen_augmentations, tokenizer, label2id, max_len=128)
            val_dataset = dataset_class(val_fold_data, tokenizer, label2id, max_len=128)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

            # create the model and optimizer
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id
                ).to(device)
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # train the model with the optimal configuration
            train_bert(train_dataloader, model, optimizer, epochs, device, which_task="ner")

            # create dictionary to save the inputs for each evaluation metric
            evaluation_inputs: Dict[str, Dict[str, Any]] = {
                "seqeval": {},
                "cross_span": {},
                "mention_detection": {},
                "sentence_level": {}
                }
                
            # loop over all metrics and get the ground truth as well as predicted labels for the validation set
            for metric in evaluation_inputs.keys():
                all_true, all_pred, _ = run_testset_ner(
                    model=model, test_dataloader=val_dataloader, id2tag=id2label, device=device, for_metric=metric
                    )
                evaluation_inputs[metric] = {
                    "all_true": all_true,
                    "all_pred":all_pred
                    }
                    
            # apply all evaluation functions and save in the dictionary
            metrics_seqeval = evaluate_seqeval(
                evaluation_inputs["seqeval"]["all_true"],
                evaluation_inputs["seqeval"]["all_pred"]
                )
            metrics_cross_span = cross_span_evaluation(
                evaluation_inputs["cross_span"]["all_true"],
                evaluation_inputs["cross_span"]["all_pred"]
                )
            metrics_mention_detection = mention_detection_evaluation(
                evaluation_inputs["mention_detection"]["all_true"],
                evaluation_inputs["mention_detection"]["all_pred"]
            )
            metrics_sentence_level = sentence_level_evaluation(
                evaluation_inputs["sentence_level"]["all_true"],
                evaluation_inputs["sentence_level"]["all_pred"]
                )

            # append all metrics to the dictionary
            fold_metrics["seqeval"].append(metrics_seqeval)
            fold_metrics["cross_span"].append(metrics_cross_span)
            fold_metrics["mention_detection"].append(metrics_mention_detection)
            fold_metrics["sentence_level"].append(metrics_sentence_level)

            # delete model instances and clear the memory
            del model, optimizer
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


        # helper function to calculate summary statistics
        def summarize(values: List[float]) -> Dict[str, float]:
            """
            Helper function to calculate mean, standard deviation and confidence intervals of evaluation values.

            Args:
                values (List[float]): List of evaluation result values.
            
            Returns:
                Dict[str, float]: Dictionary containing summary of the evaluation values.
                    - "mean" (float): Average of evaluation values.
                    - "sd" (float): Standard deviation of evaluation values.
                    - "lower" (float): Lower bound of confidence interval around the mean of evaluation values.
                    - "upper" (float): Upper bound of confidence interval around the mean of evaluation values.
            """
            mean: float = np.mean(values)
            sd: float = np.std(values)
            ci: float = 1.96 * (sd/np.sqrt(5))
            return {
                "mean": mean,
                "sd": sd,
                "lower": mean - ci,
                "upper": mean + ci
            }
        
        # compute summary for all metrics
        seqeval_metrics = {
            key: summarize([m[key] for m in fold_metrics["seqeval"]])
            for key in ["precision", "recall", "f1"]
        }
        cross_span_metrics = {
            key: summarize([m[key] for m in fold_metrics["cross_span"]])
            for key in ["precision", "recall", "f1"]
        }
        mention_detection_metrics = {
            key: summarize([m[key] for m in fold_metrics["mention_detection"]])
            for key in ["precision", "recall", "f1"]
        }
        sentence_level_metrics = {
            key: summarize([m[key] for m in fold_metrics["sentence_level"]])
            for key in ["precision", "recall", "f1"]
        }

        # store summary results in the overall dictionary
        average_metrics[model_name] = {
            "seqeval": seqeval_metrics,
            "cross_span": cross_span_metrics,
            "mention_detection": mention_detection_metrics,
            "sentence_level": sentence_level_metrics
        }
  
        print("-"*100)
    
    return average_metrics

def cv_stance(
        model_names: List[str],
        training_data: List[Dict[str, str]],
        dataset_class: Any,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        num_folds: int,
        optimal_configurations: Dict[str, Dict[str, Any]],
        device: torch.device,
        seed: int = 3
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Performs cross-validation for multiple BERT sequence classification models using precomputed optimal hyperparameters.
    These hyperparameter have been found via tuning on a held-out validation set.
    Augmentations are added for the minority (negative) class of each training fold.
    Evaluation is done on several metrics and always contains uncertainty estimates.
    All results are saved and returned in a dictionary.

    Args:
        model_names (List[str]): List of pretrained model names.
        training_data (List[Dict[str, Any]]): Full training dataset.
        dataset_class (Any): Dataset class to wrap training/validation data.
        label2id (Dict[str, int]): Label to id mapping for NER tags.
        id2label (Dict[int, str]): Id to label mapping for NER tags.
        num_folds (int): Number of cross-validation folds.
        optimal_configurations (Dict[str, Dict[str, Any]]): Optimal hyperparameters for each model.
        device (torch.device): Device (cuda/mps/cpu) to train/evaluate models on.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Average metrics across folds per model.
    """
    # load function within the function to avoid dependancy cycle
    from utils.classification import train_bert

    # empty dictionary to store the final average metrics
    average_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    # loop over all individual models
    for model_name in model_names:

        print(f"\nCross-validation for model {model_name} starts")
        print("-"*100)

        # dictionary to save the individual fold evaluation metrics
        fold_metrics: Dict[str, List[Dict[str, float]]] = {
            "negative": [],
            "neutral": [],
            "positive": [],
            "macro": []
            }
       
        # get the optimal hyperparameters for this model
        epochs: int = optimal_configurations[model_name]["best_epoch"]
        lr: float = optimal_configurations[model_name]["best_params"]["lr"]
        batch_size: int = optimal_configurations[model_name]["best_params"]["batch_size"]
        weight_decay: float = optimal_configurations[model_name]["best_params"]["weight_decay"]

        # define the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # create cross-validation object for 5 folds
        k: int = num_folds
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        all_labels: List[str] = [item[0]["stance"] for item in training_data]

        # randomly split the training data into 5 folds and loop over them
        for fold, (train_idx, val_idx) in enumerate(kf.split(training_data, all_labels)):

            print(f"\nFold number: {fold + 1}")

            # create the training and validation fold based on the provided indices
            train_fold_data = [training_data[i] for i in train_idx]
            val_fold_data = [training_data[i] for i in val_idx]

            # unpack validation data (augmentations are not needed)
            val_data = [task[0] for task in val_fold_data]
            val_dataset = dataset_class(val_data, tokenizer, max_len=128, label2id=label2id)

            # get the original training data
            train_data_original = [task[0] for task in train_fold_data]
            train_dataset = dataset_class(train_data_original, tokenizer, max_len=128, label2id=label2id)
            
            # oversample all classes to a certain proportion and creare datasets
            neg_ann = [r for r in train_fold_data if r[0]["stance"] == "neg"]
            neg_extra = resample(neg_ann, replace=False, n_samples=int(len(neg_ann)*0.5), random_state=0)
            train_data_neg_aug = [task[2] for task in neg_extra]
            train_dataset_neg_aug = dataset_class(train_data_neg_aug, tokenizer, max_len=128, label2id=label2id)
            train_dataset_balanced = ConcatDataset([train_dataset, train_dataset_neg_aug])

            # create the data loaders
            train_dataloader = DataLoader(train_dataset_balanced, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label2id),
                id2label = id2label,
                label2id=label2id
                ).to(device)
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # train the model with the optimal configuration
            train_bert(train_dataloader, model, optimizer, epochs, device, which_task="stance")
            
            true_labels, pred_labels, _ = run_testset_stance(model=model, test_dataloader=val_dataloader, device=device)
            metrics = sklearn_classification_report(
                [id2label[i] for i in true_labels],
                [id2label[i] for i in pred_labels],
                output_dict=True
                )
                
            # helper function to append to fold metrics
            def extract_fold_metrics(cr: Dict[str, Any], eval_class: str) -> Dict[str, float]:
                    """
                    Extracts precision, recall and f1-score from an sklearn classification report.

                    Args:
                        cr (Dict[str, Any]): Sklearn classification report.
                        eval_class (str): Class of the classification report to extract metrics from.

                    Returns:
                        Dict[str, float]: Precision, recall and f1-score from an sklearn classification report for the given class.
                    """
                    return {
                        "precision": cr[eval_class]["precision"],
                        "recall": cr[eval_class]["recall"],
                        "f1": cr[eval_class]["f1-score"]
                    }

            # update the fold metrics
            fold_metrics["negative"].append(extract_fold_metrics(metrics, "neg"))
            fold_metrics["neutral"].append(extract_fold_metrics(metrics, "neutral"))
            fold_metrics["positive"].append(extract_fold_metrics(metrics, "pos"))
            fold_metrics["macro"].append(extract_fold_metrics(metrics, "macro avg"))

            # delete model instances and empty the cache
            del model, optimizer
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # helper function to calculate summary statistics
        def summarize(values: List[float]) -> Dict[str, float]:
            """
            Helper function to calculate mean, standard deviation and confidence intervals of evaluation values.

            Args:
                values (List[float]): List of evaluation result values.
                
            Returns:
                Dict[str, float]: Dictionary containing summary of the evaluation values.
                    - "mean" (float): Average of evaluation values.
                    - "sd" (float): Standard deviation of evaluation values.
                    - "lower" (float): Lower bound of confidence interval around the mean of evaluation values.
                    - "upper" (float): Upper bound of confidence interval around the mean of evaluation values.
            """
            mean: float = np.mean(values)
            sd: float = np.std(values)
            ci: float = 1.96 * (sd/np.sqrt(5))
            return {
                "mean": mean,
                "sd": sd,
                "lower": mean - ci,
                "upper": mean + ci
            }

        # calculate summary statistics for all classes
        negative_metrics = {
            key: summarize([m[key] for m in fold_metrics["negative"]])
            for key in ["precision", "recall", "f1"]
        }
        neutral_metrics = {
            key: summarize([m[key] for m in fold_metrics["neutral"]])
            for key in ["precision", "recall", "f1"]
        }
        positive_metrics = {
            key: summarize([m[key] for m in fold_metrics["positive"]])
            for key in ["precision", "recall", "f1"]
        }
        macro_metrics = {
            key: summarize([m[key] for m in fold_metrics["macro"]])
            for key in ["precision", "recall", "f1"]
        }

        # add results to the overall evaluation dictionary
        average_metrics[model_name] = {
            "negative": negative_metrics,
            "neutral": neutral_metrics,
            "positive": positive_metrics,
            "macro": macro_metrics
        }
            
        print("-"*100)
    
    return average_metrics

def cv_stance_nli(
        model_name: str,
        training_data: List[Dict[str, str]],
        dataset_class: Any,
        num_folds: int,
        optimal_configurations: Dict[str, Dict[str, Any]],
        device: torch.device,
        seed: int = 3
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Performs cross-validation for a BERT sequence classification models trained on NLI using precomputed optimal hyperparameters.
    These hyperparameter have been found via tuning on a held-out validation set.
    Augmentations are added for the minority (negative) class of each training fold.
    Evaluation is done on several metrics and always contains uncertainty estimates.
    All results are saved and returned in a dictionary.

    Args:
        model_names (List[str]): List of pretrained model names.
        training_data (List[Dict[str, Any]]): Full training dataset.
        dataset_class (Any): Dataset class to wrap training/validation data.
        num_folds (int): Number of cross-validation folds.
        optimal_configurations (Dict[str, Dict[str, Any]]): Optimal hyperparameters for each model.
        device (torch.device): Device (cuda/mps/cpu) to train/evaluate models on.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Average metrics across folds per model.
    """
    # load function within the function to avoid dependency cycles
    from utils.classification import train_bert

    # empty dictionary to store the final average metrics
    average_metrics:  Dict[str, Dict[str, Dict[str, float]]] = {}

    print(f"\nCross-validation for model {model_name} starts")
    print("-"*100)

    # dictionary to save the individual fold evaluation metrics
    fold_metrics: Dict[str, List[Dict[str, float]]] = {
        "negative": [],
        "neutral": [],
        "positive": [],
        "macro": []
        }
       
    # get the optimal hyperparameters for this model
    epochs: int = optimal_configurations[model_name]["best_epoch"]
    lr: float = optimal_configurations[model_name]["best_params"]["lr"]
    batch_size: int = optimal_configurations[model_name]["best_params"]["batch_size"]
    weight_decay: float = optimal_configurations[model_name]["best_params"]["weight_decay"]

    # define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create cross-validation object for 5 folds
    k: int = num_folds
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    all_labels: List[str] = [item[0]["stance"] for item in training_data]

    # randomly split the training data into 5 folds and loop over them
    for fold, (train_idx, val_idx) in enumerate(kf.split(training_data, all_labels)):

        print(f"\nFold number: {fold + 1}")

        # create the model
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        label_to_id: Dict[str, int] = model.config.label2id

        # create the training and validation fold based on the provided indices
        train_fold_data = [training_data[i] for i in train_idx]
        val_fold_data = [training_data[i] for i in val_idx]

        # unpack validation data (augmentations are not needed)
        val_data = [task[0] for task in val_fold_data]

        # get the original training data
        train_data_original = [task[0] for task in train_fold_data]
        train_dataset = dataset_class(train_data_original, tokenizer, max_len=128, label2id=label_to_id)
            
        # oversample minority classes to a certain proportion and create datasets
        neg_ann = [r for r in train_fold_data if r[0]["stance"] == "neg"]
        neg_extra = resample(neg_ann, replace=False, n_samples=int(len(neg_ann)*0.5), random_state=0)
        train_data_neg_aug = [task[2] for task in neg_extra]
        train_dataset_neg_aug = dataset_class(train_data_neg_aug, tokenizer, max_len=128, label2id=label_to_id)
        train_dataset_balanced = ConcatDataset([train_dataset, train_dataset_neg_aug])

        # create the data loader
        train_dataloader = DataLoader(train_dataset_balanced, batch_size=batch_size, shuffle=True)

        # train the model with the optimal configuration
        train_bert(train_dataloader, model, optimizer, epochs, device, which_task="stance")

        # run evaluation
        true_labels, pred_labels = evaluate_nli_stance(
            model=model, data=val_data, tokenizer=tokenizer, device=device
            )
        metrics = sklearn_classification_report(true_labels, pred_labels, output_dict=True)
        
        # helper function to append to fold metrics
        def extract_fold_metrics(cr: Dict[str, Any], eval_class: str) -> Dict[str, float]:
            """
            Extracts precision, recall and f1-score from an sklearn classification report.

            Args:
                cr (Dict[str, Any]): Sklearn classification report.
                eval_class (str): Class of the classification report to extract metrics from.

            Returns:
                Dict[str, float]: Precision, recall and f1-score from an sklearn classification report for the given class.
            """
            return {
                "precision": cr[eval_class]["precision"],
                "recall": cr[eval_class]["recall"],
                "f1": cr[eval_class]["f1-score"]
                }

        # update the fold metrics
        fold_metrics["negative"].append(extract_fold_metrics(metrics, "neg"))
        fold_metrics["neutral"].append(extract_fold_metrics(metrics, "neutral"))
        fold_metrics["positive"].append(extract_fold_metrics(metrics, "pos"))
        fold_metrics["macro"].append(extract_fold_metrics(metrics, "macro avg"))

        # delete model instances and empty the cache
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()

    # helper function to calculate summary statistics
    def summarize(values: List[float]) -> Dict[str, float]:
        """
        Helper function to calculate mean, standard deviation and confidence intervals of evaluation values.

        Args:
            values (List[float]): List of evaluation result values.
            
        Returns:
            Dict[str, float]: Dictionary containing summary of the evaluation values.
                - "mean" (float): Average of evaluation values.
                - "sd" (float): Standard deviation of evaluation values.
                - "lower" (float): Lower bound of confidence interval around the mean of evaluation values.
                - "upper" (float): Upper bound of confidence interval around the mean of evaluation values.
        """
        mean: float = np.mean(values)
        sd: float = np.std(values)
        ci: float = 1.96 * (sd/np.sqrt(5))
        return {
            "mean": mean,
            "sd": sd,
            "lower": mean - ci,
            "upper": mean + ci
        }

    # compute summary statistics for all classes
    negative_metrics = {
        key: summarize([m[key] for m in fold_metrics["negative"]])
        for key in ["precision", "recall", "f1"]
    }
    neutral_metrics = {
        key: summarize([m[key] for m in fold_metrics["neutral"]])
        for key in ["precision", "recall", "f1"]
    }
    positive_metrics = {
        key: summarize([m[key] for m in fold_metrics["positive"]])
        for key in ["precision", "recall", "f1"]
    }
    macro_metrics = {
        key: summarize([m[key] for m in fold_metrics["macro"]])
        for key in ["precision", "recall", "f1"]
    }

    # store results in the overall dictionary
    average_metrics[model_name] = {
        "negative": negative_metrics,
        "neutral": neutral_metrics,
        "positive": positive_metrics,
        "macro": macro_metrics
    }
    
    return average_metrics

# ----------------------------------------------------------------------
# Strict Seqeval Metric
# ----------------------------------------------------------------------

def evaluate_seqeval(
        all_true_tags: List[List[str]],
        all_pred_tags: List[List[str]]
) -> Dict[str, float]:
    """
    Wrapper function to call seqeval classification report and output directly precision, recall and f1-score.

    Args:
        all_true_tags (List[List[str]]): All ground truth word-level BIO tags.
        all_pred_tags (List[List[str]]): All predicted word-level BIO tags.

    Returns:
        Dict[str, float]: Precision, recall and F1-score for the social group class.
    """
    # run the seqeval classification report
    classification_report = seqeval_classification_report(all_true_tags, all_pred_tags, output_dict=True)
    # extract metrics
    precision: float = classification_report["sg"]["precision"]
    recall: float = classification_report["sg"]["recall"]
    f1_score: float = classification_report["sg"]["f1-score"]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
        }

# ----------------------------------------------------------------------
# Custom Cross-Span Metric (Soft Seqeval)
# ----------------------------------------------------------------------

def labels_to_wordlevel_tags(
        predicted_tag_ids: List[int],
        id_to_tag: Dict[int, str],
        word_ids: List[int]
) -> Tuple[List[str], List[int]]:
    """
    Converts the predicted tag ids on the token level to string tags on the word level.
    To convert from token to words, the tokenizers' returned word ids are used.

    Args:
        predicted_tag_ids (List[int]): Predicted BIO tag ids for one sentence.
        id_to_tag (Dict[int, str]): Dictionary mapping from tag ids to the actual tags as strings.
        word_ids (List[int]): List of word ids for all tokens present in the sentence.
    
    Returns:
        Tuple[List[str], List[int]]:
            - Final word level tags.
            - All unique word ids present in the sentence.
    """
    # intialize dictionary to store all tags assigned to individual words
    word_tags: Dict[int, List[str]] = {}

    # loop through word ids
    for idx, wid in enumerate(word_ids):
        # skip special tokens and padding
        if wid is None:
            continue
        # get the bio-tag assigned to the token
        tag_id: int = predicted_tag_ids[idx]
        tag: str = id_to_tag[tag_id][0]
        # add the wid and the assigned token to the dictionary
        if wid not in word_tags:
            word_tags[wid] = []
        word_tags[wid].append(tag)

    # get all unique word ids
    unique_ids = sorted(word_tags.keys())
    
    # initialize list of final labels per word
    final_tags: List[str] = []

    # loop over all word ids
    for wid in sorted(word_tags.keys()):
        # get all unique labels assigned to this word
        unique_labels = set(word_tags[wid])
        if "O" in unique_labels:
            final_tags.append("O")
        elif "B" in unique_labels:
            final_tags.append("B")
        elif "I" in unique_labels:
            final_tags.append("I")
        else:
            final_tags.append("O")

    return final_tags, unique_ids

def extract_spans(word_tags: List[str]) -> List[List[int]]:
    """
    Computes entity spans marked by word indices in a sentence based on BIO tagging scheme.

    Args:
        word_tags (List[str]): List of BIO tags on the word level for one sentence.

    Returns:
        List[List[int]]: List of all entity spans in the sentence which are marked by word indices building the span.
    """
    
    # empty lists to collect all spans and the current span
    spans: List[List[int]] = []
    current_span: List[int] = []

    # loop through all word ids
    for idx, tag in enumerate(word_tags):
        # if the word's tag is B this is the beginning of a new span
        if tag in ("B-sg","B"):
            # if there exists already a span append it
            if current_span:
                spans.append(current_span)
            # start the new span
            current_span = [idx]
        # if tag is I this is the continuation of a span so append
        elif tag in ("I-sg"):
            current_span.append(idx)
        # if tag is O append the last span and start an empty one
        else:
            if current_span:
                spans.append(current_span)
                current_span = []
    # if there is an existing span at the end, append it  
    if current_span:
        spans.append(current_span)

    return spans

def cross_span_evaluation(
        all_true_spans: List[List[List[int]]],
        all_pred_spans: List[List[List[int]]]
) -> Dict[str, float]:
    """
    Evaluates predicted spans against true spans at the word level using a custom 'soft' implementation of the seqeval metric.

    For each predicted span, it finds the true span with the largest overlap.
    Precision, recall, and F1-score are computed per span, and unmatched spans are counted as zero.
    The final metrics are averaged across all spans in all sentences.

    Args:
        all_true_spans (List[List[List[int]]]): List of sentences, each containing a list of true spans.
        all_pred_spans (List[List[List[int]]]): List of sentences, each containing a list of predicted spans.

    Returns:
        Dict[str, float]: Dictionary containing the averaged precision, recall, and F1-score across all spans.
    """

    # empty list to store all mention-level metrics
    span_metrics: List[Dict[str, float]] = []
    
    # loop through all sentences
    for sentence_true, sentence_preds in zip(all_true_spans, all_pred_spans):
        
        # get all unique word ids for each span as a set
        true_sets: List[Set[int]] = [set(gt) for gt in sentence_true]
        pred_sets: List[Set[int]] = [set(p) for p in sentence_preds]
        # empty set that stores all visited true word ids
        matched_true_idx: Set[int] = set()

        # loop through all predicted spans
        for p_set in pred_sets:
            # variables that store with which span was the largest overlap
            best_overlap: int = 0
            best_idx = None
            # loop through true spans
            for i, t_set in enumerate(true_sets):
                # check overlap and store if it is a new best
                overlap: int = len(p_set & t_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            
            # if there was a match, calculate metrics for this predicted span
            if best_overlap > 0:
                t_set = true_sets[best_idx]
                precision: float = best_overlap / len(p_set)
                recall: float = best_overlap / len(t_set)
                f1: float = (2*precision*recall)/(precision+recall)
                # mark the true span as visited
                matched_true_idx.add(best_idx)
            
            # if no match, assign 0 to all metrics
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
            span_metrics.append({"precision": precision,
                                 "recall": recall,
                                 "f1": f1})

        # loop through all true spans 
        for i, t_set in enumerate(true_sets):
            # if not already visited, assign 0 for all metrics
            if i not in matched_true_idx:
                span_metrics.append({"precision": 0.0,
                                     "recall": 0.0,
                                     "f1": 0.0})
    
    # take cross-span averages and return
    avg_precision: float = sum(m["precision"] for m in span_metrics) / len(span_metrics)
    avg_recall: float = sum(m["recall"] for m in span_metrics) / len(span_metrics)
    avg_f1: float = sum(m["f1"] for m in span_metrics) / len(span_metrics)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }

# ----------------------------------------------------------------------
# Metric Measuring if Mention Got Generally Captured
# ----------------------------------------------------------------------

def mention_detection_evaluation(
        all_true_spans: List[List[List[int]]],
        all_pred_spans: List[List[List[int]]]
) -> Dict[str, float]:
    """
    Calculates precision, recall and F1-score for predictions of social group mentions on the sentence level.
    Counts model prediction as a TP if at least one word prediction overlaps with the ground truth.
    Thus, measures if the model finds the social group mention in general.

    Args:
        all_true_spans (List[List[List[int]]]): List of sentences, each containing a list of true spans.
        all_pred_spans (List[List[List[int]]]): List of sentences, each containing a list of predicted spans.

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1-score for all spans.

    """
    # store true positives, false positives and false negatives
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    # loop through all sentences
    for sentence_true, sentence_preds in zip(all_true_spans, all_pred_spans):
        
        # get all unique word ids for each span as a set
        true_sets: List[Set[int]] = [set(gt) for gt in sentence_true]
        pred_sets: List[Set[int]] = [set(p) for p in sentence_preds]
        # empty set that stores all visited true word ids
        matched_true_idx: Set[int] = set()

        # loop through all predicted spans
        for p_set in pred_sets:
            # variables that store with which span was the largest overlap
            best_overlap: int = 0
            best_idx = None
            # loop through true spans
            for i, t_set in enumerate(true_sets):
                # check overlap and store if it is a new best
                overlap = len(p_set & t_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            
            # if there was a match, calculate metrics for this predicted span
            if best_overlap > 0 and best_idx not in matched_true_idx:
                tp += 1
                # mark the true span as visited
                matched_true_idx.add(best_idx)
            
            # if no match, assign 0 to all metrics
            else:
                fp += 1

        # loop through all true spans 
        for i, t_set in enumerate(true_sets):
            # if not already visited, assign 0 for all metrics
            if i not in matched_true_idx:
               fn += 1
    
    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.00
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.00
    f1: float = (2*precision*recall) / (precision + recall) if (precision + recall) > 0 else 0.00

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ----------------------------------------------------------------------
# Sentence-Level Metric
# ----------------------------------------------------------------------

def sentence_level_evaluation(
        all_true_tags: List[List[str]],
        all_pred_tags: List[List[str]]
) -> Dict[str, float]:
    """
    Evaluation of social group word predictions on the sentence level. Counts a sentence as true positive
    if at least one of the actual positive words is correctly predicted as positive.

    Args:
        all_true_tags (List[List[str]]): All ground truth word-level BIO tags.
        all_pred_tags (List[List[str]]): All predicted word-level BIO tags.

    Returns:
        Dict[str, float]: Precision, recall and F1-score for the social group class.
    """
    # set true positives, false positives and false negatives to 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    # loop over all sentences
    for gt_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        # detect if ground truth and prediction do have any entity in them
        has_true = any(t.startswith(("B", "I")) for t in gt_tags)
        has_pred = any(t.startswith(("B", "I")) for t in pred_tags)
        # compute if at least one word-level prediction is correct
        has_correct = any(
            (gt.startswith(("B", "I")) and pred.startswith(("B", "I")))
            for gt, pred in zip(gt_tags, pred_tags)
        )

        # increment counts
        if has_correct:
            tp += 1
        elif has_pred and not has_true:
            fp += 1
        elif has_true and not has_pred:
            fn += 1
        elif has_true and has_pred and not has_correct:
            fn += 1
        else:
            pass

    # calculate precision, recall and f1-score based on the counts
    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.00
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.00
    f1: float = (2*precision*recall) / (precision + recall) if (precision + recall) > 0 else 0.00

    return {"precision": precision, "recall": recall, "f1": f1}