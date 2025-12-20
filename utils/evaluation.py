import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import random


# ----------------------------------------------------------------------
# Run Test set through the models
# ----------------------------------------------------------------------

def run_testset_ner(model, test_dataloader, id2tag, device, for_metric):
    
    model.eval()

    all_true_tags, all_pred_tags = [], []
    all_true_spans, all_pred_spans = [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            batch_word_ids = batch["word_ids"]

            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=tag_ids)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            predictions = torch.argmax(logits, dim=2)

            if for_metric == "seqeval":
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()

                    true_tags = []
                    pred_tags = []
                    for t, p in zip(true_seq, pred_seq):
                        if t != -100:
                            true_tags.append(id2tag[t])
                            pred_tags.append(id2tag[p])

                    all_true_tags.append(true_tags)
                    all_pred_tags.append(pred_tags)

            elif (for_metric == "cross_span") or (for_metric == "mention_detection"):
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    word_ids = batch_word_ids[i]

                    word_level_tags, _ = __labels_to_wordlevel_tags(true_seq, id2tag, word_ids)
                    all_true_spans.append(extract_spans(word_level_tags))

                    word_level_tags, _ = __labels_to_wordlevel_tags(pred_seq, id2tag, word_ids)
                    all_pred_spans.append(extract_spans(word_level_tags))

            elif for_metric == "sentence_level":
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    word_ids = batch_word_ids[i]

                    true_word_tags, _ = __labels_to_wordlevel_tags(true_seq, id2tag, word_ids)
                    pred_word_tags, _ = __labels_to_wordlevel_tags(pred_seq, id2tag, word_ids)

                    all_true_tags.append(true_word_tags)
                    all_pred_tags.append(pred_word_tags)

    avg_loss = total_loss / num_batches

    if for_metric == "seqeval":
        return all_true_tags, all_pred_tags, avg_loss
    elif for_metric == "cross_span":
        return all_true_spans, all_pred_spans, avg_loss
    elif for_metric == "mention_detection":
        return all_true_spans, all_pred_spans, avg_loss
    elif for_metric == "sentence_level":
        return all_true_tags, all_pred_tags, avg_loss
    
def run_testset_stance(model, test_dataloader, device):
    model.eval()
    true_labels, pred_labels = [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=1)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
    
    avg_loss = total_loss / num_batches
    
    return true_labels, pred_labels, avg_loss

def evaluate_nli_stance(model, data, tokenizer, device):
    model.eval()
    all_preds = []
    all_labels = []

    for item in data:
        sentence = item["sentence"]
        target = item["group"]
        gold_stance = item["stance"]
        
        hypotheses = {
            "pos": f"The text is positive towards {target}.",
            "neg": f"The text is negative towards {target}.",
            "neutral": f"The text is neutral, or contains no stance, towards {target}."
            }

        # Tokenize all 3 hypotheses as a batch
        inputs = tokenizer(
            [sentence]*3,
            list(hypotheses.values()),
            return_tensors="pt",
            padding=True,
            truncation=True
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entail_probs = probs[:, 0].tolist() # 0 is the entailment index

        # choose hypothesis with highest entailment probability
        predicted_stance = list(hypotheses.keys())[entail_probs.index(max(entail_probs))]

        all_labels.append(gold_stance)
        all_preds.append(predicted_stance)

    return all_labels, all_preds

# ----------------------------------------------------------------------
# Cross-Validation Loops
# ----------------------------------------------------------------------

def cv_ner(model_names, training_data, dataset_class, label2id, id2label, num_folds, optimal_configurations, custom_collate_fn, device, seed=3):
    
    from utils.classification import train_bert

    # empty dictionary to store the final average metrics
    average_metrics = {}

    # loop over all individual models
    for model_name in model_names:

        print(f"\nCross-validation for model {model_name} starts")
        print("-"*100)

        # dictionary to save the individual fold evaluation metrics
        fold_metrics = {
            "seqeval": [],
            "cross_span": [],
            "mention_detection": [],
            "sentence_level": []
            }
       
        # get the optimal hyperparameters for this model
        epochs = optimal_configurations[model_name]["best_epoch"]
        lr = optimal_configurations[model_name]["best_params"]["lr"]
        batch_size = optimal_configurations[model_name]["best_params"]["batch_size"]
        weight_decay = optimal_configurations[model_name]["best_params"]["weight_decay"]

        # define the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # create cross-validation object for 5 folds
        k = num_folds
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        # randomly split the training data into 5 folds and loop over them
        for fold, (train_idx, val_idx) in enumerate(kf.split(training_data)):

            print(f"\nFold number: {fold + 1}")

            # create the training and validation fold based on the provided indices
            train_fold_data = [training_data[i] for i in train_idx]
            val_fold_data = [training_data[i] for i in val_idx]

            # create tensor dataset and respective data loaders
            augmented_train_data_generative = []
            num_augmentations_to_add = int(len(train_fold_data) * 0.25)
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
            evaluation_inputs = {
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

            del model, optimizer
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()


        # helper function to calculate summary statistics
        def summarize(values):
            mean = np.mean(values)
            sd = np.std(values)
            ci = 1.96 * (sd/np.sqrt(5))
            return {
                "mean": mean,
                "sd": sd,
                "lower": mean - ci,
                "upper": mean + ci
            }
            
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

        average_metrics[model_name] = {
            "seqeval": seqeval_metrics,
            "cross_span": cross_span_metrics,
            "mention_detection": mention_detection_metrics,
            "sentence_level": sentence_level_metrics
        }
  
        print("-"*100)
    
    return average_metrics

def cv_stance(model_names, training_data, dataset_class, label2id, id2label, num_folds, optimal_configurations, device, seed=3):
    
    from utils.classification import train_bert

    # empty dictionary to store the final average metrics
    average_metrics = {}

    # loop over all individual models
    for model_name in model_names:

        print(f"\nCross-validation for model {model_name} starts")
        print("-"*100)

        # dictionary to save the individual fold evaluation metrics
        fold_metrics = {
            "negative": [],
            "neutral": [],
            "positive": [],
            "macro": []
            }
       
        # get the optimal hyperparameters for this model
        epochs = optimal_configurations[model_name]["best_epoch"]
        lr = optimal_configurations[model_name]["best_params"]["lr"]
        batch_size = optimal_configurations[model_name]["best_params"]["batch_size"]
        weight_decay = optimal_configurations[model_name]["best_params"]["weight_decay"]

        # define the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # create cross-validation object for 5 folds
        k = num_folds
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        all_labels = [item[0]["stance"] for item in training_data]

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
            def extract_fold_metrics(cr, eval_class):
                precision = cr[eval_class]["precision"]
                recall = cr[eval_class]["recall"]
                f1 = cr[eval_class]["f1-score"]

                return {"precision": precision,
                        "recall": recall,
                        "f1": f1}

            # update the fold metrics
            fold_metrics["negative"].append(extract_fold_metrics(metrics, "neg"))
            fold_metrics["neutral"].append(extract_fold_metrics(metrics, "neutral"))
            fold_metrics["positive"].append(extract_fold_metrics(metrics, "pos"))
            fold_metrics["macro"].append(extract_fold_metrics(metrics, "macro avg"))

            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


        # helper function to calculate summary statistics
        def summarize(values):
            mean = np.mean(values)
            sd = np.std(values)
            ci = 1.96 * (sd/np.sqrt(5))
            return {
                "mean": mean,
                "sd": sd,
                "lower": mean - ci,
                "upper": mean + ci
            }

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

        average_metrics[model_name] = {
            "negative": negative_metrics,
            "neutral": neutral_metrics,
            "positive": positive_metrics,
            "macro": macro_metrics
        }
            
        print("-"*100)
    
    return average_metrics


def cv_stance_nli(model_name, training_data, dataset_class, num_folds, optimal_configurations, device, seed=3):
    
    from utils.classification import train_bert

    # empty dictionary to store the final average metrics
    average_metrics = {}

    print(f"\nCross-validation for model {model_name} starts")
    print("-"*100)

    # dictionary to save the individual fold evaluation metrics
    fold_metrics = {
        "negative": [],
        "neutral": [],
        "positive": [],
        "macro": []
        }
       
    # get the optimal hyperparameters for this model
    epochs = optimal_configurations[model_name]["best_epoch"]
    lr = optimal_configurations[model_name]["best_params"]["lr"]
    batch_size = optimal_configurations[model_name]["best_params"]["batch_size"]
    weight_decay = optimal_configurations[model_name]["best_params"]["weight_decay"]

    # define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create cross-validation object for 5 folds
    k = num_folds
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    all_labels = [item[0]["stance"] for item in training_data]

    # randomly split the training data into 5 folds and loop over them
    for fold, (train_idx, val_idx) in enumerate(kf.split(training_data, all_labels)):

        print(f"\nFold number: {fold + 1}")

        # create the model
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        label_to_id = model.config.label2id

        # create the training and validation fold based on the provided indices
        train_fold_data = [training_data[i] for i in train_idx]
        val_fold_data = [training_data[i] for i in val_idx]

        # unpack validation data (augmentations are not needed)
        val_data = [task[0] for task in val_fold_data]

        # get the original training data
        train_data_original = [task[0] for task in train_fold_data]
        train_dataset = dataset_class(train_data_original, tokenizer, max_len=128, label2id=label_to_id)
            
        # oversample minority classes to a certain proportion and creare datasets
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
        def extract_fold_metrics(cr, eval_class):
            precision = cr[eval_class]["precision"]
            recall = cr[eval_class]["recall"]
            f1 = cr[eval_class]["f1-score"]

            return {"precision": precision,
                    "recall": recall,
                    "f1": f1}
        

        print(f"Current macro f1-score: {extract_fold_metrics(metrics, "macro avg")}")

        # update the fold metrics
        fold_metrics["negative"].append(extract_fold_metrics(metrics, "neg"))
        fold_metrics["neutral"].append(extract_fold_metrics(metrics, "neutral"))
        fold_metrics["positive"].append(extract_fold_metrics(metrics, "pos"))
        fold_metrics["macro"].append(extract_fold_metrics(metrics, "macro avg"))

        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


    # helper function to calculate summary statistics
    def summarize(values):
        mean = np.mean(values)
        sd = np.std(values)
        ci = 1.96 * (sd/np.sqrt(5))
        return {
            "mean": mean,
            "sd": sd,
            "lower": mean - ci,
            "upper": mean + ci
        }

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

def evaluate_seqeval(all_true_tags, all_pred_tags):
    classification_report = seqeval_classification_report(all_true_tags, all_pred_tags, output_dict=True)
    precision = classification_report["sg"]["precision"]
    recall = classification_report["sg"]["recall"]
    f1_score = classification_report["sg"]["f1-score"]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
        }

# ----------------------------------------------------------------------
# Custom Cross-Span Metric
# ----------------------------------------------------------------------
    
def __labels_to_wordlevel_tags(predicted_tag_ids, id_to_tag, word_ids):

    # intialize dictionary to store all tags assigned to individual words
    word_tags = {}

    # loop through word ids
    for idx, wid in enumerate(word_ids):
        # skip special tokens and padding
        if wid is None:
            continue
        # get the bio-tag assigned to the token
        tag_id = predicted_tag_ids[idx]
        tag = id_to_tag[tag_id][0]
        # add the wid and the assigned token to the dictionary
        if wid not in word_tags:
            word_tags[wid] = []
        word_tags[wid].append(tag)

    # get all unique word ids
    unique_ids = sorted(word_tags.keys())
    
    # initialize list of final labels per word
    final_tags = []

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

def extract_spans(word_tags):
    
    # empty lists to collect all spans and the current span
    spans = []
    current_span = []

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

def cross_span_evaluation(all_true_spans, all_pred_spans):

    # empty list to store all mention-level metrics
    span_metrics = []
    
    # loop through all sentences
    for sentence_true, sentence_preds in zip(all_true_spans, all_pred_spans):
        
        # get all unique word ids for each span as a set
        true_sets = [set(gt) for gt in sentence_true]
        pred_sets = [set(p) for p in sentence_preds]
        # empty set that stores all visited true word ids
        matched_true_idx = set()

        # loop through all predicted spans
        for p_set in pred_sets:
            # variables that store with which span was the largest overlap
            best_overlap = 0
            best_idx = None
            # loop through true spans
            for i, t_set in enumerate(true_sets):
                # check overlap and store if it is a new best
                overlap = len(p_set & t_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            
            # if there was a match, calculate metrics for this predicted span
            if best_overlap > 0:
                t_set = true_sets[best_idx]
                precision = best_overlap / len(p_set)
                recall = best_overlap / len(t_set)
                f1 = (2*precision*recall)/(precision+recall)
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
    avg_precision = sum(m["precision"] for m in span_metrics) / len(span_metrics)
    avg_recall = sum(m["recall"] for m in span_metrics) / len(span_metrics)
    avg_f1 = sum(m["f1"] for m in span_metrics) / len(span_metrics)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }

def mention_detection_evaluation(all_true_spans, all_pred_spans):

    # store true positives, false positives and false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # loop through all sentences
    for sentence_true, sentence_preds in zip(all_true_spans, all_pred_spans):
        
        # get all unique word ids for each span as a set
        true_sets = [set(gt) for gt in sentence_true]
        pred_sets = [set(p) for p in sentence_preds]
        # empty set that stores all visited true word ids
        matched_true_idx = set()

        # loop through all predicted spans
        for p_set in pred_sets:
            # variables that store with which span was the largest overlap
            best_overlap = 0
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
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.00
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.00
    f1 = (2*precision*recall) / (precision + recall) if (precision + recall) > 0 else 0.00

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ----------------------------------------------------------------------
# Sentence-Level Metric
# ----------------------------------------------------------------------

# evaluate on the sentence level
def sentence_level_evaluation(all_true_tags, all_pred_tags):
    tp = fp = fn = 0

    for gt_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        has_true = any(t.startswith(("B", "I")) for t in gt_tags)
        has_pred = any(t.startswith(("B", "I")) for t in pred_tags)

        # at least one token-level correct prediction for the group
        has_correct = any(
            (gt.startswith(("B", "I")) and pred.startswith(("B", "I")))
            for gt, pred in zip(gt_tags, pred_tags)
        )

        if has_correct:
            tp += 1
        elif has_pred and not has_true:
            fp += 1
        elif has_true and not has_pred:
            fn += 1
        elif has_true and has_pred and not has_correct:
            fn += 1
        else:
            pass  # no entity in gold or prediction → ignore sentence

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.00
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.00
    f1 = (2*precision*recall) / (precision + recall) if (precision + recall) > 0 else 0.00

    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_nli_stance(model, data, tokenizer, device):
    model.eval()
    all_preds = []
    all_labels = []

    for item in data:
        sentence = item["sentence"]
        target = item["group"]
        gold_stance = item["stance"]
        
        hypotheses = {
            "pos": f"The text is positive towards {target}.",
            "neg": f"The text is negative towards {target}.",
            "neutral": f"The text is neutral, or contains no stance, towards {target}."
            }

        # Tokenize all 3 hypotheses as a batch
        inputs = tokenizer(
            [sentence]*3,
            list(hypotheses.values()),
            return_tensors="pt",
            padding=True,
            truncation=True
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entail_probs = probs[:, 0].tolist() # 0 is the entailment index

        # choose hypothesis with highest entailment probability
        predicted_stance = list(hypotheses.keys())[entail_probs.index(max(entail_probs))]

        all_labels.append(gold_stance)
        all_preds.append(predicted_stance)

    return all_labels, all_preds