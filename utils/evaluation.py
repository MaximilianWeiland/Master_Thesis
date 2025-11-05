import torch
from seqeval.metrics import classification_report as seqeval_classification_report


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

            elif for_metric == "cross_span":
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    word_ids = batch_word_ids[i]

                    word_level_tags, _ = labels_to_wordlevel_tags(true_seq, id2tag, word_ids)
                    all_true_spans.append(extract_spans(word_level_tags))

                    word_level_tags, _ = labels_to_wordlevel_tags(pred_seq, id2tag, word_ids)
                    all_pred_spans.append(extract_spans(word_level_tags))

            elif for_metric == "sentence_level":
                for i in range(len(tag_ids)):
                    true_seq = tag_ids[i].cpu().numpy()
                    pred_seq = predictions[i].cpu().numpy()
                    word_ids = batch_word_ids[i]

                    true_word_tags, _ = labels_to_wordlevel_tags(true_seq, id2tag, word_ids)
                    pred_word_tags, _ = labels_to_wordlevel_tags(pred_seq, id2tag, word_ids)

                    all_true_tags.append(true_word_tags)
                    all_pred_tags.append(pred_word_tags)

    avg_loss = total_loss / num_batches

    if for_metric == "seqeval":
        return all_true_tags, all_pred_tags, avg_loss
    elif for_metric == "cross_span":
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
    
    avg_loss = total_loss / num_batches
    
    return true_labels, pred_labels, avg_loss

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
    
def labels_to_wordlevel_tags(predicted_tag_ids, id_to_tag, word_ids):

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

def mention_level_evaluation(all_true_spans, all_pred_spans):

    # empty list to store all mention-level metrics
    span_metrics = []
    
    # loop through all sentences
    for sentence_preds, sentence_true in zip(all_true_spans, all_pred_spans):
        # get all unique word ids for each span as a set
        pred_sets = [set(p) for p in sentence_preds]
        true_sets = [set(gt) for gt in sentence_true]
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

# ----------------------------------------------------------------------
# Sentence-Level Metric
# ----------------------------------------------------------------------

# evaluate on the sentence level
def sentence_level_evaluation(all_true_tags, all_pred_tags):
    tp = fp = fn = 0

    for gt_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        has_true = any(tag.startswith(("B", "I")) for tag in gt_tags)
        has_pred = any(tag.startswith(("B", "I")) for tag in pred_tags)

        if has_true and has_pred:
            tp += 1
        elif has_pred and not has_true:
            fp += 1
        elif has_true and not has_pred:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.00
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.00
    f1 = (2*precision*recall) / (precision + recall) if (precision + recall) > 0 else 0.00

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }