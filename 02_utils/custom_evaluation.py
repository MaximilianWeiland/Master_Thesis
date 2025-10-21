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

def mention_level_evaluation(all_predicted_spans, all_true_spans):

    # empty list to store all mention-level metrics
    span_metrics = []
    
    # loop through all sentences
    for sentence_preds, sentence_true in zip(all_predicted_spans, all_true_spans):
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