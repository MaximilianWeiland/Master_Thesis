# load all necessary libraries
library(tidyverse)
library(here)
library(quanteda)
library(rjson)
library(stringr)
library(caret)

# set the working directory to the project root
project_root <- here()

# import JSON data from file
file_path <- here("01_data", "annotations_reduced.json")
data <- fromJSON(file=file_path)

# keep only the sentences with annotations
data_with_annotations <- data[sapply(data, function(task) length(task$annotations) > 0)]

# turn the data into a df
df <- lapply(data_with_annotations, function(task) {
  anns <- do.call(rbind, lapply(task$annotations, function(ann) {
    data.frame(
      sentence = task$sentence,
      span_text = ann$text,
      sentiment = substr(ann$tag, 4, nchar(ann$tag)),
      stringsAsFactors = FALSE
    )
  }))
  return(anns)
})  |> 
  # bind all rows into one df
  bind_rows() |> 
  select(sentence, span_text, sentiment)


# function to dynamically compute context around the span
compute_contexts <- function(sentences, spans, window = 6) {
  
  # vectorize all sentences
  contexts <- vector("character", length(sentences))
  
  # loop over all sentences
  for (i in seq_along(sentences)) {
    
    # split the sentence and the span into individual tokens/words
    sentence_tokens <- str_extract_all(sentences[i], "\\w+|['’]\\w+|[^\\w\\s]")[[1]]
    span_tokens <- str_extract_all(spans[i], "\\w+|['’]\\w+|[^\\w\\s]")[[1]]
    
    # if no span tokens found take the whole sentence
    if (length(span_tokens) == 0) {
      contexts[i] <- paste(sentence_tokens, collapse = " ")
      
      # otherwise determine the context sentence
    } else {
      # find the indices of the first match
      match_idx <- which(
        sapply(
          1:(length(sentence_tokens) - length(span_tokens) + 1),
          function(j) all(sentence_tokens[j:(j+length(span_tokens)-1)] == span_tokens))
        )
      # get the start and end token indices of the context sentence
      if (length(match_idx) > 0) {
        start <- max(1, match_idx[1] - window)
        end <- min(length(sentence_tokens), match_idx[1] + length(span_tokens) - 1 + window)
        contexts[i] <- paste(sentence_tokens[start:end], collapse = " ")
      }
      # if the span was not found just return the full sentence
      else {
        contexts[i] <- paste(sentence_tokens, collapse = " ")
      }
    }
  }
  return(contexts)
}

# function to calculate sentiment prediction
compute_sentiment_prediction <- function(context, pos_thresh, neg_thresh) {
  
  # get the dictionary, tokenize and create dfm
  data_lex <- data_dictionary_LSD2015
  toks <- tokens(context)
  dfm_context <- dfm(toks)
  dfm_sent <- dfm_lookup(dfm_context, dictionary = data_dictionary_LSD2015)
  
  # create vectors of counts for positive and negative words
  pos_count <- rowSums(dfm_sent[, "positive", drop = FALSE])
  neg_count <- rowSums(dfm_sent[, "negative", drop = FALSE])
  
  # compute compound scores
  compound_score <- log((pos_count + 0.5) / (neg_count + 0.5))
  
  # get prediction based on custom thresholds
  pred <- ifelse(is.na(compound_score), "neutral",
                 ifelse(compound_score > pos_thresh, "pos",
                        ifelse(compound_score < neg_thresh, "neg", "neutral")))
  
}

# function to calculate the macro f1 score
compute_macro_f1 <- function(true_labels, pred_labels) {
  
  # ensure true and predicted labels are factor variables
  true_labels <- factor(true_labels)
  pred_labels <- factor(pred_labels, levels = levels(true_labels))
  
  # create the confusion matrix
  cm <- caret::confusionMatrix(pred_labels, true_labels)
  
  # compute per-class F1 score
  precision <- cm$byClass[, "Precision"]
  recall <- cm$byClass[, "Recall"]
  f1 <- 2 * precision * recall / (precision + recall)
  
  # take the average of these scores for the macro score
  macro_f1 <- mean(f1, na.rm = TRUE)
  macro_f1
}


# determine which combinations to test
window_sizes <- c(6, 8, 10, 12, 14, 16, 18, 20)
pos_threshs <- c(0.4, 0.5, 0.6)
neg_threshs <- c(-0.4, -0.5, -0.6)

# store best score and best combination
best_macro_f1 <- 0
best_combination <- list()

# get instances that will not change in the loop
true_labels <- df$sentiment
all_sentences <- df$sentence
all_spans <- df$span_text


# loop over all window sizes and thresholds
for (window_size in window_sizes) {
  for (idx in seq_along(pos_threshs)) {
    
    # extract the current thresholds
    pos_thresh <- pos_threshs[idx]
    neg_thresh <- neg_threshs[idx]
    
    # get context, predicted labels and compute macro f1
    context_sentences <- compute_contexts(all_sentences, all_spans, window = window_size)
    pred_labels <- compute_sentiment_prediction(context_sentences, pos_thresh, neg_thresh)
    macro_f1 <- compute_macro_f1(true_labels, pred_labels)
    
    # if score is an improvement, save it
    if (macro_f1 > best_macro_f1) {
      best_combination <- list(window_size = window_size,
                               pos_threshold = pos_thresh,
                               neg_threshold = neg_thresh)
      best_macro_f1 <- macro_f1
    }
  }
}
paste("Best combination:", 
      "window_size =", best_combination$window_size,
      ", pos_threshold =", best_combination$pos_threshold,
      ", neg_threshold =", best_combination$neg_threshold,
      "-> Macro F1 =", round(best_macro_f1, 4))

# apply this combination again to get all classification results
test_metrics <- list()
context_sentences <- compute_contexts(all_sentences, all_spans, window = 10)
pred_labels <- compute_sentiment_prediction(context_sentences, 0.4, -0.4)
true_labels <- factor(true_labels)
pred_labels <- factor(pred_labels, levels = levels(true_labels))
cm <- caret::confusionMatrix(pred_labels, true_labels)
precision <- cm$byClass[, "Precision"]
recall <- cm$byClass[, "Recall"]
f1_scores <- 2 * precision * recall / (precision + recall)

# map F1 scores by clean class names
class_names <- rownames(cm$byClass)
class_names_clean <- sub("^Class: ", "", class_names)
f1_named <- setNames(f1_scores, class_names_clean)

# store everything in dictionary and export as a json
test_metrics <- list(
  dictionary_baseline = list(
    negative = f1_named["neg"],
    neutral  = f1_named["neutral"],
    positive = f1_named["pos"]
  )
)
json_string <- toJSON(test_metrics)
output_path = "04_classification_evaluation/sentiment_classification/evaluation_metrics_dictionary.json"
write(json_string, file = output_path)

