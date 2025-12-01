# load all necessary libraries
library(tidyverse)
library(here)
library(quanteda)
library(rjson)
library(stringr)
library(caret)

# import training dataset
file_path <- here("01_data", "training_validation_sets", "stance", "training_set.json")
data <- fromJSON(file=file_path)

# import optimal configs
file_path <- here("04_classification_evaluation", "stance_classification",
                  "dictionary_baseline", "hyperparameter_tuning_results",
                  "ht_dictionary_stance.json")
optimal_configs <- fromJSON(file=file_path)


# turn the training data into a df
training_df <- lapply(data, function(task) {
  first <- task[[1]]
  data.frame(
    sentence  = first$sentence,
    span_text = first$group,
    sentiment = first$stance,
    stringsAsFactors = FALSE
  )
}) |>
  bind_rows() |>
  dplyr::select(sentence, span_text, sentiment)


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




# extract the optimal hyperparameters
window_size <- optimal_configs$window_size
pos_threshold <- optimal_configs$positive_threshold
neg_threshold <- optimal_configs$negative_threshold

nfolds <- 5 
n <- nrow(training_df)
set.seed(3)
permutation <- sample(1:n)
fold_metrics <- list()

for (j in 1:nfolds) {
  
  # define validation indices
  val_indices <- permutation[floor((j-1)*n/nfolds+1) : floor(j*n/nfolds)]
  validation_fold <- training_df[val_indices, , drop = FALSE]
  
  # extract all sentences, spans and ground truth
  true_labels <- validation_fold$sentiment
  all_sentences <- validation_fold$sentence
  all_spans <- validation_fold$span_text
  
  context_sentences <- compute_contexts(all_sentences, all_spans, window = window_size)
  pred_labels <- compute_sentiment_prediction(context_sentences, pos_threshold, neg_threshold)
  true_labels <- factor(true_labels)
  pred_labels <- factor(pred_labels, levels = levels(true_labels))
  
  cm <- caret::confusionMatrix(pred_labels, true_labels)
  precision <- cm$byClass[, "Precision"]
  recall <- cm$byClass[, "Recall"]
  f1_scores <- 2 * precision * recall / (precision + recall)
  
  # map F1 scores by clean class names
  class_names <- rownames(cm$byClass)
  class_names_clean <- sub("^Class: ", "", class_names)
  clean_precision <- setNames(precision, class_names_clean)
  clean_recall <- setNames(recall, class_names_clean)
  clean_f1_scores <- setNames(f1_scores, class_names_clean)
  
  # create a list with these results and append
  fold_result <- list(
    negative = list(
      precision = clean_precision["neg"],
      recall    = clean_recall["neg"],
      f1        = clean_f1_scores["neg"]
    ),
    neutral = list(
      precision = clean_precision["neutral"],
      recall    = clean_recall["neutral"],
      f1        = clean_f1_scores["neutral"]
    ),
    positive = list(
      precision = clean_precision["pos"],
      recall    = clean_recall["pos"],
      f1        = clean_f1_scores["pos"]
    ),
    macro = list(
      precision = mean(clean_precision),
      recall    = mean(clean_recall),
      f1        = mean(clean_f1_scores)
    )
  )
  fold_metrics[[j]] <- fold_result
  
}

# helper function to calculate summary results
summarize <- function(values) {
  mean = mean(values)
  sd = sd(values)
  ci = 1.96 * (sd/sqrt(5))
  summary_list = list(
    mean = mean,
    sd = sd,
    lower = mean - ci,
    upper = mean + ci
  )
}

# calculate summary statistics for each class
negative_metrics <- list(
  precision = summarize(sapply(fold_metrics, function(x) x$negative$precision)),
  recall    = summarize(sapply(fold_metrics, function(x) x$negative$recall)),
  f1        = summarize(sapply(fold_metrics, function(x) x$negative$f1))
)

neutral_metrics <- list(
  precision = summarize(sapply(fold_metrics, function(x) x$neutral$precision)),
  recall    = summarize(sapply(fold_metrics, function(x) x$neutral$recall)),
  f1        = summarize(sapply(fold_metrics, function(x) x$neutral$f1))
)

positive_metrics <- list(
  precision = summarize(sapply(fold_metrics, function(x) x$positive$precision)),
  recall    = summarize(sapply(fold_metrics, function(x) x$positive$recall)),
  f1        = summarize(sapply(fold_metrics, function(x) x$positive$f1))
)

macro_metrics <- list(
  precision = summarize(sapply(fold_metrics, function(x) x$macro$precision)),
  recall    = summarize(sapply(fold_metrics, function(x) x$macro$recall)),
  f1        = summarize(sapply(fold_metrics, function(x) x$macro$f1))
)

summary_results <- list(
  negative = negative_metrics,
  neutral  = neutral_metrics,
  positive = positive_metrics,
  macro    = macro_metrics
)


# export the results
json_string <- toJSON(summary_results)
output_path <- here("04_classification_evaluation", "stance_classification",
                    "cross_val_results", "evaluation_metrics_dictionary.json")
write(json_string, file = output_path)
