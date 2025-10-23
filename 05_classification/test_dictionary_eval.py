# import libraries
import pandas as pd
import json
import re
import os
import sys
from sklearn.metrics import classification_report as sklearn_classification_report
from seqeval.metrics import classification_report as seqeval_classification_report

# import helper functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import create_regex_pattern
from utils.custom_evaluation import (extract_spans, mention_level_evaluation, sentence_level_evaluation)

# import the annotations
with open("01_data/annotations_reduced.json", "r") as f:
    data = json.load(f)

# import the dictionary
group_dictionary_df = pd.read_csv("01_data/groups_dictionary.csv")

# function to convert the annotations to bio tags on the word level
def tokenize_word_level(sentence):
    
    # get all words' start and end index
    word_spans = []
    char_idx = 0

    # split sentence via regex which ensures to also split at punctuation
    words = re.findall(r"\w+|'\w+|[^\w\s]", sentence)

    for word in words:
        start_idx = sentence.find(word, char_idx)
        end_idx = start_idx + len(word)
        word_spans.append((start_idx, end_idx))
        char_idx = end_idx
    
    return words, word_spans

def text_to_bio(task):

    # extract sentence and annotations
    sentence = task["sentence"]
    annotations = task["annotations"]

    words, word_spans = tokenize_word_level(sentence)

    # initialize all tags as being O
    bio_tags = ["O"] * len(words)
    
    # loop through the annotations
    for annotation in annotations:
        start_ann, end_ann = annotation["start"], annotation["end"]
        for idx, (start_idx, end_idx) in enumerate(word_spans):
            if start_idx == start_ann:
                bio_tags[idx] = "B-sg"
            elif start_idx > start_ann and end_idx <= end_ann:
                bio_tags[idx] = "I-sg"

    return bio_tags

# add bio tags to the dataset
for task in data:
    bio_tags = text_to_bio(task)
    task["bio_tags"] = bio_tags

# create the regex pattern
combined_regex = create_regex_pattern(group_dictionary_df)

def find_dictionary_matches(sentence, dictionary_regex):
    
    # first tokenize the sentence
    words, word_spans = tokenize_word_level(sentence)

    bio_tags = ["O"] * len(words)

    for match in re.finditer(dictionary_regex, sentence, re.IGNORECASE):
        start_match, end_match = match.span()

        for idx, (start_idx, end_idx) in enumerate(word_spans):
            if start_idx == start_match:
                bio_tags[idx] = "B-sg"
            elif start_idx > start_match and end_idx <= end_match:
                bio_tags[idx] = "I-sg"
    
    return bio_tags

# evaluate on the word level

# store all bio tags in a list
gt_bio_flat = [tag for sent in data for tag in sent["bio_tags"]]
gt_bio_nested = [sent["bio_tags"] for sent in data]

pred_bio_nested = []

for task in data:
    sentence = task["sentence"]
    pred_tags = find_dictionary_matches(sentence, combined_regex)
    pred_bio_nested.append(pred_tags)

# flatten the list
pred_bio_flat = [tag for sent in pred_bio_nested for tag in sent]

# evaluate at the word and at the entity level using seqeval
print("Evaluation at the word level")
print("-"*60)
print(sklearn_classification_report(gt_bio_flat, pred_bio_flat))
print("Evaluation at the entity level with seqeval")
print("-"*60)
print(seqeval_classification_report(gt_bio_nested, pred_bio_nested))