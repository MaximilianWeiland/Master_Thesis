import re
from transformers import RobertaTokenizerFast

# ----------------------------------------------------------------------
# Dictionary Baseline
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# BERT-Based Models
# ----------------------------------------------------------------------

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# function that creates BIO-tags for text
def bert_tokenization_labelling(text, entities, tag_to_id):

    # tokenize and get offsets
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True)
    # initialize label list with as many "O" labels as there are tokens
    tags = ["O"] * len(encoding.offset_mapping)
    
    # loop through all spans, get start and end position as well as the label
    for ent in entities:
        start, end = ent["start"], ent["end"]
        ent_tag = ent["label"]
        
        # loop through all tokens in the sentence
        for idx, (token_start, token_end) in enumerate(encoding.offset_mapping):

            # if token is at the start of the annotation, this is the B token
            if token_start == start:
                tags[idx] = f"B-{ent_tag}"
            # if token starts after the start and before the end (end is character after the last) this is an I token
            if token_start > start and token_end <= end:
                tags[idx] = f"I-{ent_tag}"

    # convert the labels to ids and return
    tag_ids = [tag_to_id.get(tag, tag_to_id["O"]) for tag in tags]

    return encoding["input_ids"], encoding["attention_mask"], tag_ids