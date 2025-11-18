# import libraries and dictionary
import re
import random
from nltk.corpus import stopwords
import nltk

# download all nltk stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def __find_most_similar(word, model, top_n=3):
    word_lower = word.lower()
    if word_lower not in model:
        return word
    similar_words = model.most_similar(word_lower, topn=top_n)
    return random.choice([w for w, _ in similar_words])

def __apply_augmentation(text, model, aug_p):

    # inner function to replace matched token
    def replace_token(match):
        token = match.group(0)
        # Only replace alphabetic non-stopwords
        if token.isalpha() and token.lower() not in stop_words and random.random() < aug_p:
            return __find_most_similar(token, model)
        return token

    # split into words, punctuation, punctuation followed by characters or whitespace, and replace in-place
    pattern = re.compile(r"\w+(?:'\w+)?|[^\w\s]|\s+")
    augmented_text = pattern.sub(replace_token, text)
    return augmented_text

def augmentation_non_entity(task, model, aug_p=0.3):
    
    # extract the text and all annotations
    text = task["sentence"]
    entities = sorted(task["annotations"], key=lambda x: x["start"])
    
    # if no entities present, replace synonyms in the full sentence
    if not entities:
        new_sentence = __apply_augmentation(text, model, aug_p)
        return {"method": "synonym_non_entity", "sentence": new_sentence, "annotations": []}
    
    # sort the entities
    entities = sorted(entities, key=lambda x: x["start"])

    # create a new task
    new_sentence = ""
    span_indices = []
    prev_end = 0

    # loop through annotations
    for ent in entities:

        # extract their metadata
        start, end = ent["start"], ent["end"]
        ent_text = ent["text"]
        ent_tag = ent["tag"]

        # get the prefix before the entity and apply augmentation
        prefix = text[prev_end:start]
        augmented_prefix = __apply_augmentation(prefix, model, aug_p)

        # get metadata for new annotation
        new_start = len(new_sentence) + len(augmented_prefix)
        new_end = new_start + len(ent_text)
        span_indices.append({"start": new_start, "end": new_end, "text": ent_text, "tag": ent_tag})

        # update the sentence with the entitiy text
        new_sentence += augmented_prefix + ent_text
        prev_end = end

    # add suffix after the last entity
    suffix = text[prev_end:]
    augmented_suffix = __apply_augmentation(suffix, model, aug_p)
    new_sentence += augmented_suffix

    return {"method": "synonym_non_entity", "sentence": new_sentence, "annotations": span_indices}

def augmentation_entity(task, model, aug_p_entity=1, aug_p_non_entitiy=.3):

    # extract the text and all annotations
    text = task["sentence"]
    entities = sorted(task["annotations"], key=lambda x: x["start"])

    # if no entities present, replace synonyms in the full sentence
    if not entities:
        new_sentence = __apply_augmentation(text, model, aug_p_non_entitiy)
        return {"method": "synonym_entity", "sentence": new_sentence, "annotations": []}
    
    # sort the entities first
    entities = sorted(entities, key=lambda x: x["start"])

    # create a new task
    new_sentence = ""
    span_indices = []
    prev_end = 0

    for ent in entities:

        # get start, end, text and label of the span
        start, end = ent["start"], ent["end"]
        ent_text = ent["text"]
        ent_tag = ent["tag"]

        # extract prefix
        prefix = text[prev_end:start]

        # augment the annotation
        augmented_entity = __apply_augmentation(ent_text, model, aug_p_entity)

        # get the new span indices
        new_start = len(new_sentence) + len(prefix)
        new_end = new_start + len(augmented_entity)
        span_indices.append({"start": new_start, "end": new_end, "text": augmented_entity, "tag": ent_tag})

        # update the sentence with the entitiy text
        new_sentence += prefix + augmented_entity
        prev_end = end

    suffix = text[prev_end:]
    new_sentence += suffix

    return {"method": "synonym_entity", "sentence": new_sentence, "annotations": span_indices}


def find_entity_span(task, augmented_sentence):
    entities = task["annotations"]

    if not entities:
        return {
            "method": "generative_paraphrase",
            "sentence": augmented_sentence,
            "annotations": []
        }

    # Sort and deduplicate entities by (text, tag)
    entities = sorted(entities, key=lambda x: x["start"])
    seen = set()
    unique_entities = []
    for ent in entities:
        pair = (ent["text"], ent["tag"])
        if pair not in seen:
            unique_entities.append(pair)
            seen.add(pair)

    # Find all matching spans
    spans = []
    for ent_text, ent_tag in unique_entities:
        for match in re.finditer(re.escape(ent_text), augmented_sentence, re.IGNORECASE):
            start, end = match.span()
            spans.append({
                "start": start,
                "end": end,
                "text": augmented_sentence[start:end],
                "tag": ent_tag
            })

    # Sort by span length descending, then start ascending
    spans.sort(key=lambda x: (-(x["end"] - x["start"]), x["start"]))

    # Filter out overlapping spans — keep the longest one
    filtered = []
    occupied = set()
    for span in spans:
        if not any(i in occupied for i in range(span["start"], span["end"])):
            filtered.append(span)
            occupied.update(range(span["start"], span["end"]))

    # Sort final entities by start position
    filtered.sort(key=lambda x: x["start"])

    return {
        "method": "generative_paraphrase",
        "sentence": augmented_sentence,
        "annotations": filtered
    }