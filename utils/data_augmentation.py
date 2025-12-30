import re
import random
from nltk.corpus import stopwords
import nltk
from typing import Dict, List, Any, Set, Tuple
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def __find_most_similar(
        word: str,
        model: Any,
        top_n: int=5
) -> str:
    """
    Finds the most similar word based on cosine similarity of its static word embedding.

    Args:
        word (str): The word for which similar word should be found.
        model (Any): GloVe or Word2vec model providing static word embeddings.
        top_n (int): The number of most similar words to consider.

    Returns:
        str: Word randomly selected from the top_n most similar words.
    """
    # lowercase the word
    word_lower: str = word.lower()
    # if word is not in the model vocabulary just return the original word
    if word_lower not in model:
        return word
    # find the top-n most similar words and pick one randomly
    similar_words = model.most_similar(word_lower, topn=top_n)

    return random.choice([w for w, _ in similar_words])

def __apply_augmentation(
        text: str,
        model: Any,
        aug_p: float
) -> str:
    """
    Augments original sentence by randomly replacing words with words most similar to them.

    Args:
        text (str): The text to augment.
        model (Any): GloVe or Word2vec model providing static word embeddings.
        aug_p (float): Probability with which each word should be replaced.

    Returns:
        str: The augmented sentence.
    """
    # inner function to replace matched token
    def replace_token(match) -> str:
        """
        Replaces a single matched token with a similar word from the embedding model
        based on a probability threshold.

        Args:
            match: A regex match object representing a token in the text.

        Returns:
            str: Either the original token or a replacement token.
        """
        token = match.group(0)
        # only replace alphabetic non-stopwords
        if token.isalpha() and token.lower() not in stop_words and random.random() < aug_p:
            return __find_most_similar(token, model)
        return token

    # split into words, punctuation, punctuation followed by characters or whitespace, and replace in-place
    pattern = re.compile(r"\w+(?:'\w+)?|[^\w\s]|\s+")
    augmented_text = pattern.sub(replace_token, text)
    return augmented_text

def augmentation_non_entity(
        task: Dict[str, Any],
        model: Any,
        aug_p: float = 0.3
) -> Dict[str, Any]:
    """
    Performs synonym-based augmentation on the non-entity parts of a sentence while preserving entity spans.

    Args:
        task (Dict[str, Any]): Dictionary containing:
            - "sentence" (str): Original sentence.
            - "annotations" (List[Dict]): List of entity annotations, each containing:
                - "start" (int): Start index of the entity.
                - "end" (int): End index of the entity.
                - "text" (str): Text of the entity.
                - "tag" (str): Entity label/tag.
        model (Any): Word embedding model used for synonym replacement (e.g., GloVe or Word2Vec).
        aug_p (float, optional): Probability of replacing a word in non-entity parts. Defaults to 0.3.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "method" (str): Indicates augmentation method.
            - "sentence" (str): Augmented sentence.
            - "annotations" (List[Dict]): Updated entity spans.
    """    
    # extract the text and all annotations
    text: str = task["sentence"]
    entities: List[Dict[str, Any]] = sorted(task["annotations"], key=lambda x: x["start"])
    
    # if no entities present, replace synonyms in the full sentence
    if not entities:
        new_sentence: str = __apply_augmentation(text, model, aug_p)
        return {"method": "synonym_non_entity", "sentence": new_sentence, "annotations": []}
    
    # sort the entities
    entities: List[Dict[str, Any]] = sorted(entities, key=lambda x: x["start"])

    # create a new task
    new_sentence: str = ""
    span_indices: List[Dict[str, Any]] = []
    prev_end: int = 0

    # loop through annotations
    for ent in entities:

        # extract their metadata
        start: int = ent["start"]
        end: int = ent["end"]
        ent_text: str = ent["text"]
        ent_tag: str = ent["tag"]

        # get the prefix before the entity and apply augmentation
        prefix: str = text[prev_end:start]
        augmented_prefix: str = __apply_augmentation(prefix, model, aug_p)

        # get metadata for new annotation
        new_start: int = len(new_sentence) + len(augmented_prefix)
        new_end: int= new_start + len(ent_text)
        span_indices.append({"start": new_start, "end": new_end, "text": ent_text, "tag": ent_tag})

        # update the sentence with the entitiy text
        new_sentence += augmented_prefix + ent_text
        prev_end = end

    # add suffix after the last entity
    suffix: str = text[prev_end:]
    augmented_suffix: str = __apply_augmentation(suffix, model, aug_p)
    new_sentence += augmented_suffix

    return {"method": "synonym_non_entity", "sentence": new_sentence, "annotations": span_indices}

def augmentation_entity(
    task: Dict[str, Any],
    model: Any,
    aug_p_entity: float = 1.0,
) -> Dict[str, Any]:
    """
    Performs synonym-based augmentation on entity spans in a sentence while keeping non-entity text unchanged.

    Args:
        task (Dict[str, Any]): Dictionary containing:
            - "sentence" (str): Original sentence.
            - "annotations" (List[Dict]): List of entity annotations, each containing:
                - "start" (int): Start index of the entity.
                - "end" (int): End index of the entity.
                - "text" (str): Text of the entity.
                - "tag" (str): Entity label/tag.
        model (Any): Word embedding model used for synonym replacement.
        aug_p_entity (float, optional): Probability of replacing words within entity spans. Defaults to 1.0.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "method" (str): Indicates augmentation method.
            - "sentence" (str): Augmented sentence.
            - "annotations" (List[Dict]): Updated entity spans with augmented text.
    """
    # extract the text and all annotations
    text: str = task["sentence"]
    entities: List[Dict[str, Any]] = sorted(task["annotations"], key=lambda x: x["start"])

    # if no entities present, replace synonyms in the full sentence
    if not entities:
        new_sentence: str = __apply_augmentation(text, model)
        return {"method": "synonym_entity", "sentence": new_sentence, "annotations": []}
    
    # sort the entities first
    entities: List[Dict[str, Any]] = sorted(entities, key=lambda x: x["start"])

    # create a new task
    new_sentence: str = ""
    span_indices: List[Dict[str, Any]] = []
    prev_end: int = 0

    for ent in entities:

        # get start, end, text and label of the span
        start: int = ent["start"]
        end: int = ent["end"]
        ent_text: str = ent["text"]
        ent_tag: str = ent["tag"]

        # extract prefix
        prefix: str = text[prev_end:start]

        # augment the annotation
        augmented_entity: str = __apply_augmentation(ent_text, model, aug_p_entity)

        # get the new span indices
        new_start: int = len(new_sentence) + len(prefix)
        new_end: int = new_start + len(augmented_entity)
        span_indices.append({"start": new_start, "end": new_end, "text": augmented_entity, "tag": ent_tag})

        # update the sentence with the entitiy text
        new_sentence += prefix + augmented_entity
        prev_end = end

    suffix: str = text[prev_end:]
    new_sentence += suffix

    return {"method": "synonym_entity", "sentence": new_sentence, "annotations": span_indices}


def find_entity_span(
        task: Dict[str, Any],
        augmented_sentence: str
) -> Dict[str, Any]:
    """
    Finds and updates entity spans in an generatively augmented sentence based on the original entity annotations.

    Args:
        task (Dict[str, Any]): Original task containing:
            - "sentence" (str): Original sentence.
            - "annotations" (List[Dict]): List of entity annotations, each containing:
                - "start" (int): Start index of the entity.
                - "end" (int): End index of the entity.
                - "text" (str): Text of the entity.
                - "tag" (str): Entity label/tag.
        augmented_sentence (str): The augmented sentence in which to locate the entity spans.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "method" (str): Indicates augmentation method used.
            - "sentence" (str): The augmented sentence.
            - "annotations" (List[Dict]): Updated entity spans with start/end indices and tag information.
    """
    # extract all annotations for the sentence
    entities: List[Dict[str, Any]] = task["annotations"]

    # if not entities present return empty list for the annotations
    if not entities:
        return {
            "method": "generative_paraphrase",
            "sentence": augmented_sentence,
            "annotations": []
        }

    # sort and deduplicate entities
    entities: List[Dict[str, Any]] = sorted(entities, key=lambda x: x["start"])
    seen: Set[Tuple[str, str]] = set()
    unique_entities: List[Tuple[str, str]] = []
    for ent in entities:
        pair = (ent["text"], ent["tag"])
        if pair not in seen:
            unique_entities.append(pair)
            seen.add(pair)

    # find all matching spans
    spans: List[Dict[str, Any]] = []
    for ent_text, ent_tag in unique_entities:
        pattern = r"(?<!\w){}(?:s)?(?:'s|')?(?:-)?(?!\w)".format(re.escape(ent_text))
        for match in re.finditer(pattern, augmented_sentence, re.IGNORECASE):
            start, end = match.span()
            spans.append({
                "start": start,
                "end": end,
                "text": augmented_sentence[start:end],
                "tag": ent_tag
            })

    # sort by span length descending, then start ascending
    spans.sort(key=lambda x: (-(x["end"] - x["start"]), x["start"]))

    # filter out overlapping spans by keeping the longest one
    filtered: List[Dict[str, Any]] = []
    occupied: Set[int] = set()
    for span in spans:
        if not any(i in occupied for i in range(span["start"], span["end"])):
            filtered.append(span)
            occupied.update(range(span["start"], span["end"]))

    # sort final entities by start position
    filtered.sort(key=lambda x: x["start"])

    return {
        "method": "generative_paraphrase",
        "sentence": augmented_sentence,
        "annotations": filtered
    }