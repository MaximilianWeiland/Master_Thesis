import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet', aug_p = 0.5)

def augmentation_non_entity(task):

    # extract text and annotations
    text = task["sentence"]
    entities = task["annotations"]

    # augment the whole text and return if no labels present
    if not entities:
        new_sentence = aug.augment(text)[0]
        new_task = {
            "sentence": new_sentence,
            "annotations": []
        }
        return new_task
    
    # variables to store the new sentence and keep track of new indices
    new_sentence = ""
    span_indices = []
    prev_end = 0

    # loop through annotations
    for ent in entities:

        # get start, end, text and label of the span
        start, end = ent["start"], ent["end"]
        ent_text = ent["text"]
        ent_label = ent["label"]

        # extract prefix and count number of leading and trailing whitespaces
        prefix = text[prev_end:start]
        leading_wspaces = len(prefix) - len(prefix.lstrip(" "))
        trailing_wspaces = len(prefix) - len(prefix.rstrip(" "))

        # augment the core prefix and add the correct number of whitespaces
        core_prefix = prefix.strip(" ")
        if core_prefix:
            core_prefix = aug.augment(core_prefix)[0]
        augmented_prefix = ' ' * leading_wspaces + core_prefix + ' ' * trailing_wspaces

        # get the new span indices
        new_start = len(new_sentence) + len(augmented_prefix)
        new_end = new_start + len(ent_text)

        # append the annotations data to the list
        span_indices.append({
            "start": new_start,
            "end": new_end,
            "text": ent_text,
            "label": ent_label
        })

        # update the new sentence and index variable
        new_sentence += augmented_prefix + ent_text
        prev_end = end

    # add rest of the sentence after the last annotation 
    suffix = text[prev_end:]
    leading_spaces = len(suffix) - len(suffix.lstrip(' '))
    trailing_spaces = len(suffix) - len(suffix.rstrip(' '))
    core_suffix = suffix.strip(' ')
    if core_suffix:
        core_suffix = aug.augment(core_suffix)[0]
    augmented_suffix = ' ' * leading_spaces + core_suffix + ' ' * trailing_spaces
    new_sentence += augmented_suffix

    return {"sentence": new_sentence, "annotations": span_indices}

def augmentation_entity(task):

    # extract text and annotations
    text = task["sentence"]
    entities = task["annotations"]
    
    # augment the whole text and return if no labels present
    if not entities:
        new_sentence = aug.augment(text)[0]
        new_task = {
            "sentence": new_sentence,
            "annotations": []
        }
        return new_task
    
    # variables to store the new sentence and keep track of new indices
    new_sentence = ""
    span_indices = []
    prev_end = 0

    # loop through annotations
    for ent in entities:

        # get start, end, text and label of the span
        start, end = ent["start"], ent["end"]
        ent_text = ent["text"]
        ent_label = ent["label"]

        # extract prefix
        prefix = text[prev_end:start]

        # augment the annotation
        augmented_entity = aug.augment(ent_text)[0]

        # get the new span indices
        new_start = len(new_sentence) + len(prefix)
        new_end = new_start + len(augmented_entity)

        # append the annotations data to the list
        span_indices.append({
            "start": new_start,
            "end": new_end,
            "text": augmented_entity,
            "label": ent_label
        })

        # update the new sentence and index variable
        new_sentence += prefix + augmented_entity
        prev_end = end

    # add rest of the sentence after the last annotation 
    suffix = text[prev_end:]
    new_sentence += suffix

    return {"sentence": new_sentence, "annotations": span_indices}

def find_entity_span(task, augmented_sentence):
   
    # extract all annotations
    entities = task["annotations"]

    # if there are no annotations return empty list
    if not entities:
        new_task = {
            "sentence": augmented_sentence,
            "annotations": []
        }
        return new_task

    # instantiate empty list to store all entities
    new_entities = []
    search_start = 0

    # loop through entities and extract text and label
    for ent in entities:
        ent_text = ent["text"]
        label = ent["label"]

        # find the first match
        match = re.search(re.escape(ent_text), augmented_sentence[search_start:], re.IGNORECASE)

        # if there is a match find the start and end of the span and append to list
        if match:
            start = search_start + match.start()
            end = search_start + match.end()
            new_entities.append({
                "start": start,
                "end": end,
                "text": augmented_sentence[start:end],
                "label": label
            })
            search_start = end

        # only proceed if all entities could have been found
        else:
            return None
    
    # return in the correct format
    new_task = {
        "sentence": augmented_sentence,
        "annotations": new_entities
    }

    return new_task