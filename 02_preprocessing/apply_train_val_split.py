# load necessary libraries
import json
from sklearn.model_selection import train_test_split

# load the data
with open("01_data/annotations/annotations_augmentations.json", "r") as f:
    data = json.load(f)

# split the data into a training and a validation set for ner
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=7)

# export both the training and the validation set to respective data folder
with open("01_data/training_validation_sets/ner/training_set.json", "w") as f:
    json.dump(train_data, f)

with open("01_data/training_validation_sets/ner/validation_set.json", "w") as f:
    json.dump(val_data, f)


# subset to only data with annotations
data_with_annotations = []
for task in data:
    if task["annotations"]:
        data_with_annotations.append(task)

# extract augmentations separately
data_synonym_aug = [
    {"sentence": task["augmentations"][0]["sentence"],
     "annotations": task["augmentations"][0]["annotations"]} for task in data_with_annotations]
data_paraphrase_aug = [
    {"sentence": task["augmentations"][-1]["sentence"],
     "annotations": task["augmentations"][-1]["annotations"]} for task in data_with_annotations]

# flatten the data
def flatten_data(data):
    data_flat = []
    for item in data:
        sentence = item["sentence"]
        for ann in item["annotations"]:
            record = {
                "sentence": sentence,
                "group": ann["text"],
                "stance": ann["tag"].lower()[3:]
            }
            data_flat.append(record)
    return data_flat

data_flattened = flatten_data(data_with_annotations)
data_synonym_flattened = flatten_data(data_synonym_aug)
data_paraphrase_flattened = flatten_data(data_paraphrase_aug)

# bind sentences from all methods together
data_bound = [(data_flattened[idx], data_synonym_flattened[idx], data_paraphrase_flattened[idx]) for idx in range(len(data_flattened))]

# get the label per sample
labels = [data_bound[idx][0]["stance"] for idx in range(len(data_bound))]

# split the dataset
train_bound, val_bound = train_test_split(
    data_bound,
    test_size=0.2,
    random_state=7,
    stratify=labels
)

# export both training and validation set to respective folder
with open("01_data/training_validation_sets/stance/training_set.json", "w") as f:
    json.dump(train_bound, f)

with open("01_data/training_validation_sets/stance/validation_set.json", "w") as f:
    json.dump(val_bound, f)


