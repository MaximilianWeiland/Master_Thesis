# load necessary libraries
import json
from sklearn.model_selection import train_test_split

# load the data
with open("01_data/annotations/annotations_augmentations.json", "r") as f:
    data = json.load(f)

# split the data into a training and a validation set
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

# export both the training and the validation set to respective data folder
with open("01_data/training_validation_set/training_set.json", "w") as f:
    json.dump(train_data, f)

with open("01_data/training_validation_set/validation_set.json", "w") as f:
    json.dump(val_data, f)

