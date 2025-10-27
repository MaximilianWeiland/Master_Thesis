# import libraries
import json
import os

# set input and output path explicitly
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "../01_data/test_annotations_complete.json")
output_path = os.path.join(script_dir, "../01_data/test_annotations_reduced.json")

# load the data
with open(input_path, "r") as f:
    data = json.load(f)

# convert the data into the optimal structure
optimal_data = []

for task in data:
    text = task["data"]["sentence"]
    results = task["annotations"][0]["result"]
    labels = [r["value"]["text"] for r in results]
    spans = [
        {
            "start": r["value"]["start"],
            "end": r["value"]["end"],
            "text": r["value"]["text"],
            "tag": r["value"]["labels"][0]
        }
        for r in results if r["type"] == "labels"
    ]
    optimal_data.append({"sentence": text,
                         "annotations": spans})
    
with open(output_path, "w") as f:
    json.dump(optimal_data, f)