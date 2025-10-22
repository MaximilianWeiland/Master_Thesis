# import libraries
import json

# load the data
with open("../01_data/annotations.json", "r") as f:
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
            "label": r["value"]["labels"][0]
        }
        for r in results if r["type"] == "labels"
    ]
    optimal_data.append({"sentence": text,
                         "annotations": spans})
    
with open("../01_data/annotations.json", "w") as f:
    json.dump(optimal_data, f)