import pandas as pd
import time
import json
import ast
import sys
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from transformers import AutoTokenizer

# set directory to the project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.preprocessing import create_category_regex, match_dictionary
from utils.clustering import get_mask_embedding, ModelMask

# choose to only use one GPU and set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = Path("/dataHDD1/max_weiland")

# load the data as a dataframe
df = pd.read_csv(
    data_root / "data/speech_datasets/researchperiod_sentencelevel_socialgroups_stances.csv",
    parse_dates=["date", "birth_date"],
    dtype={
        "days_until_election": "int64",
        "agenda": "string",
        "text": "string",
        "sentence": "string",
        "speech_id": "int64",
        "speechnumber": "int64",
        "speaker": "string",
        "party": "category",
        "chair": "boolean",
        "age": "int64",
        "gender": "category",
        "vulnerability": "float64",
        "backbencher": "boolean",
        "constituency_name": "string",
        "ons_id": "string",
        "under_30": "float64",
        "over_65": "float64",
    },
    converters={
        "sg_spans": ast.literal_eval,
        "stances": ast.literal_eval,
    }
)

# load the model with weights, as well as the tokenizer

# load model from checkpoint
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = ModelMask(tokenizer).to(device)
model.load_state_dict(torch.load(f=data_root / "final_models/clustering_model.pt", map_location=device))

# reduce df to all speeches given during parliamentary question debates
question_df = df[df["agenda"].str.contains("question", case=False, na=False)].copy()

# get all detected social group mentions as a df, keep the row id to be able to later match back
question_df_long = question_df.loc[:, ["sg_spans"]].explode("sg_spans").dropna().reset_index(names="row_id")

# find dictionary matches and assign the group category
groups_dict_original = pd.read_csv(data_root / "data/dictionary/groups_dictionary.csv")
category_regex, group_lookup = create_category_regex(groups_dict_original)
question_df_long[["in_dictionary", "category"]] = question_df_long["sg_spans"].apply(lambda x: pd.Series(match_dictionary(category_regex, group_lookup, x)))

# pass all mentions through the model
all_embeddings = get_mask_embedding(model, question_df_long["sg_spans"].tolist(), tokenizer, device)

# convert embeddings to numpy array
X = all_embeddings.detach().cpu().numpy()

# set number of clusters to consider
num_potential_clusters = 100
ks = list(range(2, num_potential_clusters+1))

# empty lists to store the results
silhouette_scores = []
nmi_scores = []
ari_scores = []

# create mask for terms appearing in dictionary and get their group categories 
dict_mask = question_df_long["in_dictionary"].values
dict_categories = question_df_long.loc[dict_mask, "category"].values

# set starting time
start_time = time.time()

# loop over k
for k in ks:

    # create model object
    kmeans = KMeans(
        n_clusters=k,
        random_state=7,
        n_init=10,
        max_iter=300
    )

    # fit the model and predict cluster ids
    pred_labels = kmeans.fit_predict(X)

    # compute all metrics
    sil_score = silhouette_score(X, pred_labels)
    nmi = normalized_mutual_info_score(
        dict_categories,
        pred_labels[dict_mask]
    )
    ari = adjusted_rand_score(
        dict_categories,
        pred_labels[dict_mask]
    )

    # append all metrics to the lists
    silhouette_scores.append(sil_score)
    nmi_scores.append(nmi)
    ari_scores.append(ari)

    # print status
    print(f"{k}/{num_potential_clusters} processed")
    current_time = time.time()
    print(f"{current_time - start_time} seconds processed")

# store results in a dictionary and export as a json
results_dict = {
    "silhouette_scores": silhouette_scores,
    "nmi_scores": nmi_scores,
    "ari_scores": ari_scores
}
with open(data_root / "clustering_results/cluster_metrics.json", "w") as f:
    json.dump(results_dict, f)