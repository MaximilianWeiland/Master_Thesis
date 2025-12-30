# import necessary libraries
import pandas as pd
import gdown
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import re
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.preprocessing import create_regex_pattern, clean_text, split_sentences, proportional_stratified_sample

# specify if dataset should be downloaded locally or from Google Drive
download_location = "local" # "drive"


######################################### Data Loading ######################################### 

# load the data based on the download location
if download_location == "drive":
    file_id = "1DUgfdEhSo425FA2LJbpgFKH8Jr2jAHV_"
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    output = "../01_data/parlspeech_dataset.csv"
    gdown.download(url, output, quiet=False)
    parlspeech_df = pd.read_csv("../01_data/raw_datasets/parlspeech_dataset.csv")
elif download_location == "local":
    parlspeech_df = pd.read_csv("../01_data/raw_datasets/parlspeech_dataset.csv")

# load the dictionary from local folder
group_dictionary_df = pd.read_csv("../01_data/classification/dictionary/groups_dictionary.csv")


##################################  Data Manipulation and Subsetting  ################################## 

# convert date column to datetime and extract month and year
parlspeech_df["date"] = pd.to_datetime(parlspeech_df["date"])
parlspeech_df["month"] = parlspeech_df["date"].dt.month
parlspeech_df["year"] = parlspeech_df["date"].dt.year

# reduce the df to only speeches within the relevant electoral cycles and check number of characters
start_date = pd.to_datetime("2010-05-06")
end_date = pd.to_datetime("2019-12-12")
parlspeech_df_subset = parlspeech_df[(parlspeech_df["date"] >= start_date) & (parlspeech_df["date"] <= end_date) &
                                     (parlspeech_df["speaker"] != "CHAIR") & (parlspeech_df["text"].str.len() > 40)].copy()

# reduce the dataset to contain only speeches within parliamentary questions
all_agendas = parlspeech_df["agenda"].unique()
parliamentary_questions_agendas = [agenda for agenda in all_agendas if "questions" in str(agenda).lower()]
parliamentary_questions_df = parlspeech_df_subset[parlspeech_df_subset["agenda"].isin(parliamentary_questions_agendas)].copy()


################################## Text Cleaning and Sentence Splitting ##################################

punkt_param = PunktParameters()
abbreviations = ['hon', 'mr', 'mrs', 'dr', 'ms', 'sir', 'prof']  # lowercase
punkt_param.abbrev_types = set(abbreviations)
tokenizer = PunktSentenceTokenizer(punkt_param)

# apply cleaning and sentence splitting
parliamentary_questions_df['clean_text'] = parliamentary_questions_df['text'].apply(clean_text)
parliamentary_questions_df['sentences'] = parliamentary_questions_df['clean_text'].apply(lambda x: split_sentences(x, tokenizer))

# get all individual sentences
sentences_df = parliamentary_questions_df.explode('sentences').reset_index(drop=True)
sentences_df = sentences_df.rename(columns={'sentences': 'sentence'})


################################## Apply Dictionary and Stratified Sampling ##################################

# apply function to the groups dictionary
combined_regex = create_regex_pattern(group_dictionary_df)

# apply the compound regex pattern to all sentences
matched_df = sentences_df[sentences_df["sentence"].str.contains(combined_regex, flags=re.IGNORECASE, regex=True)]

# get all sentences that did not match the regex pattern
unmatched_df = sentences_df[~sentences_df["sentence"].isin(matched_df["sentence"])]

# apply sampling to both the matched and unmatched dataframes
sample_matched = proportional_stratified_sample(matched_df, strata_cols=['party', 'year'], total_samples=5000)
sample_unmatched = proportional_stratified_sample(unmatched_df, strata_cols=['party', 'year'], total_samples=5000)

# concatenate the two and mix up randomly
sample_annotations = pd.concat([sample_matched, sample_unmatched]).sample(frac=1)

# print the number of annotations (will vary slightly from 1000)
print(f"Total number of sampled sentences: {len(sample_annotations)}")

# export as a csv file
sample_annotations["sentence"].to_csv("../01_data/classification/sample_annotations.csv", index=False)

