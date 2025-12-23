# import necessary libraries
import pandas as pd
import gdown
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import re

# specify if dataset should be downloaded locally or from Google Drive
download_location = "local" # "drive"


######################################### Function Definitions #########################################

# define all functions needed for preprocessing

def clean_text(text):
    # return empty string if the text column is not a string
    if not isinstance(text, str):
        return ""

    # replace misencoded punctuation
    replacements = {
        "Â£": "£",
        "â€œ": "“",
        "â€": "”",
        "â€˜": "‘",
        "â€™": "’",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "â€": '"',
        "Ã©": "é",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # normalize whitespaces
    text = re.sub(r"\s+", " ", text)

    # remove leading and trailing whitespaces
    return text.strip()


def split_sentences(text, tokenizer):
    if not isinstance(text, str) or not text.strip():
        return []
    return tokenizer.tokenize(text)


def create_regex_pattern(dictionary_df):
    # get all patterns in the dictionary to look for
    patterns = [word for col in dictionary_df.columns for word in dictionary_df[col] if pd.notna(word)]

    # initialize empty list to store the regex patterns
    regex_patterns = []

    # loop through all patterns
    for pat in patterns:

        # split pattern into words by spaces
        tokens = pat.split()
        regex_parts = []

        # loop through all tokens of the pattern
        for token in tokens:

            # match any word if asterisk is a separate token
            if token == '*':
                regex_parts.append(r'\w+')
            
            # match any prefix to the word
            elif token.startswith('*') and len(token) > 1:
                word = token[1:]
                regex_parts.append(rf'\w*{re.escape(word)}')
            
            # match any suffix to the word
            elif token.endswith('*') and len(token) > 1:
                word = token[:-1]
                regex_parts.append(rf'{re.escape(word)}\w*')

            # if no asterisk, take the word as is
            else:
                regex_parts.append(re.escape(token))

        # join tokens with \s+ to match spaces and append to the list of patterns
        regex_pattern = r'\b' + r'\s+'.join(regex_parts) + r'\b'
        regex_patterns.append(regex_pattern)

    # combine all regex patterns to one with or condition
    combined_regex = "|".join(regex_patterns)

    return combined_regex


def proportional_stratified_sample(df, strata_cols, total_samples, random_state=42):

    # get the size of all strata and their proportions
    stratum_counts = df.groupby(strata_cols).size()
    stratum_proportions = stratum_counts / stratum_counts.sum()
    
    # get the proportional count for each stratum
    stratum_sample_counts = (stratum_proportions * total_samples).round().astype(int)
    
    # empty list to store the samples for each stratum
    sampled_list = []

    # loop over all combinations
    for stratum_vals, n_samples in stratum_sample_counts.items():

        # set up logical mask and sub df
        stratum_df = df
        mask = True

        # update mask based on all conditions and filter based on it
        for col, val in zip(strata_cols, stratum_vals):
            mask &= df[col] == val
        stratum_df = df[mask]
        
        # take either correct number of samples ar all observations if fewer than needed samples
        n_samples = min(n_samples, len(stratum_df))
        sampled_list.append(stratum_df.sample(n=n_samples, random_state=random_state))
    
    # concatenate all strata samples and return
    sampled_df = pd.concat(sampled_list).reset_index(drop=True)
    return sampled_df


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
parliamentary_questions_df['sentences'] = parliamentary_questions_df['clean_text'].apply(split_sentences)

# export the df as a csv
parliamentary_questions_df.to_csv("../01_data/raw_datasets/parliamentary_questions_df.csv", index=False)

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

