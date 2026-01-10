# load all libraries
import pandas as pd
import requests
import wikipediaapi
from dateutil.relativedelta import relativedelta
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.preprocessing import clean_text, split_sentences

# load the main parlspeech dataset
parlspeech_df = pd.read_csv("../01_data/raw_datasets/parlspeech_dataset.csv")

# load the general election results
results_2010 = pd.read_csv("../01_data/empirical_analysis/election_results/results_2010.csv")
results_2015 = pd.read_csv("../01_data/empirical_analysis/election_results/results_2015.csv")
results_2017 = pd.read_csv("../01_data/empirical_analysis/election_results/results_2017.csv")

# load the census BES dataset
bes_census_df = pd.read_spss("../01_data/empirical_analysis/bes/bes_census/bes_census.sav")

# set the election dates
election_dates = pd.to_datetime(["2010-05-06", "2015-05-07", "2017-06-08", "2019-12-12"])

# subset parlspeech df to research period
parlspeech_df["date"] = pd.to_datetime(parlspeech_df["date"])
research_period_df = parlspeech_df[parlspeech_df["date"] >= election_dates[0]]

# remove all speeches given by the speaker
research_period_df = research_period_df[research_period_df["speaker"] != "CHAIR"]
research_period_df = research_period_df[research_period_df["chair"] != True]

# remove all speeches where speaker is not assigned a party
research_period_df.dropna(subset=["party"], inplace=True)

# clean the text column
punkt_param = PunktParameters()
abbreviations = ['hon', 'mr', 'mrs', 'dr', 'ms', 'sir', 'prof']
punkt_param.abbrev_types = set(abbreviations)
tokenizer = PunktSentenceTokenizer(punkt_param)
research_period_df['text'] = research_period_df['text'].apply(clean_text)

# cut the df into sub dfs for each distinct period
first_period = research_period_df[research_period_df["date"] < election_dates[1]]
second_period = research_period_df[(research_period_df["date"] >= election_dates[1]) & (research_period_df["date"] < election_dates[2])]
third_period = research_period_df[(research_period_df["date"] >= election_dates[2]) & (research_period_df["date"] < election_dates[3])]

# get unique speakers for each individual period
speakers_first_period = first_period[["speaker", "date"]].drop_duplicates(subset=["speaker"])
speakers_second_period = second_period[["speaker", "date"]].drop_duplicates(subset=["speaker"])
speakers_third_period = third_period[["speaker", "date"]].drop_duplicates(subset=["speaker"])

# define functions to retrieve data from parliamentary database
def get_id_gender(base, name, date):
    r = requests.get(
        f"{base}/Members/SearchHistorical",
        params={"Name": name, "dateToSearchFor": date}
    )
    r.raise_for_status()
    items = r.json()["items"]
    if items:
        id = items[0]["value"]["id"]
        gender = items[0]["value"]["gender"]
        return id, gender
    else:
        return None, None
    
def get_constituency_backbencher(base, mp_id, date):

    target_date = pd.to_datetime(date)
    r = requests.get(f"{base}/Members/{mp_id}/Biography")
    r.raise_for_status()

    representations = r.json()["value"]["representations"]
    government_posts = r.json()["value"]["governmentPosts"]
    opposition_posts = r.json()["value"]["oppositionPosts"]

    constituency_name = None
    constituency_id = None

    for rep in representations:
        start_date = pd.to_datetime(rep["startDate"])
        end_date = pd.to_datetime(rep["endDate"])
        if start_date <= target_date <= end_date:
            constituency_name = rep["name"]
            constituency_id = rep["id"]

    backbencher = True

    for gov in government_posts:
        start_date = pd.to_datetime(gov["startDate"])
        end_date = pd.to_datetime(gov["endDate"])
        if start_date <= target_date <= end_date:
            backbencher = False
    
    for opp in opposition_posts:
        start_date = pd.to_datetime(opp["startDate"])
        end_date = pd.to_datetime(opp["endDate"])
        if start_date <= target_date <= end_date:
            backbencher = False

    return constituency_name, constituency_id, backbencher

base = "https://members-api.parliament.uk/api"
dfs_list = [speakers_first_period, speakers_second_period, speakers_third_period]

nonfound_mps = set()

for df in dfs_list:

    df["mp_id"] = None
    df["gender"] = None

    for idx, row in df.iterrows():
        name = row["speaker"]
        date = row["date"]

        mp_id, gender = get_id_gender(base, name, date)

        if mp_id is None:
            nonfound_mps.add(name)
        else:
            df.at[idx, "mp_id"] = mp_id
            df.at[idx, "gender"] = gender

# build manual remapping for names and dates
remapping_df = pd.read_csv("../01_data/empirical_analysis/manual_remappings/remapping_parl_database.csv")

# apply this to the speakers datasets
def apply_remapping(df, remapping_df):
    for _, row in remapping_df.iterrows():
        mask = df["speaker"] == row["old_name"]

        if not mask.any():
            continue

        if pd.notna(row["new_name"]):
            df.loc[mask, "speaker"] = row["new_name"]

        if pd.notna(row["new_date"]):
            df.loc[mask, "date"] = row["new_date"]

    return df

for df in dfs_list:
    apply_remapping(df, remapping_df)
    missing_mask = df["mp_id"].isna()
    if not missing_mask.any():
        continue
    for idx, row in df.loc[missing_mask].iterrows():
        name = row["speaker"]
        date = row["date"]

        mp_id, gender = get_id_gender(base, name, date)

        if mp_id is not None:
            df.at[idx, "mp_id"] = mp_id
            df.at[idx, "gender"] = gender

    print(f"Number of missing values: {sum(df["mp_id"].isna())}")

# apply name remapping to the main dfs as well
for _, row in remapping_df.iterrows():
    mask_1 = first_period["speaker"] == row["old_name"]
    mask_2 = second_period["speaker"] == row["old_name"]
    mask_3 = third_period["speaker"] == row["old_name"]
    if pd.notna(row["new_name"]):
        first_period.loc[mask_1, "speaker"] = row["new_name"]
        second_period.loc[mask_2, "speaker"] = row["new_name"]
        third_period.loc[mask_3, "speaker"] = row["new_name"]


nonfound_constituencies = set()

for df in dfs_list:

    df["constituency_name"] = None
    df["constituency_id"] = None
    df["backbencher"] = None

    for idx, row in df.iterrows():
        mp_id = row["mp_id"]
        date = row["date"]
        const_name, const_id, backbencher_status = get_constituency_backbencher(base, mp_id, date)
    
        if const_id is None:
            nonfound_constituencies.add(row["speaker"])
        else:
            df.at[idx, "constituency_name"] = const_name
            df.at[idx, "constituency_id"] = const_id
            df.at[idx, "backbencher"] = backbencher_status

    # drop rows with missing constituency ids (these have been checked and should not be present)
    df.dropna(subset=["constituency_id"], inplace=True)


# merge with election results on constituency name and MP name
election_results_dfs_list = [results_2010, results_2015, results_2017]
cols_to_keep = ["ONS ID", "Constituency name", "Country name", "Member first name", "Member surname", "Valid votes", "Majority", "vulnerability"]

# compute vulnerability variable
for election_df in election_results_dfs_list:
    election_df["vulnerability"] = election_df["Majority"]/election_df["Valid votes"]*100
    election_df.drop(columns=[c for c in election_df.columns if c not in cols_to_keep], inplace=True)

# create a surname column
for df in dfs_list:
    df["surname"] = df["speaker"].str.split().str[-1]

# join on surname and constituency name
speakers_first_period = pd.merge(speakers_first_period, results_2010, how="left", left_on=["constituency_name", "surname"], right_on=["Constituency name", "Member surname"])
speakers_second_period = pd.merge(speakers_second_period, results_2015, how="left", left_on=["constituency_name", "surname"], right_on=["Constituency name", "Member surname"])
speakers_third_period = pd.merge(speakers_third_period, results_2017, how="left", left_on=["constituency_name", "surname"], right_on=["Constituency name", "Member surname"])

# upload the merged dfs to manually correct and add by elections
speakers_first_period.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_2010.csv", index=False)
speakers_second_period.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_2015.csv", index=False)
speakers_third_period.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_2017.csv", index=False)

# manually add by-elections and load the data
speakers_first_period = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_2010.csv")
speakers_second_period = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_2015.csv")
speakers_third_period = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_2017.csv")


# merge with census data

# get all unique constituencies
ons_union_df = (pd.concat([speakers_first_period[["ONS ID", "constituency_name"]], speakers_second_period[["ONS ID", "constituency_name"]], speakers_third_period[["ONS ID", "constituency_name"]]], ignore_index=True).dropna().drop_duplicates(subset=["ONS ID"]).reset_index(drop=True))

# select census data to keep and subset
census_cols = ["ONSConstID", "c11Age0to4", "c11Age5to7", "c11Age8to9", "c11Age10to14", "c11Age15", "c11Age16to17", "c11Age18to19", "c11Age20to24", "c11Age25to29",
               "c11Age30to44", "c11Age45to59", "c11Age60to64", "c11Age65to74", "c11Age75to84", "c11Age85to89", "c11Age90plus"]
bes_census_df = bes_census_df.drop(columns=[c for c in bes_census_df.columns if c not in census_cols])

# left join on all constituencies
all_constituencies = pd.merge(ons_union_df, bes_census_df, how="left", left_on="ONS ID", right_on="ONSConstID").drop(columns=["ONSConstID"])

# calculate age categories
under_30_cols = census_cols[1:10]
over_65_cols = census_cols[13:]
all_constituencies["under_30"] = all_constituencies[under_30_cols].sum(axis=1)
all_constituencies["over_65"] = all_constituencies[over_65_cols].sum(axis=1)

# drop irrelevant colums
all_constituencies = all_constituencies.drop(columns=[c for c in all_constituencies.columns if c not in ["ONS ID", "constituency_name", "under_30", "over_65"]])

# export for manual additions
all_constituencies.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_census.csv", index=False)

# read constituency data back in
all_constituencies = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_census.csv")

# join on speakers dataframes
speakers_first_period = pd.merge(speakers_first_period, all_constituencies, how="left", on="ONS ID")
speakers_second_period = pd.merge(speakers_second_period, all_constituencies, how="left", on="ONS ID")
speakers_third_period = pd.merge(speakers_third_period, all_constituencies, how="left", on="ONS ID")

# get date of birth via wikipedia
wiki = wikipediaapi.Wikipedia(
    user_agent="master_thesis_research",
    language="en"
)

headers = {
    "User-Agent": (
        "AcademicResearch-MasterThesis "
        "(contact: maximilian.weiland@uni-konstanz.de)"
    ),
    "Accept": "application/json",
}

def get_wikidata_id(page_title):

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "pageprops",
        "format": "json",
    }

    r = requests.get(url, params=params, headers=headers, timeout=15)

    if r.status_code != 200:
        return None

    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("pageprops", {}).get("wikibase_item")

    return None


def get_birth_date(mp_name):
    wikidata_id = get_wikidata_id(mp_name)

    if wikidata_id is None:
        wikidata_id = get_wikidata_id(f"{mp_name} (UK politician)")
        if wikidata_id is None:
            return None

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    r = requests.get(url, headers=headers, timeout=15)

    if r.status_code != 200:
        return None

    try:
        claims = r.json()["entities"][wikidata_id]["claims"]
        birth_time = claims["P569"][0]["mainsnak"]["datavalue"]["value"]["time"]
        return birth_time[1:11]
    except (KeyError, IndexError, ValueError):
        return None

speakers_first_period["birth_date"] = speakers_first_period["speaker"].apply(get_birth_date)
speakers_second_period["birth_date"] = speakers_second_period["speaker"].apply(get_birth_date)
speakers_third_period["birth_date"] = speakers_third_period["speaker"].apply(get_birth_date)

speakers_first_period.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_birthdates_2010.csv", index=False)
speakers_second_period.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_birthdates_2015.csv", index=False)
speakers_third_period.to_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_birthdates_2017.csv", index=False)

speakers_first_period = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_birthdates_2010.csv")
speakers_second_period = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_birthdates_2015.csv")
speakers_third_period = pd.read_csv("../01_data/empirical_analysis/manual_remappings/manual_correction_birthdates_2017.csv")

# drop duplicates which occurred during merging
speakers_first_period = speakers_first_period.drop_duplicates(subset=["speaker"])
speakers_second_period = speakers_second_period.drop_duplicates(subset=["speaker"])
speakers_third_period = speakers_third_period.drop_duplicates(subset=["speaker"])

cols_to_keep = ["speaker", "birth_date", "gender", "constituency_name", "backbencher", "ONS ID", "vulnerability", "under_30", "over_65"]
speakers_first_period.drop(columns=[c for c in speakers_first_period.columns if c not in cols_to_keep], inplace=True)
speakers_second_period.drop(columns=[c for c in speakers_second_period.columns if c not in cols_to_keep], inplace=True)
speakers_third_period.drop(columns=[c for c in speakers_third_period.columns if c not in cols_to_keep], inplace=True)

# merge back with speeches sub-dataframes
first_period_merged = pd.merge(first_period, speakers_first_period, how="inner", on="speaker")
second_period_merged = pd.merge(second_period, speakers_second_period, how="inner", on="speaker")
third_period_merged = pd.merge(third_period, speakers_third_period, how="inner", on="speaker")

# check that no speakers are lost due to mismatches
print(f"Number of all speakers: {len(speakers_first_period)}. Number of all speakers after join: {len(first_period_merged.groupby("speaker"))}")
print(f"Number of all speakers: {len(speakers_second_period)}. Number of all speakers after join: {len(second_period_merged.groupby("speaker"))}")
print(f"Number of all speakers: {len(speakers_third_period)}. Number of all speakers after join: {len(third_period_merged.groupby("speaker"))}")

# assure all date columns are converted to datetime
first_period_merged["date"] = pd.to_datetime(first_period_merged["date"])
first_period_merged["birth_date"] = pd.to_datetime(first_period_merged["birth_date"])
second_period_merged["date"] = pd.to_datetime(second_period_merged["date"])
second_period_merged["birth_date"] = pd.to_datetime(second_period_merged["birth_date"])
third_period_merged["date"] = pd.to_datetime(third_period_merged["date"])
third_period_merged["birth_date"] = pd.to_datetime(third_period_merged["birth_date"])

# calculate age
def calculate_age(birth, ref):
    return relativedelta(ref, birth).years

first_period_merged["age"] = first_period_merged.apply(lambda row: calculate_age(row["birth_date"], row["date"]), axis=1)
second_period_merged["age"] = second_period_merged.apply(lambda row: calculate_age(row["birth_date"], row["date"]), axis=1)
third_period_merged["age"] = third_period_merged.apply(lambda row: calculate_age(row["birth_date"], row["date"]), axis=1)

first_period_merged["gender"] = first_period_merged["gender"].replace({"M": "male", "F": "female"})
second_period_merged["gender"] = second_period_merged["gender"].replace({"M": "male", "F": "female"})
third_period_merged["gender"] = third_period_merged["gender"].replace({"M": "male", "F": "female"})

first_period_merged.rename(columns={"ONS ID": "ons_id"}, inplace=True)
second_period_merged.rename(columns={"ONS ID": "ons_id"}, inplace=True)
third_period_merged.rename(columns={"ONS ID": "ons_id"}, inplace=True)

cols_to_keep = ["date", "agenda", "text", "speechnumber", "speaker", "party", "chair", "age", "gender", "birth_date", "vulnerability", "backbencher", "constituency_name", "ons_id", "under_30", "over_65"]
first_period_merged = first_period_merged.loc[:, cols_to_keep]
second_period_merged = second_period_merged.loc[:, cols_to_keep]
third_period_merged = third_period_merged.loc[:, cols_to_keep]

# concatenate all sub dataframes
final_df = pd.concat(
    [first_period_merged, second_period_merged, third_period_merged],
    axis=0,
    ignore_index=True
)

# create an id variable
final_df["speech_id"] = final_df.index

# add days until election variable

# for each speech search for the next election
idx = election_dates.searchsorted(final_df["date"], side="right")

# count days until this election and add as a variable
final_df["days_until_election"] = [
    (election_dates[i] - d).days if i < len(election_dates) else None
    for d, i in zip(final_df["date"], idx)
]

# export the df
final_df.to_csv("../01_data/empirical_analysis/main_speech_datasets/researchperiod_speechlevel.csv", index=False)

# split into individual sentences and export this version as well
final_df['sentences'] = final_df['text'].apply(lambda x: split_sentences(x, tokenizer))
sentences_df = final_df.explode('sentences').reset_index(drop=True)
sentences_df = sentences_df.rename(columns={'sentences': 'sentence'})

# correct column order
col_order = ["date", "days_until_election", "agenda", "text", "sentence", "speech_id", "speechnumber", "speaker", "party", "chair", "age", "gender", "birth_date", "vulnerability", "backbencher", "constituency_name", "ons_id", "under_30", "over_65"]
sentences_df = sentences_df.loc[:, col_order]

# export the sentences df
sentences_df.to_csv("../01_data/empirical_analysis/main_speech_datasets/researchperiod_sentencelevel.csv", index=False)
