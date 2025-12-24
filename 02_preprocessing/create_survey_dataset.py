# load all required libraries
import pandas as pd

# load the BES datasets
bes_2015_df = pd.read_spss("../01_data/empirical_analysis/bes/survey_data/raw/2015/bes_2015.sav")
bes_2017_df = pd.read_spss("../01_data/empirical_analysis/bes/survey_data/raw/2017/bes_2017.sav")
bes_2019_df = pd.read_spss("../01_data/empirical_analysis/bes/survey_data/raw/2019/bes_2019.sav")

# load census data
census_df = pd.read_csv("../01_data/empirical_analysis/bes/census_data/cleaned/bes_census_cleaned.csv")

# set which columns to keep
cols_to_keep_2015 = ["b01", "b02", "e01", "Age", "y09", "Constit_Code", "Constit_Name", "wt_combined_main_capped"]
cols_to_keep_2017_2019 = ["b01", "b02", "e01", "Age", "y09", "Constit_Code", "Constit_Name", "wt_vote"]
cols_remapping = {
    "b01": "did_vote",
    "b02": "vote_decision",
    "e01": "left_right",
    "Age": "age",
    "y09": "gender",
    "Constit_Code": "ons_const_code",
    "Constit_Name": "constituency_name",
    "wt_combined_main_capped": "weight",
    "wt_vote": "weight"
}

# subset dataframes to relevant columns
bes_2015_df = bes_2015_df.loc[:, cols_to_keep_2015]
bes_2017_df = bes_2017_df.loc[:, cols_to_keep_2017_2019]
bes_2019_df = bes_2019_df.loc[:, cols_to_keep_2017_2019]

# rename column names
bes_2015_df = bes_2015_df.rename(columns=cols_remapping)
bes_2017_df = bes_2017_df.rename(columns=cols_remapping)
bes_2019_df = bes_2019_df.rename(columns=cols_remapping)

# subset to only people who voted
bes_2015_df = bes_2015_df[bes_2015_df["did_vote"] == "Yes, voted"]
bes_2017_df = bes_2017_df[bes_2017_df["did_vote"] == "Yes, voted"]
bes_2019_df = bes_2019_df[bes_2019_df["did_vote"] == "Yes, voted"]

# subset to only Labour and Conservatives
parties = ["Labour", "Labour Party", "Conservatives", "Conservative Party"]
bes_2015_df = bes_2015_df[bes_2015_df["vote_decision"].isin(parties)]
bes_2017_df = bes_2017_df[bes_2017_df["vote_decision"].isin(parties)]
bes_2019_df = bes_2019_df[bes_2019_df["vote_decision"].isin(parties)]

# rename categories for 2019 df
bes_2019_df["vote_decision"] = bes_2019_df["vote_decision"].cat.rename_categories({
    "Labour Party": "Labour",
    "Conservative Party": "Conservatives"
})

# restrict to only the two categories you want
bes_2019_df["vote_decision"] = bes_2019_df["vote_decision"].cat.set_categories(["Labour", "Conservatives"])

# restrict gender in 2019 df to make it consistent
bes_2019_df["gender"] = bes_2019_df["gender"].cat.set_categories(["Male", "Female"])

# replace left right values and make numeric
bes_2015_df["left_right"] = (
    bes_2015_df["left_right"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype("Int64")
)

bes_2017_df["left_right"] = (
    bes_2017_df["left_right"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype("Int64")
)

bes_2019_df["left_right"] = (
    bes_2019_df["left_right"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype("Int64")
)

# convert age to integer
bes_2015_df["age"] = pd.to_numeric(bes_2015_df["age"], errors="coerce").astype("Int64")
bes_2017_df["age"] = pd.to_numeric(bes_2017_df["age"], errors="coerce").astype("Int64")
bes_2019_df["age"] = pd.to_numeric(bes_2019_df["age"], errors="coerce").astype("Int64")

# merge with census data
bes_2015_census_df = pd.merge(bes_2015_df, census_df[["ONS ID", "under_30", "over_65"]], how="inner", left_on="ons_const_code", right_on="ONS ID").drop(columns=["ONS ID"])
bes_2017_census_df = pd.merge(bes_2017_df, census_df[["ONS ID", "under_30", "over_65"]], how="inner", left_on="ons_const_code", right_on="ONS ID").drop(columns=["ONS ID"])
bes_2019_census_df = pd.merge(bes_2019_df, census_df[["ONS ID", "under_30", "over_65"]], how="inner", left_on="ons_const_code", right_on="ONS ID").drop(columns=["ONS ID"])

# check for lost constituencies
print(f"Lost constituencies for 2015: {len(bes_2015_df["constituency_name"].unique()) - len(bes_2015_census_df["constituency_name"].unique())}")
print(f"Lost constituencies for 2017: {len(bes_2017_df["constituency_name"].unique()) - len(bes_2017_census_df["constituency_name"].unique())}")
print(f"Lost constituencies for 2019: {len(bes_2019_df["constituency_name"].unique()) - len(bes_2019_census_df["constituency_name"].unique())}")

# drop vote decision column
bes_2015_census_df = bes_2015_census_df.drop(columns=["did_vote"])
bes_2017_census_df = bes_2017_census_df.drop(columns=["did_vote"])
bes_2019_census_df = bes_2019_census_df.drop(columns=["did_vote"])

# add year variable
bes_2015_census_df["year"] = 2015
bes_2017_census_df["year"] = 2017
bes_2019_census_df["year"] = 2019

# set column order
col_order = ["vote_decision", "year", "ons_const_code", "constituency_name", "under_30", "over_65", "age", "gender", "left_right", "weight"]
bes_2015_census_df = bes_2015_census_df.loc[:, col_order]
bes_2017_census_df = bes_2017_census_df.loc[:, col_order]
bes_2019_census_df = bes_2019_census_df.loc[:, col_order]

# create one concatenated df
bes_census_df_combined = pd.concat([bes_2015_census_df, bes_2017_census_df, bes_2019_census_df], axis=0, ignore_index=True)

# export the datasets
bes_2015_census_df.to_csv("../01_data/empirical_analysis/bes/survey_data/cleaned/2015/bes_2015.csv", index=False)
bes_2017_census_df.to_csv("../01_data/empirical_analysis/bes/survey_data/cleaned/2017/bes_2017.csv", index=False)
bes_2019_census_df.to_csv("../01_data/empirical_analysis/bes/survey_data/cleaned/2019/bes_2019.csv", index=False)
bes_census_df_combined.to_csv("../01_data/empirical_analysis/bes/survey_data/cleaned/all_years.csv", index=False)