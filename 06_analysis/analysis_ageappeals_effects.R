################################ Data Loading ################################
library(tidyverse)
library(sandwich)
library(lme4)

bes_path <- "../01_data/empirical_analysis/bes/survey_data/cleaned/all_years.csv"
speech_df_path <- "../01_data/empirical_analysis/speech_datasets/datasets_cluster_labels/parl_questions_clusterlabels.csv"

bes_df <- read_csv(bes_path)
questions_df <- read_csv(speech_df_path, col_types = cols(
  date = col_date(),
  party = col_factor(),
  gender = col_factor(),
  birth_date = col_date()))

################################ Preprocessing ################################

# convert categorical variables to factor and create age group and vote choice variables
bes_df <- bes_df |> 
  mutate(
    gender = factor(gender,
                    levels = c("Male", "Female")),
    age_group = factor(
      case_when(
        age < 35 ~ "young",
        age >= 35 & age <= 65 ~ "middle",
        age > 65 ~ "old"
      ),
      levels = c("young", "middle", "old")
    ),
    labour_choice = case_when(
      vote_decision == "Labour" ~ 1,
      vote_decision == "Conservatives" ~ 0
    ),
    conservatives_choice = case_when(
      vote_decision == "Conservatives" ~ 1,
      vote_decision == "Labour" ~ 0
    )
  )
  
# unpack lists, count appeals per sentence
questions_df <- questions_df |> 
  mutate(
    stances = map(str_replace_all(stances, "'", "\""), fromJSON),
    sg_categories = map(str_replace_all(sg_categories, "'", "\""), fromJSON),
    sentence_id = row_number(),
    young_positive = map2_int(sg_categories, stances,
                              ~ sum(.x == "Young people" & .y == "positive")),
    elderly_positive = map2_int(sg_categories, stances,
                                ~ sum(.x == "Elderly people" & .y == "positive"))
  ) 

# filter for only sentences with appeals from Labour or Conservatives
questions_df <- questions_df |> 
  filter(party %in% c("Con", "Lab")) #|> 
#  filter(lengths(sg_categories) > 0)

# aggregate counts per speech
speech_counts <- questions_df |> 
  group_by(speech_id) |> 
  summarise(
    date = first(date),
    speaker = first(speaker),
    party   = first(party),
    age = first(age),
    gender = first(gender),
    vulnerability = first(vulnerability),
    backbencher = first(backbencher),
    ons_id = first(ons_id),
    young_positive_total = sum(young_positive, na.rm = TRUE),
    elderly_positive_total = sum(elderly_positive, na.rm = TRUE),
    young_positive_bool = ifelse(sum(young_positive, na.rm = TRUE) > 0, 1, 0),
    elderly_positive_bool = ifelse(sum(elderly_positive, na.rm = TRUE) > 0, 1, 0),
    total_sentences = n(),
    .groups = "drop"
  )

speech_counts <- speech_counts |>
  mutate(
    election_year = case_when(
      date < as.Date("2015-05-07") ~ 2015,
      date >= as.Date("2015-05-07") & date < as.Date("2017-06-08") ~ 2017,
      date >= as.Date("2017-06-08") ~ 2019
    )
  )

# aggregate appeals by constituency
# by grouping on constituency I collapse two Labour MPs from the same constituency
# I manually checked that I do collapse counts from different parties
mp_counts <- speech_counts |>
  group_by(election_year, ons_id) |>
  summarise(
    party = first(party),
    ons_id = first(ons_id),
    total_youth_appeals = sum(young_positive_total, na.rm = TRUE),
    youth_appeals_per_speech = sum(young_positive_total, na.rm = TRUE) / n(),
    total_elderly_appeals = sum(elderly_positive_total, na.rm = TRUE),
    elderly_appeals_per_speech = sum(elderly_positive_total, na.rm = TRUE) / n(),
    .groups = "drop"
  )

# join with survey data
mp_bes_merged <- mp_counts |>
  inner_join(
    bes_df,
    by = c(
      "election_year" = "year",
      "ons_id" = "ons_const_code"
    )
  )

########################### Effect of Youth Appeals ############################

# specify the cluster variable
cluster_var <- mp_bes_merged$ons_id

youth_logreg <- glm(
  labour_choice ~ youth_appeals_per_speech * age_group * party + age + under_30,
  family = binomial(link = "logit"),
  data = mp_bes_merged
)
youth_logreg_cse <- coeftest(youth_logreg, vcov = vcovCL(youth_logreg, cluster = cluster_var))
youth_logreg_cse

youth_logreg_ml <- glmer(
  labour_choice ~ youth_appeals_per_speech * age_group * party + 
    age + under_30 + 
    (1 | ons_id),
  family = binomial(link = "logit"),
  data = mp_bes_merged
)
youth_logreg_ml

########################### Effect of Elderly Appeals ##########################

elderly_logreg <- glm(
  conservatives_choice ~ elderly_appeals_per_speech * age_group * party + age + over_65,
  family = binomial(link = "logit"),
  data = mp_bes_merged
)
elderly_logreg_cse <- coeftest(elderly_logreg, vcov = vcovCL(elderly_logreg, cluster = cluster_var))
elderly_logreg_cse

elderly_logreg_ml <- glmer(
  conservatives_choice ~ elderly_appeals_per_speech * age_group * party + 
    age + over_65 + 
    (1 | ons_id),
  family = binomial(link = "logit"),
  data = mp_bes_merged
)
elderly_logreg_ml
