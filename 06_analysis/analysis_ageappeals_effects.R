################################ Data Loading ################################
library(tidyverse)
library(sandwich)
library(lmtest)
library(lme4)
library(lubridate)
library(effects)
library(margins)
library(jsonlite)
library(MASS)

bes_path <- "../01_data/empirical_analysis/bes/survey_data/cleaned/all_years.csv"
speech_df_path <- "../01_data/empirical_analysis/speech_datasets/datasets_cluster_labels/parl_questions_clusterlabels.csv"

bes_df <- read_csv(bes_path)
questions_df <- read_csv(speech_df_path, col_types = cols(
  date = col_date(),
  party = col_factor(),
  gender = col_factor(),
  birth_date = col_date()))

source("functions_analysis.R")

################################ Preprocessing ################################

# preprocess survey data
bes_df <- bes_df |>
  
  # delete missing rows
  filter(if_all(c(age, vote_decision), ~ !is.na(.))) |>
  
  # create variables for regression
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

# preprocess speech data
questions_df <- questions_df |>
  
  # filter for only major parties
  filter(party %in% c("Con", "Lab")) |>
  
  # create election year variable, count appeals per sentence
  mutate(
    party = factor(party, levels = c("Con", "Lab")),
    election_year = case_when(
      date < as.Date("2015-05-07") ~ 2015,
      date >= as.Date("2015-05-07") & date < as.Date("2017-06-08") ~ 2017,
      date >= as.Date("2017-06-08") ~ 2019
    ),
    stances = map(str_replace_all(stances, "'", "\""), fromJSON),
    sg_categories = map(str_replace_all(sg_categories, "'", "\""), fromJSON),
    sentence_id = row_number(),

    young_positive = map2_int(
      sg_categories, stances,
      ~ sum(.x == "Young people" & .y == "positive")
    ),

    elderly_positive = map2_int(
      sg_categories, stances,
      ~ sum(.x == "Elderly people" & .y == "positive")
    ),

    total_positive = map_int(
      stances,
      ~ sum(.x == "positive")
    ),
    total_negative = map_int(
      stances,
      ~ sum(.x == "negative")
    )
  )
  
# aggregate counts per speech
speech_counts <- questions_df |> 
  group_by(speech_id) |> 
  summarise(
    date = first(date),
    election_year = first(election_year),
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
    total_positive = sum(total_positive, na.rm = TRUE),
    total_negative= sum(total_negative, na.rm = TRUE),
    total_appeals = sum(total_positive, na.rm = TRUE) + sum(total_negative, na.rm = TRUE),
    total_sentences = n(),
    .groups = "drop"
  )

# aggregate appeals by constituency for each election cycle
# consider only legislators who generally made appeals
mp_counts <- speech_counts |>
  group_by(election_year, ons_id) |>
  filter(sum(total_appeals) > 0) |> 
  summarise(
    party = first(party),
    backbencher = first(backbencher),
    gender = first(gender),
    vulnerability = first(vulnerability),
    ons_id = first(ons_id),
    total_positive = sum(total_positive, na.rm = TRUE),
    total_negative = sum(total_negative, na.rm = TRUE),
    total_appeals = sum(total_appeals, na.rm = TRUE),
    share_youth_appeals = (sum(young_positive_total, na.rm = TRUE) / sum(total_appeals, na.rm = TRUE)) * 100,
    share_elderly_appeals = (sum(elderly_positive_total, na.rm = TRUE) / sum(total_appeals, na.rm = TRUE)) * 100,
    .groups = "drop"
  )

# join with survey data
mp_bes_merged <- mp_counts |>
  inner_join(
    bes_df,
    by = c(
      "election_year" = "year",
      "ons_id" = "ons_const_code"
    ),
    suffix = c("_mp", "_respondent")
  )

########################### Effect of Youth Appeals ############################

# specify the cluster variable
cluster_var <- factor(mp_bes_merged$ons_id)

# logistic regression with clustered standard errors
youth_logreg <- glm(
  labour_choice ~ share_youth_appeals * age_group * party + under_30 + gender_respondent + factor(election_year),
  family = binomial(link = "logit"),
  data = mp_bes_merged,
  #weights = weight
)
youth_logreg_cse <- coeftest(youth_logreg, vcov = vcovCL(youth_logreg, cluster = cluster_var))
youth_logreg_cse

# application of parametric bootstrap
youth_vals <- c(1:100)
parties <- c("Con", "Lab")
age_groups <- c("old", "young")
pred_probs_youth <- pb_predicted_probs(
  b_iterations = 1000,
  regression = youth_logreg,
  iv = "youth",
  cluster_var = cluster_var,
  df = mp_bes_merged,
  age_appeal_vals = youth_vals,
  party_vals = parties,
  age_group_vals = age_groups
)
write.csv(pred_probs_youth, "analysis_results/pred_probs_youth_appeals.csv", row.names = FALSE)

# visualize predicted probabilities
ggplot(pred_probs_youth,
       aes(x = age_appeal_share,
           y = pred,
           color = age_group,
           fill = age_group)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.2,
              color = NA) +
  facet_wrap(~party) +
  labs(
    x = "Youth Appeal Share",
    y = "Predicted Probability (Vote = Lab)",
    color = "Age Group",
    fill = "Age Group"
  ) +
  theme_minimal()

# difference in AMEs between young and old

# manual hypothesis test

# get average marginal effects for all subgroups
v_cov_youth <- vcovCL(youth_logreg, cluster = cluster_var)
me_cse_youth <- margins(
  youth_logreg,
  variables = "share_youth_appeals",
  at = list(
    age_group = c("young", "old"),
    party = c("Con", "Lab")
  ),
  vcov = v_cov_youth
)
me_df_cse_youth <- summary(me_cse_youth)

# calculate differences between young and old for Labour and Conservatives
lab_diff_youth <- with(
  subset(me_df_cse_youth, party == "Lab"),
  AME[age_group == "young"] - AME[age_group == "old"]
)
con_diff_youth <- with(
  subset(me_df_cse_youth, party == "Con"),
  AME[age_group == "young"] - AME[age_group == "old"]
)

# calculate standard error of the difference and compute z- and p-values
lab_se_diff_youth <- sqrt(
  subset(me_df_cse_youth, party == "Lab")$SE[1]^2 +
    subset(me_df_cse_youth, party == "Lab")$SE[2]^2
)
con_se_diff_youth <- sqrt(
  subset(me_df_cse_youth, party == "Con")$SE[1]^2 +
    subset(me_df_cse_youth, party == "Con")$SE[2]^2
)
lab_z_youth <- lab_diff_youth / lab_se_diff_youth
con_z_youth <- con_diff_youth / con_se_diff_youth
lab_p_youth <- 2 * (1 - pnorm(abs(lab_z_youth)))
con_p_youth <- 2 * (1 - pnorm(abs(con_z_youth)))
cat(
  "AME of the difference between young and old for Labour:", lab_diff_youth, "\n",
  "Associated p-value:", lab_p_youth, "\n\n",
  "AME of the difference between young and old for Conservatives:", con_diff_youth, "\n",
  "Associated p-value:", con_p_youth, "\n"
)

# compute difference of average marginal effects for each iteration
ame_diffs_youth <- diff_in_ame(
  b_iterations = 1000,
  regression = youth_logreg,
  iv = "share_youth_appeals",
  cluster_var = cluster_var
)

# show side by side histograms
labour_diffs_ame_youth <- ame_diffs_youth$labour_boot
con_diffs_ame_youth <- ame_diffs_youth$con_boot
ame_boot_df_youth <- data.frame(
  Labour = labour_diffs_ame_youth,
  Conservative = con_diffs_ame_youth
) |> 
  pivot_longer(cols = everything(), names_to = "party", values_to = "ame_diff")

ggplot(ame_boot_df_youth, aes(x = ame_diff, fill = party)) +
  geom_histogram(position = "dodge", bins = 30, alpha = 0.7, color = "black") +
  facet_wrap(~ party) +
  labs(
    title = "Bootstrapped Differences in AME: Young vs Old",
    x = "AME Difference",
    y = "Frequency"
  ) +
  theme_minimal()

ame_summary_df_youth <- ame_boot_df_youth |> 
  group_by(party) |> 
  summarise(
    mean_diff = mean(ame_diff),
    q_025 = quantile(ame_diff, 0.025),
    q_975 = quantile(ame_diff, 0.975)
  )

write.csv(ame_boot_df_youth, "analysis_results/ame_diffs_youth.csv", row.names = FALSE)

########################### Effect of Elderly Appeals ##########################

elderly_logreg <- glm(
  conservatives_choice ~ share_elderly_appeals * age_group * party + over_65 + gender_respondent + factor(election_year),
  family = binomial(link = "logit"),
  data = mp_bes_merged,
  #weights = weight
)
elderly_logreg_cse <- coeftest(elderly_logreg, vcov = vcovCL(elderly_logreg, cluster = cluster_var))
elderly_logreg_cse

# application of parametric bootstrap
elderly_vals <- c(1:100)
parties <- c("Con", "Lab")
age_groups <- c("old", "young")
pred_probs_elderly <- pb_predicted_probs(
  b_iterations = 1000,
  regression = elderly_logreg,
  iv = "elderly",
  cluster_var = cluster_var,
  df = mp_bes_merged,
  age_appeal_vals = elderly_vals,
  party_vals = parties,
  age_group_vals = age_groups
)
write.csv(pred_probs_elderly, "analysis_results/pred_probs_elderly_appeals.csv", row.names = FALSE)

# visualize predicted probabilities
ggplot(pred_probs_elderly,
       aes(x = age_appeal_share,
           y = pred,
           color = age_group,
           fill = age_group)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.2,
              color = NA) +
  facet_wrap(~party) +
  labs(
    x = "Elderly Appeal Share",
    y = "Predicted Probability (Vote = Con)",
    color = "Age Group",
    fill = "Age Group"
  ) +
  theme_minimal()

# difference in AMEs between young and old

# manual hypothesis test

# get average marginal effects for all subgroups
v_cov_elderly <- vcovCL(elderly_logreg, cluster = cluster_var)
me_elderly_cse <- margins(
  elderly_logreg,
  variables = "share_elderly_appeals",
  at = list(
    age_group = c("young", "old"),
    party = c("Con", "Lab")
  ),
  vcov = v_cov_elderly
)
me_elderly_df_cse <- summary(me_elderly_cse)

# calculate differences between young and old for Labour and Conservatives
lab_diff_elderly <- with(
  subset(me_elderly_df_cse, party == "Lab"),
  AME[age_group == "old"] - AME[age_group == "young"]
)
con_diff_elderly <- with(
  subset(me_elderly_df_cse, party == "Con"),
  AME[age_group == "old"] - AME[age_group == "young"]
)

# calculate standard error of the difference and compute z- and p-values
lab_se_diff_elderly <- sqrt(
  subset(me_elderly_df_cse, party == "Lab")$SE[1]^2 +
    subset(me_elderly_df_cse, party == "Lab")$SE[2]^2
)
con_se_diff_elderly <- sqrt(
  subset(me_elderly_df_cse, party == "Con")$SE[1]^2 +
    subset(me_elderly_df_cse, party == "Con")$SE[2]^2
)
lab_z_elderly <- lab_diff_elderly / lab_se_diff_elderly
con_z_elderly <- con_diff_elderly / con_se_diff_elderly
lab_p_elderly <- 2 * (1 - pnorm(abs(lab_z_elderly)))
con_p_elderly <- 2 * (1 - pnorm(abs(con_z_elderly)))
cat(
  "AME of the difference between old and young for Labour:", lab_diff_elderly, "\n",
  "Associated p-value:", lab_p_elderly, "\n\n",
  "AME of the difference between old and young for Conservatives:", con_diff_elderly, "\n",
  "Associated p-value:", con_p_elderly, "\n"
)

# compute difference of average marginal effects for each iteration
ame_diffs_elderly <- diff_in_ame(
  b_iterations = 1000,
  regression = elderly_logreg,
  iv = "share_elderly_appeals",
  cluster_var = cluster_var
)

# show side by side histograms
labour_diffs_ame_elderly <- ame_diffs_elderly$labour_boot
con_diffs_ame_elderly <- ame_diffs_elderly$con_boot
ame_boot_df_elderly <- data.frame(
  Labour = labour_diffs_ame_elderly,
  Conservative = con_diffs_ame_elderly
) |> 
  pivot_longer(cols = everything(), names_to = "party", values_to = "ame_diff")

ggplot(ame_boot_df_elderly, aes(x = ame_diff, fill = party)) +
  geom_histogram(position = "dodge", bins = 30, alpha = 0.7, color = "black") +
  facet_wrap(~ party) +
  labs(
    title = "Bootstrapped Differences in AME: Young vs Old",
    x = "AME Difference",
    y = "Frequency"
  ) +
  theme_minimal()

ame_summary_df_elderly <- ame_boot_df_elderly |> 
  group_by(party) |> 
  summarise(
    mean_diff = mean(ame_diff),
    q_025 = quantile(ame_diff, 0.025),
    q_975 = quantile(ame_diff, 0.975)
  )

write.csv(ame_boot_df_elderly, "analysis_results/ame_diffs_elderly.csv", row.names = FALSE)
