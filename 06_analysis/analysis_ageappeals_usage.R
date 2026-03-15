################################ Data Loading ################################
library(tidyverse)
library(stringr)
library(jsonlite)
library(purrr)
library(pscl)
library(sandwich)
library(lmtest)
library(glmmTMB)
library(lme4)
library(MASS)
library(lubridate)
data_path <- "../01_data/empirical_analysis/speech_datasets/datasets_cluster_labels/parl_questions_clusterlabels.csv"
questions_df <- read_csv(data_path, col_types = cols(
  date = col_date(),
  party = col_factor(),
  gender = col_factor(),
  birth_date = col_date()))
source("functions_analysis.R")

################################ Preprocessing ################################

# preprocess speech data
questions_df <- questions_df |> 
  
  # unpack lists, count appeals per sentence
  mutate(
    backbencher = factor(ifelse(backbencher == TRUE, 1, 0),
                         levels = c(0, 1)),
    opposition = factor(ifelse(party == "Con", 0, 1),
                        levels = c(0, 1)),
    moderately_vulnerable = factor(case_when(
      vulnerability < 10 ~ 1,
      vulnerability >= 10 ~ 0
    ), levels = c(0, 1)),
    
    highly_vulnerable = factor(case_when(
      vulnerability < 5 ~ 1,
      vulnerability >= 5 ~ 0
    ), levels = c(0, 1)),
    stances = map(str_replace_all(stances, "'", "\""), fromJSON),
    sg_categories = map(str_replace_all(sg_categories, "'", "\""), fromJSON),
    sentence_id = row_number(),
    young_positive = map2_int(sg_categories, stances,
                              ~ sum(.x == "Young people" & .y == "positive")),
    elderly_positive = map2_int(sg_categories, stances,
                                ~ sum(.x == "Elderly people" & .y == "positive")),
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
    days_until_election = first(days_until_election),
    text = first(text),
    speaker = first(speaker),
    party = first(party),
    opposition = first(opposition),
    age = first(age),
    gender = first(gender),
    moderately_vulnerable = first(moderately_vulnerable),
    highly_vulnerable = first(highly_vulnerable),
    backbencher = first(backbencher),
    under_30 = first(under_30),
    over_65 = first(over_65),
    total_positive = sum(total_positive, na.rm = TRUE),
    positive_bool = ifelse(sum(total_positive, na.rm = TRUE) > 0, 1, 0),
    negative_bool = ifelse(sum(total_negative, na.rm = TRUE) > 0, 1, 0),
    total_negative = sum(total_negative, na.rm = TRUE),
    total_appeals = sum(total_positive, na.rm = TRUE) + sum(total_negative, na.rm = TRUE),
    appeal_bool = ifelse(sum(total_positive, na.rm = TRUE) + sum(total_negative, na.rm = TRUE) > 0, 1, 0),
    young_positive_total = sum(young_positive, na.rm = TRUE),
    elderly_positive_total = sum(elderly_positive, na.rm = TRUE),
    young_positive_bool = ifelse(sum(young_positive, na.rm = TRUE) > 0, 1, 0),
    elderly_positive_bool = ifelse(sum(elderly_positive, na.rm = TRUE) > 0, 1, 0),
    total_sentences = n(),
    log_total_sentences = log(n()),
    .groups = "drop"
  )

# reduce df to contain only appeals (positive or negative) from Labour or Tories
speech_counts_appeals <- speech_counts |> 
  filter(total_appeals > 0) |> 
  filter(party %in% c("Lab", "Con")) |> 
  mutate(party = factor(party, levels = c("Con", "Lab")))

############################## Election Strategy ###############################

cluster_var <- factor(speech_counts$speaker)

# test if appeals are more likely closer to next general election
reg_election_strategy_logit <- glm(
  appeal_bool ~ days_until_election + highly_vulnerable + backbencher + total_sentences,
  family = binomial(link = "logit"),
  data = speech_counts
)
reg_election_strategy_logit_cse <- coeftest(reg_election_strategy_logit,
                                            vcov = vcovCL(reg_election_strategy_logit,
                                                          cluster = cluster_var))
reg_election_strategy_logit_cse

# test if negative appeals are more likely by opposition parties
reg_negative_logit <- glm(
  negative_bool ~ opposition + days_until_election + highly_vulnerable + backbencher + age + total_sentences,
  family = binomial(link = "logit"),
  data = speech_counts
)
reg_negative_logit_cse <- coeftest(reg_negative_logit, vcov = vcovCL(reg_negative_logit, cluster = cluster_var))
reg_negative_logit_cse

################################ Youth Appeals ################################

# specify cluster variable for clustered standard errors
cluster_var <- factor(speech_counts_appeals$speaker)

# likelihood ratio test to choose between Poisson and Negative Binomial
poisson_model <- glm(
  young_positive_total ~ party + highly_vulnerable + under_30 + days_until_election + age + backbencher + offset(log_total_sentences),
  family = poisson(link = "log"),
  data = speech_counts_appeals
)
nb_model <- glm.nb(
  young_positive_total ~ party + highly_vulnerable + under_30 + days_until_election + age + backbencher + offset(log_total_sentences),
  data = speech_counts_appeals
)
R <- 2 * (as.numeric(logLik(nb_model)) - as.numeric(logLik(poisson_model)))
p_value <- 1 - pchisq(R, df = 1)
if (p_value < 0.001) {
  cat("The null hypothesis of equidispersion (Poisson variance = mean) can be rejected at the 0.001 significance level.")
}

# negative binomial model (with CRSEs and multilevel)

# first check baseline differences
youth_nb_baseline <- glm.nb(
  young_positive_total ~ party + under_30 + highly_vulnerable + days_until_election + age + gender + backbencher + offset(log_total_sentences),
  data = speech_counts_appeals
)
youth_nb_baseline_cse <- coeftest(youth_nb_baseline, vcov = vcovCL(youth_nb_baseline, cluster = cluster_var))
youth_nb_baseline_cse

youth_nb_ints <- glm.nb(
  young_positive_total ~ party*under_30 + highly_vulnerable + days_until_election + age + gender + backbencher + offset(log_total_sentences),
  data = speech_counts_appeals
)
youth_nb_ints_cse <- coeftest(youth_nb_ints, vcov = vcovCL(youth_nb_ints, cluster = cluster_var))
youth_nb_ints_cse

# compute predicted counts for interactions via parametric bootstrap
pred_youth_count <- pb_predicted_counts(
  b_iterations = 10,
  regression = youth_nb_ints,
  dv = "youth",
  cluster_var = cluster_var,
  df = speech_counts_appeals,
  constituency_vals = 1:60,
  party_vals = c("Con", "Lab"),
  vulnerability_vals = c(0, 1)
)

# export the predicted counts as a csv
write.csv(pred_youth_count, "analysis_results/pred_counts_youth_appeals.csv", row.names = F)

# visualize predicted counts

# for constituency-share interaction
ggplot(pred_youth_count,
       aes(x = constituency_share, y = pred, color = party, fill = party)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.2, colour = NA)

################################ Elderly Appeals ################################

# likelihood ratio test to choose between Poisson and Negative Binomial
poisson_model <- glm(
  elderly_positive_total ~ party + highly_vulnerable + over_65 + days_until_election + age + backbencher + offset(log_total_sentences),
  family = poisson(link = "log"),
  data = speech_counts_appeals
)
nb_model <- glm.nb(
  elderly_positive_total ~ party + highly_vulnerable + over_65 + days_until_election + age + backbencher + offset(log_total_sentences),
  data = speech_counts_appeals
)
R <- 2 * (as.numeric(logLik(nb_model)) - as.numeric(logLik(poisson_model)))
p_value <- 1 - pchisq(R, df = 1)
if (p_value < 0.001) {
  cat("The null hypothesis of equidispersion (Poisson variance = mean) can be rejected at the 0.001 significance level.")
}

## Negative binomial model (with CRSEs and ML)

# first check baseline differences
elderly_nb_baseline <- glm.nb(
  elderly_positive_total ~ party + highly_vulnerable + over_65 + days_until_election + age + gender + backbencher + offset(log_total_sentences),
  data = speech_counts_appeals
)
elderly_nb_baseline_cse <- coeftest(elderly_nb_baseline, vcov = vcovCL(elderly_nb_baseline, cluster = cluster_var))
elderly_nb_baseline_cse

elderly_nb_ints <- glm.nb(
  elderly_positive_total ~ party*over_65 + highly_vulnerable + days_until_election + age + gender + backbencher + offset(log_total_sentences),
  data = speech_counts_appeals
)
elderly_nb_ints_cse <- coeftest(elderly_nb_ints, vcov = vcovCL(elderly_nb_ints, cluster = cluster_var))
elderly_nb_ints_cse

elderly_nb_ints$aic

# compute predicted counts for interactions via parametric bootstrap
pred_elderly_count <- pb_predicted_counts(
  b_iterations = 1000,
  regression = elderly_nb_ints,
  dv = "elderly",
  cluster_var = cluster_var,
  df = speech_counts_appeals,
  constituency_vals = 1:35,
  party_vals = c("Con", "Lab"),
  vulnerability_vals = c(0, 1)
)

# export the predicted counts as a csv
write.csv(pred_elderly_count, "analysis_results/pred_counts_elderly_appeals.csv", row.names = F)

# visualize predicted counts

# for constituency-share interaction
ggplot(pred_elderly_count,
       aes(x = constituency_share, y = pred, color = party, fill = party)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.2, colour = NA)