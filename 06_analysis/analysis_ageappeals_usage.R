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
data_path <- "../01_data/empirical_analysis/speech_datasets/datasets_cluster_labels/parl_questions_clusterlabels.csv"
questions_df <- read_csv(data_path, col_types = cols(
  date = col_date(),
  party = col_factor(),
  gender = col_factor(),
  birth_date = col_date()))

################################ Preprocessing ################################

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
  filter(lengths(sg_categories) > 0) |> 
  filter(party %in% c("Con", "Lab"))

# aggregate counts per speech
speech_counts <- questions_df |> 
  group_by(speech_id) |> 
  summarise(
    days_until_election = first(days_until_election),
    text = first(text),
    speaker = first(speaker),
    party   = first(party),
    age = first(age),
    gender = first(gender),
    vulnerability = first(vulnerability),
    backbencher = first(backbencher),
    under_30 = first(under_30),
    over_65 = first(over_65),
    young_positive_total = sum(young_positive, na.rm = TRUE),
    elderly_positive_total = sum(elderly_positive, na.rm = TRUE),
    young_positive_bool = ifelse(sum(young_positive, na.rm = TRUE) > 0, 1, 0),
    elderly_positive_bool = ifelse(sum(elderly_positive, na.rm = TRUE) > 0, 1, 0),
    total_sentences = n(),
    log_total_sentences = log(n()),
    .groups = "drop"
  )

################################ Youth Appeals ################################

# specify cluster variable for clustered standard errors
cluster_var <- speech_counts$party

# H1 - baseline party differences

## Logistic regressions (with CRSEs and multilevel)
youth_base_logmod <- glm(
  young_positive_bool ~ party + vulnerability + age + backbencher + gender + under_30,
  family = binomial(link = "logit"),
  data = speech_counts
)
youth_base_logmod_cse <- coeftest(youth_base_logmod, vcov = vcovCL(youth_base_logmod, cluster = cluster_var))
youth_base_logmod_ml <- glmer(
  young_positive_bool ~ party + vulnerability + age + backbencher + gender + under_30 + (1 | speaker),
  family = binomial(link = "logit"),
  data = speech_counts
)

## Zero-inflated regressions (with CRSEs and multilevel)
youth_base_zinb <- zeroinfl(
  young_positive_total ~ party + vulnerability + age + backbencher + gender + under_30 + offset(log_total_sentences) | party + vulnerability,
  data = speech_counts,
  dist = "negbin"
)
youth_base_zinb_cse <- coeftest(youth_base_zinb, vcov = vcovCL(youth_base_zinb, cluster = cluster_var))
youth_base_zinb_ml <- glmmTMB(
  young_positive_total ~ party + vulnerability + age + backbencher + gender + under_30 + offset(log_total_sentences) + (1|speaker),
  ziformula = ~ party + vulnerability,
  family = nbinom2,
  data = speech_counts
)

# check model summaries
summary(youth_base_zinb_ml)



# H2 - Vulnerability

## Logistic regressions (with CRSEs and multilevel)
youth_vulnerability_logmod <- glm(
  young_positive_bool ~ party * vulnerability + age + backbencher + gender + under_30,
  family = binomial(link = "logit"),
  data = speech_counts
)
youth_vulnerability_logmod_cse <- coeftest(youth_vulnerability_logmod, vcov = vcovCL(youth_vulnerability_logmod, cluster = cluster_var))
youth_vulnerability_logmod_ml <- glmer(
  young_positive_bool ~ party * vulnerability + age + backbencher + gender + under_30 + (1 | speaker),
  family = binomial(link = "logit"),
  data = speech_counts
)

## Zero-inflated regressions (with CRSEs and multilevel)
youth_vulnerability_zinb <- zeroinfl(
  young_positive_total ~ party * vulnerability + age + backbencher + gender + under_30 + offset(log_total_sentences) | party + vulnerability,
  data = speech_counts,
  dist = "negbin"
)
youth_vulnerability_zinb_cse <- coeftest(youth_vulnerability_zinb, vcov = vcovCL(youth_vulnerability_zinb, cluster = cluster_var))
youth_vulnerability_zinb_ml <- glmmTMB(
  young_positive_total ~ party * vulnerability + age + backbencher + gender + under_30 + offset(log_total_sentences) + (1|speaker),
  ziformula = ~ party + vulnerability,
  family = nbinom2,
  data = speech_counts
)

# check model summaries
summary(youth_vulnerability_zinb_ml)

# H3 - Composition

## Logistic regressions (with CRSEs and multilevel)
youth_composition_logmod <- glm(
  young_positive_bool ~ party * under_30 + vulnerability + age + backbencher + gender,
  family = binomial(link = "logit"),
  data = speech_counts
)
youth_composition_logmod_cse <- coeftest(youth_composition_logmod, vcov = vcovCL(youth_composition_logmod, cluster = cluster_var))
youth_composition_logmod_ml <- glmer(
  young_positive_bool ~ party * under_30 + vulnerability + age + backbencher + gender + (1 | speaker),
  family = binomial(link = "logit"),
  data = speech_counts
)

## Zero-inflated regressions (with CRSEs and multilevel)
youth_composition_zinb <- zeroinfl(
  young_positive_total ~ party * under_30 + vulnerability + age + backbencher + gender + offset(log_total_sentences) | party + vulnerability,
  data = speech_counts,
  dist = "negbin"
)
youth_composition_zinb_cse <- coeftest(youth_composition_zinb, vcov = vcovCL(youth_composition_zinb, cluster = cluster_var))
youth_composition_zinb_ml <- glmmTMB(
  young_positive_total ~ party * under_30 + vulnerability + age + backbencher + gender + offset(log_total_sentences) + (1|speaker),
  ziformula = ~ party + vulnerability,
  family = nbinom2,
  data = speech_counts
)

# check model summaries
summary(youth_composition_zinb_ml)



################################ Elderly Appeals ################################

# specify cluster variable for clustered standard errors
cluster_var <- speech_counts$party

# H1 - baseline party differences

## Logistic regressions (with CRSEs and multilevel)
elderly_base_logmod <- glm(
  elderly_positive_bool ~ party + vulnerability + age + backbencher + gender + over_65,
  family = binomial(link = "logit"),
  data = speech_counts
)
elderly_base_logmod_cse <- coeftest(elderly_base_logmod, vcov = vcovCL(elderly_base_logmod, cluster = cluster_var))
elderly_base_logmod_ml <- glmer(
  elderly_positive_bool ~ party + vulnerability + age + backbencher + gender + over_65 + (1 | speaker),
  family = binomial(link = "logit"),
  data = speech_counts
)

## Zero-inflated regressions (with CRSEs and multilevel)
elderly_base_zinb <- zeroinfl(
  elderly_positive_total ~ party + vulnerability + age + backbencher + gender + over_65 + offset(log_total_sentences) | party + vulnerability,
  data = speech_counts,
  dist = "negbin"
)
elderly_base_zinb_cse <- coeftest(elderly_base_zinb, vcov = vcovCL(elderly_base_zinb, cluster = cluster_var))
elderly_base_zinb_ml <- glmmTMB(
  elderly_positive_total ~ party + vulnerability + age + backbencher + gender + over_65 + offset(log_total_sentences) + (1|speaker),
  ziformula = ~ party + vulnerability,
  family = nbinom2,
  data = speech_counts
)

# check model summaries
summary(elderly_base_zinb_ml)



# H2 - Vulnerability

## Logistic regressions (with CRSEs and multilevel)
elderly_vulnerability_logmod <- glm(
  elderly_positive_bool ~ party * vulnerability + age + backbencher + gender + over_65,
  family = binomial(link = "logit"),
  data = speech_counts
)
elderly_vulnerability_logmod_cse <- coeftest(elderly_vulnerability_logmod, vcov = vcovCL(elderly_vulnerability_logmod, cluster = cluster_var))
elderly_vulnerability_logmod_ml <- glmer(
  elderly_positive_bool ~ party * vulnerability + age + backbencher + gender + over_65 + (1 | speaker),
  family = binomial(link = "logit"),
  data = speech_counts
)

## Zero-inflated regressions (with CRSEs and multilevel)
elderly_vulnerability_zinb <- zeroinfl(
  elderly_positive_total ~ party * vulnerability + age + backbencher + gender + over_65 + offset(log_total_sentences) | party + vulnerability,
  data = speech_counts,
  dist = "negbin"
)
elderly_vulnerability_zinb_cse <- coeftest(elderly_vulnerability_zinb, vcov = vcovCL(elderly_vulnerability_zinb, cluster = cluster_var))
elderly_vulnerability_zinb_ml <- glmmTMB(
  elderly_positive_total ~ party * vulnerability + age + backbencher + gender + over_65 + offset(log_total_sentences) + (1|speaker),
  ziformula = ~ party + vulnerability,
  family = nbinom2,
  data = speech_counts
)

# check model summaries
summary(elderly_vulnerability_zinb_ml)

# H3 - Composition

## Logistic regressions (with CRSEs and multilevel)
elderly_composition_logmod <- glm(
  elderly_positive_bool ~ party * over_65 + vulnerability + age + backbencher + gender,
  family = binomial(link = "logit"),
  data = speech_counts
)
elderly_composition_logmod_cse <- coeftest(elderly_composition_logmod, vcov = vcovCL(elderly_composition_logmod, cluster = cluster_var))
elderly_composition_logmod_ml <- glmer(
  elderly_positive_bool ~ party * over_65 + vulnerability + age + backbencher + gender + (1 | speaker),
  family = binomial(link = "logit"),
  data = speech_counts
)

## Zero-inflated regressions (with CRSEs and multilevel)
elderly_composition_zinb <- zeroinfl(
  elderly_positive_total ~ party * over_65 + vulnerability + age + backbencher + gender + offset(log_total_sentences) | party + vulnerability,
  data = speech_counts,
  dist = "negbin"
)
elderly_composition_zinb_cse <- coeftest(elderly_composition_zinb, vcov = vcovCL(elderly_composition_zinb, cluster = cluster_var))
elderly_composition_zinb_ml <- glmmTMB(
  elderly_positive_total ~ party * over_65 + vulnerability + age + backbencher + gender + offset(log_total_sentences) + (1|speaker),
  ziformula = ~ party + vulnerability,
  family = nbinom2,
  data = speech_counts
)

# check model summaries
summary(elderly_composition_zinb_ml)