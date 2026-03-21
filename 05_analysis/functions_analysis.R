###################### Functions for Parametric Bootstrap ######################
pb_predicted_counts <- function(b_iterations, regression, dv, cluster_var, df,
                                constituency_vals, party_vals, vulnerability_vals) {
  
  # sample from multivariate normal likelihood function
  beta_hat <- coef(regression)
  vcov_mat <- vcovCL(regression, cluster = cluster_var)
  coef_boot <- mvrnorm(
    n = b_iterations, mu = beta_hat, Sigma = vcov_mat
  )
  
  # arrays to store predicted counts
  pred_counts_constituency <- array(
    NA,
    dim = c(b_iterations, length(party_vals), length(constituency_vals))
  )
  
  # loop over all bootstrap iterations
  for (b in 1:b_iterations) {
    
    # print status
    if (b %% 100 == 0) {
      cat("Iteration:", b, "\n")
    }
    
    # draw bootstrapped beta coefficients
    b_hat_iter <- coef_boot[b, ]
    
    # constituency interaction
    for (p in seq_along(party_vals)) {
      for (c in seq_along(constituency_vals)) {
        
        # construct temporary df with interaction values
        df_temp <- df
        df_temp$party <- factor(
          rep(party_vals[p], nrow(df)),
          levels = levels(df$party)
        )
        if (dv == "youth") {
          df_temp$under_30 <- constituency_vals[c]
        } else {
          df_temp$over_65 <- constituency_vals[c]
        }
        
        # compute count and store in array
        X_temp <- model.matrix(formula(regression), df_temp)
        lambda_hat <- exp(X_temp %*% b_hat_iter)
        pred_counts_constituency[b, p, c] <- mean(lambda_hat)
      }
    }
  }
  
  # summarize across bootstrap iterations
  mean_const <- apply(pred_counts_constituency, c(2,3), mean)
  lower_const <- apply(pred_counts_constituency, c(2,3), quantile, 0.025)
  upper_const <- apply(pred_counts_constituency, c(2,3), quantile, 0.975)
  
  # store bootstrap results in dataframes
  pred_const_df <- expand.grid(
    party = party_vals,
    constituency_share = constituency_vals
  )
  pred_const_df$pred  <- as.vector(mean_const)
  pred_const_df$lower <- as.vector(lower_const)
  pred_const_df$upper <- as.vector(upper_const)
  
  # filter for only observed values
  max_const <- df |>
    group_by(party) |>
    summarise(
      max_share = if (dv == "youth") {
        max(under_30, na.rm = TRUE)
      } else {
        max(over_65, na.rm = TRUE)
      },
      .groups = "drop"
    )
  pred_const_df <- pred_const_df |>
    left_join(max_const, by = "party") |>
    filter(constituency_share <= max_share)
  
  return(pred_const_df)
}

pb_predicted_probs <- function(b_iterations, regression, iv, cluster_var, df,
                               age_appeal_vals, party_vals, age_group_vals) {
  
  # sample from multivariate normal likelihood function
  beta_hat <- coef(regression)
  vcov <- vcovCL(regression, cluster_var)
  coef_boot <- mvrnorm(
    n = b_iterations, mu = beta_hat, Sigma = vcov
  )
  
  # construct empty array to store in predicted probabilities
  pred_probs_array <- array(
    NA,
    dim = c(b_iterations, length(age_appeal_vals), length(party_vals), length(age_group_vals))
  )
  
  # loop over all possible combination of values
  for (b in 1:b_iterations) {
    # print status
    if (b %% 100 == 0) {
      cat("Iteration:", b, "\n")
    }
    b_hat_iter <- coef_boot[b, ]
    for (i in seq_along(age_appeal_vals)) {
      for (p in seq_along(party_vals)) {
        for (a in seq_along(age_group_vals)) {
          
          # store value combination for current iteration in temporary df
          df_temp <- df
          if (iv == "youth") {
            df_temp$share_youth_appeals <- age_appeal_vals[i]
          }
          else (
            df_temp$share_elderly_appeals <- age_appeal_vals[i]
          )
          df_temp$party <- factor(
            rep(party_vals[p], nrow(df)),
            levels = levels(df$party)
          )
          df_temp$age_group <- factor(
            rep(age_group_vals[a], nrow(df)),
            levels = levels(df$age_group)
          )
          
          # compute predicted values and store mean in array
          X_temp <- model.matrix(formula(regression), df_temp)
          p_hat <- plogis(X_temp %*% b_hat_iter)
          pred_probs_array[b, i, p, a] <- mean(p_hat)
        }
      }
    }
  }
  
  # compute mean and confidence intervals across bootstrap iterations
  mean_pred  <- apply(pred_probs_array, c(2,3,4), mean)
  lower_ci   <- apply(pred_probs_array, c(2,3,4), quantile, 0.025)
  upper_ci   <- apply(pred_probs_array, c(2,3,4), quantile, 0.975)
  
  # assign them to a df
  pred_probs_df <- expand.grid(
    age_appeal_share = age_appeal_vals,
    party = party_vals,
    age_group = age_group_vals
  )
  pred_probs_df$pred <- as.vector(mean_pred)
  pred_probs_df$lower <- as.vector(lower_ci)
  pred_probs_df$upper <- as.vector(upper_ci)
  
  # get max shares of age appeals
  max_shares_age_appeals <- df |>
    group_by(party) |>
    summarise(
      max_share = if (iv == "youth") {
        max(share_youth_appeals, na.rm = TRUE)
      } else {
        max(share_elderly_appeals, na.rm = TRUE)
      },
      .groups = "drop"
    )
  # join it to the df and filter for only observed values
  pred_probs_df <- pred_probs_df |>
    left_join(max_shares_age_appeals, by = "party")
  pred_probs_df <- pred_probs_df |>
    filter(age_appeal_share <= max_share)
  
  return(pred_probs_df)
}

diff_in_ame <- function (b_iterations, regression, iv, party_val, cluster_var) {
  
  # sample from multivariate normal likelihood function
  beta_hat <- coef(regression)
  vcov <- vcovCL(regression, cluster_var)
  coef_boot <- mvrnorm(
    n = b_iterations, mu = beta_hat, Sigma = vcov
  )
  
  # create empty vectors to store AME differences
  diff_boot <- numeric(b_iterations)

  # loop over bootstrap iterations
  for (b in 1:b_iterations) {
    
    if (b %% 100 == 0) {
      cat("Iteration:", b, "\n")
    }
    
    # overwrite coefficients
    model_boot <- regression
    model_boot$coefficients <- coef_boot[b, ]
    
    # get AMEs for all interactions
    me <- margins(
      model_boot,
      variables = iv,
      at = list(
        age_group = c("young", "old"),
        party = c(party_val)
      )
    )
    me_df <- summary(me)
    
    # calculate difference between young and old for Labour and Conservatives
    diff_boot[b] <- with(
      subset(me_df, party == party_val),
      if (party_val == "Lab") {
        AME[age_group == "young"] - AME[age_group == "old"]
      }
      else {
        AME[age_group == "old"] - AME[age_group == "young"]
      }
    )
  }
  return(diff_boot)
}
