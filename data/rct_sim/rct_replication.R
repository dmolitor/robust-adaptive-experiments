
# -------------------------------------------------------------------------
# Description: Replicating RCT results 
# Author: Sammy Gold 
# Date Created: March 10th, 2025 
# Logs:
#       3.10.25 - file created 


# -------------------------------------------------------------------------



# Setup -------------------------------------------------------------------
options(scipen = 999)

library(here)
library(tidyverse)
library(tidymodels)
library(sandwich)
library(lmtest)

wd <- here()
data_fp <- here(wd, "data/rct_sim/SDC - Data - Recoded.csv")
treat_labels_fp <- here(wd, "data/rct_sim/SDC - Data - Intervention Names.csv")
out_labels_fp <- here(wd, "data/rct_sim/SDC - Data - Outcome Names.csv")

# Data Tidying ------------------------------------------------------------
data <- read_csv(data_fp) |> 
  mutate(Condition = relevel(as.factor(Condition), ref = "Null_Control")) 


# Gathering estimates -----------------------------------------------------
outcomes <- read_csv(out_labels_fp) |> slice(1:3) # only interested in partisan animosity, undemocratic practices, partisan violence 
plots <- lapply(outcomes |> pull(Outcome_Name_Data), function(out_var){
  # Print outcome 
  cat("Outcome = ", out_var)
  
  # weight for the outcome 
  weight_var <- outcomes |> filter(Outcome_Name_Data == out_var) |> pull(Weights_Name_Data)
  outcome_name <- outcomes |> filter(Outcome_Name_Data == out_var) |> pull(Outcome_Name_Manuscript)
  
  # make sure outcome isn't missing 
  modelDat <- data |> 
    filter(!is.na(!!sym(out_var)))
  
  # run model 
  model_weighted <- lm(
    formula = as.formula(paste0(out_var, " ~ ", "1 + Condition + Gender + Age + Race + Education + Inparty_Person + PI_Pre + Supplier")), 
    data = modelDat, 
    weights = modelDat |> pull(weight_var)
  )
  model_unweighted <- lm(
    formula = as.formula(paste0(out_var, " ~ ", "1 + Condition + Gender + Age + Race + Education + Inparty_Person + PI_Pre + Supplier")), 
    data = modelDat
  )
  
  # tidying results 
  results_weighted <- broom::tidy(model_weighted, conf.int = TRUE) |> 
    slice(-1) |> # removing Intercept term
    filter(str_detect(term, "Condition")) |> 
    mutate(term = str_remove(term, "Condition")) |> 
    mutate(across(-term, ~round(.x, 3))) |> 
    arrange(estimate) |> 
    inner_join(read_csv(treat_labels_fp), by = c("term" = "Intervention_Name_Data")) |>
    mutate(type = "weighted")

  results_unweighted <- broom::tidy(model_unweighted, conf.int = TRUE) |> 
    slice(-1) |> # removing Intercept term
    filter(str_detect(term, "Condition")) |> 
    mutate(term = str_remove(term, "Condition")) |> 
    mutate(across(-term, ~round(.x, 3))) |> 
    arrange(estimate) |> 
    inner_join(read_csv(treat_labels_fp), by = c("term" = "Intervention_Name_Data")) |>
    mutate(type = "unweighted")

  results <- bind_rows(results_weighted, results_unweighted)
  
  # Plot 
  figure <- ggplot(
    results,
    aes(
      x = estimate,
      y = reorder(Intervention_Name_Manuscript, desc(estimate)),
      color = type
    )
  ) + 
    geom_point(position = position_dodge(width = 0.3)) + 
    geom_linerange(aes(xmin = conf.low, xmax = conf.high), position = position_dodge(width = 0.3)) + 
    theme(axis.title.x = element_blank(), 
          axis.title.y = element_blank()) +
    theme_minimal() +
    geom_vline(xintercept = 0, color = "gray70") +
    labs(x = "Estimate", y = "Intervention", title = paste("Outcome:", outcome_name))
  
  return(figure)
})





