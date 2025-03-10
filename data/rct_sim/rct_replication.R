
# -------------------------------------------------------------------------
# Description: Replicating RCT results 
# Author: Sammy Gold 
# Date Created: March 10th, 2025 
# Logs:
#       3.10.25 - file created 


# -------------------------------------------------------------------------



# Setup -------------------------------------------------------------------
options(scipen = 999)

library(tidyverse)
library(tidymodels)
library(sandwich)
library(lmtest)

wd <- "/Users/sammygold/Documents/GitHub/robust-adaptive-experiments/"
data_fp <- paste0(wd, "data/rct_sim/SDC - Data - Recoded.csv")
treat_labels_fp <- paste0(wd, "data/rct_sim/SDC - Data - Intervention Names.csv")
out_labels_fp <- paste0(wd, "data/rct_sim/SDC - Data - Outcome Names.csv")

# Data Tidying ------------------------------------------------------------
data <- read_csv(data_fp) %>% 
  mutate(Condition = relevel(as.factor(Condition), ref = "Null_Control")) 


# Gathering estimates -----------------------------------------------------
outcomes <- read_csv(out_labels_fp) %>% slice(1:3) # only interested in partisan animosity, undemocratic practices, partisan violence 
plots <- lapply(outcomes %>% pull(Outcome_Name_Data), function(out_var){
  
  # Print outcome 
  cat("Outcome = ", out_var)
  
  # weight for the outcome 
  weight_var <- outcomes %>% filter(Outcome_Name_Data == out_var) %>% pull(Weights_Name_Data)
  
  # make sure outcome isn't missing 
  modelDat <- data %>% 
    filter(!is.na(!!sym(out_var))) 
  
  # run model 
  model <- lm(formula = as.formula(paste0(out_var, " ~ ", "1 + Condition + Gender + Age + Race + Education + Inparty_Person + PI_Pre + Supplier")), 
              data = modelDat, 
              weights = modelDat %>% pull(weight_var))
  
  # tidying results 
  results <- broom::tidy(model, conf.int = TRUE) %>% 
    slice(-1) %>% # removing Intercept term
    filter(str_detect(term, "Condition")) %>% 
    mutate(term = str_remove(term, "Condition")) %>% 
    mutate(across(-term, ~round(.x, 3))) %>% 
    arrange(estimate) %>% 
    inner_join(read_csv(treat_labels_fp), by = c("term" = "Intervention_Name_Data"))
  
  # Plot 
  figure <- ggplot(results, aes(x = estimate, y = reorder(Intervention_Name_Manuscript, desc(estimate)))) + 
    geom_point() + 
    geom_errorbar(aes(xmin = conf.low, xmax = conf.high)) + 
    theme(axis.title.x = element_blank(), 
          axis.title.y = element_blank()) 
  
  return(figure)
  
  
})





