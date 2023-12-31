library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(themis) 

amazontrain <- vroom("./train.csv")
amazontest <- vroom("./test.csv")
amazontrain <- amazontrain %>%
  mutate(ACTION = as.factor(ACTION))

#########################################################
#Categorical Random Forest
###########################################################
my_recipe_new <- recipe(ACTION ~ ., data=amazontrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # %>%
  #step_normalize(all_numeric_predictors()) %>%
  #step_pca(all_predictors(), threshold = .8) %>%
  #step_smote(all_outcomes(), neighbors=20)

my_mod_RF <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

amazon_workflow_RF <- workflow() %>%
  add_recipe(my_recipe_new) %>%
  add_model(my_mod_RF)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(range = c(20, 40)),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazontrain, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow_RF %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- amazon_workflow_RF %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazontrain)

amazon_predictions_RF <- final_wf %>% predict(new_data = amazontest, type="prob")

submission <- amazon_predictions_RF %>%
  mutate(id = amazontest$id) %>%
  mutate(Action = .pred_1) %>%
  select(3, 4)

vroom_write(submission, "amazonrf.csv", delim = ",")
