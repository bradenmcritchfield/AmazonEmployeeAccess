#######################################################
#This file is to run on the remote server
#######################################################

library(tidyverse)
library(tidymodels)
library(vroom)

amazontrain <- vroom("./train.csv")
amazontest <- vroom("./test.csv")
amazontrain <- amazontrain %>%
  mutate(ACTION = as.factor(ACTION))


########################################################################
# K Nearest Neighbors
########################################################################
library(tidymodels)
my_recipe_K <- recipe(ACTION ~ ., data=amazontrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe_K) %>%
  add_model(knn_model)

## Fit or Tune Model HERE

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazontrain, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('roc_auc') 

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazontrain)


## Predict
KNNpredictions <- predict(final_wf, new_data=amazontest, type="prob")

submission <- KNNpredictions %>%
  mutate(id = amazontest$id) %>%
  mutate(Action = .pred_1) %>%
  select(3, 4)

vroom_write(submission, "amazonKNN.csv", delim = ",")