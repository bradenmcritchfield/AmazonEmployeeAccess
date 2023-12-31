#######################################################
#This file is to run on the remote server
#######################################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)
library(discrim)
library(naivebayes)
library(embed)

amazontrain <- vroom("./train.csv")
amazontest <- vroom("./test.csv")
amazontrain <- amazontrain %>%
  mutate(ACTION = as.factor(ACTION))

##################################################
#Set recipe and balance data
#####################################################
my_recipe <- recipe(ACTION ~ ., data=amazontrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  #step_pca(all_predictors(), threshold = .9) %>%
  step_smote(all_outcomes(), neighbors=20)

###########################################################
#Logistic Regression
###########################################################
my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")
amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazontrain) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=amazontest,
                              type="prob") # "class" or "prob" (see doc)

submission <- amazon_predictions %>%
  mutate(id = amazontest$id) %>%
  mutate(Action = .pred_1) %>%
  select(3, 4)

vroom_write(submission, "amazonlog.csv", delim = ",")


#############################################################
#Penalized Logistic Regression
#############################################################
my_mod_PLR <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow_PLR <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_PLR)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazontrain, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow_PLR %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- amazon_workflow_PLR %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazontrain)

amazon_predictions_PLR <- final_wf %>% predict(new_data = amazontest, type="prob")

submission <- amazon_predictions_PLR %>%
  mutate(id = amazontest$id) %>%
  mutate(Action = .pred_1) %>%
  select(3, 4)

vroom_write(submission, "amazonlogpr.csv", delim = ",")

#########################################################
#Categorical Random Forest
###########################################################
my_mod_RF <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

amazon_workflow_RF <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_RF)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,(ncol(amazontrain)-1))),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazontrain, v = 2, repeats=1)

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

####################################################
#Naive Bayes
####################################################
## nb model3
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes engine
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(smoothness(),
                            Laplace(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazontrain, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazontrain)


## Predict
nbpredictions <- predict(final_wf, new_data=amazontest, type="prob")

submission <- nbpredictions %>%
  mutate(id = amazontest$id) %>%
  mutate(Action = .pred_1) %>%
  select(3, 4)

vroom_write(submission, "amazonnb.csv", delim = ",")


########################################################################
# K Nearest Neighbors
########################################################################
## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
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


########################################################################
#Support Vector Machines
########################################################################
library(kernlab)
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune())%>%
  set_mode("classification") %>%
  set_engine()

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

## Tune smoothness and Laplace here

## Grid of values to tune over
library(kernlab)
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV15
folds <- vfold_cv(amazontrain, v = 5, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#Find the best tuning parameters
bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazontrain)


## Predict
svmpredictions <- predict(final_wf, new_data=amazontest, type="prob")

submission <- svmpredictions %>%
  mutate(id = amazontest$id) %>%
  mutate(Action = .pred_1) %>%
  select(3, 4)

vroom_write(submission, "amazonsvm.csv", delim = ",")
