#######################################
#Amazon Data
#######################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

amazontrain <- vroom("./train.csv")
amazontest <- vroom("./test.csv")
amazontrain <- amazontrain %>%
  mutate(ACTION = as.factor(ACTION))
#########################################
#Visualize Data
#########################################
amazontrain
library(ggmosaic)
ggplot(data =amazontrain) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))

ggplot(data= amazontrain) + geom_boxplot(aes(x = , y=RESOURCE))

ggplot(data= amazontrain) + geom_boxplot(aes(x = , y=ROLE_TITLE))

amazontrain %>%
  summarise(n = n_distinct(ROLE_CODE))

#Should have 112 columns
library(tidymodels)
library(embed)

my_recipe <- recipe(ACTION ~ ., data=amazontrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazontrain)


###########################################################
#Logistic Regression
###########################################################
library(tidymodels)
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

hist(submission$Action)
