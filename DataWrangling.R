#######################################
#Amazon Data
#######################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

amazon <- vroom("./train.csv")

amazon %>% 

