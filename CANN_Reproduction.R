library(maidrr)
library(tidyverse)
library(keras)
library(tensorflow)
library(mgcv)

# Load the Belgian Motor Third Party Liability dataset
data('mtpl_be')

#First Look
str(mtpl_be)
summary(mtpl_be)

#### First classical actuarial models ####
#GLM
model_1 <- glm(nclaims ~ coverage + ageph + sex + bm + power + agec +
                 fuel + use + fleet + postcode, 
               family = poisson(), 
               data = mtpl_be, 
               offset = log(expo))
summary(model_1)

#### Neural Network ####
#One-hot encoding for factors
mtpl_be_onehot <- mtpl_be %>%
  select(nclaims, coverage, ageph, sex, bm, power, agec, fuel, use, fleet, postcode) %>%
  mutate(coverage = factor(coverage, ordered = FALSE)) %>%
  data.table::data.table() %>%
  mltools::one_hot() # We went form 13 to 96 variables

n = as.numeric(dim(mtpl_be_onehot)[1])

model_2 <- keras_model_sequential() %>%
  layer_dense(units = 150, 
              activation = "tanh", 
              input_shape = c(n)) %>%
  layer_dense(units = 50, 
              activation = "tanh") %>%
  layer_dense(units = 1, 
              activation = "exponential") %>%
  compile(loss = "poisson", 
          optimize = optimizer_adam())
