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

n = as.numeric(dim(mtpl_be_onehot)[2])

Design <- layer_input(shape = c(n), dtype = 'float32', name = 'Design')
LogExpo <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

Network <- Design %>%
  layer_dense(units = 100, activation = "tanh", name = 'hidden1') %>%
  layer_dense(units = 100, activation = "tanh", name = 'hidden1') %>%
  layer_dense(units = 50, activation = "tanh", name = 'hidden2') %>%
  layer_dense(units = 25, activation = "tanh", name = 'hidden3') %>%
  layer_dense(units = 1, activation = "linear", name = "Network")

Response <- list(Network, LogExpo) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim=c(1,1)), array(0, dim = c(1))))

model_2 <- keras_model(inputs = c(Design, LogExpo), outputs = c(Response))

model_2 %>% compile(optimizer = optimizer_nadam(), 
                    loss = "poisson")

Xfeat <- as.matrix(mtpl_be_onehot)
Xexpo <- as.matrix(mtpl_be$expo)
Ylearn <- as.matrix(mtpl_be$nclaims)

fit_2 <- model_2 %>% fit(list(Xfeat, Xexpo), Ylearn, 
                         epochs = 10, 
                         batch_size = 1718,
                         validation_split = 0.2)

#Embedding for factor variables
non_cat <- c(5, 7:9)
n0 <- length(non_cat)

Xlearn <- as.matrix(mtpl_be[,non_cat])
