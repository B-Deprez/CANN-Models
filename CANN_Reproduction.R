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
  layer_dense(units = 100, activation = "tanh", name = 'hidden2') %>%
  layer_dense(units = 50, activation = "tanh", name = 'hidden3') %>%
  layer_dense(units = 25, activation = "tanh", name = 'hidden4') %>%
  layer_dense(units = 1, activation = "linear", name = "Network")

#First you add the two via layer_add
#Then you want an output. Here the second weight is 0 since you want no bias term
Response <- list(Network, LogExpo) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim=c(1,1)), array(0, dim = c(1))))

model_2 <- keras_model(inputs = c(Design, LogExpo), outputs = c(Response))

model_2 %>% compile(optimizer = optimizer_nadam(), 
                    loss = "poisson")

Xfeat <- as.matrix(mtpl_be_onehot)
Xexpo <- as.matrix(log(mtpl_be$expo))
Ylearn <- as.matrix(mtpl_be$nclaims)

fit_2 <- model_2 %>% fit(list(Xfeat, Xexpo), Ylearn, 
                         epochs = 10, 
                         batch_size = 1718,
                         validation_split = 0.2)
fit_2$metrics$loss[10]

#Embedding for factor variables
non_cat <- c(5, 7:9)
n0 <- length(non_cat)

Xlearn <- as.matrix(mtpl_be[,non_cat])
Vlearn <- as.matrix(log(mtpl_be$expo))
CovLearn <- as.matrix(as.integer(mtpl_be$coverage))-1
NCov <- length(unique(CovLearn))
SexLearn <- as.matrix(as.integer(mtpl_be$sex))-1
NSex <- length(unique(SexLearn))
FuelLearn <- as.matrix(as.integer(mtpl_be$fuel))-1
NFuel <- length(unique(FuelLearn))
UseLearn <- as.matrix(as.integer(mtpl_be$use))-1
NUse <- length(unique(UseLearn))
FleetLearn <- as.matrix(as.integer(mtpl_be$fleet))-1
NFleet <- length(unique(FleetLearn))
PcLearn <- as.matrix(as.integer(mtpl_be$postcode))-1
NPc <- length(unique(PcLearn))

Ylearn <- as.matrix(mtpl_be$nclaims)

Design <- layer_input(shape = c(n0), dtype = 'float32', name = 'Design')

LogVol <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

Coverage <- layer_input(shape = c(1), dtype = 'int32', name = 'Coverage')
Sex <- layer_input(shape = c(1), dtype = 'int32', name = 'Sex')
Fuel <- layer_input(shape = c(1), dtype = 'int32', name = 'Fuel')
Usage <- layer_input(shape = c(1), dtype = 'int32', name = 'Usage')
Fleet <- layer_input(shape = c(1), dtype = 'int32', name = 'Fleet')
PostalCode <- layer_input(shape = c(1), dtype = 'int32', name = 'PostalCode')

CovEmb = Coverage %>%
  layer_embedding(input_dim = NCov, output_dim = 1, input_length = 1, name = "CovEmb") %>%
  layer_flatten(name = "Cov_flat") 

SexEmb = Sex %>%
  layer_embedding(input_dim = NSex, output_dim = 1, input_length = 1, name = "SexEmb") %>%
  layer_flatten(name = "Sex_flat") 

FuelEmb = Fuel %>%
  layer_embedding(input_dim = NFuel, output_dim = 1, input_length = 1, name = "FuelEmb") %>%
  layer_flatten(name = "Fuel_flat") 

UsageEmb = Usage %>%
  layer_embedding(input_dim = NUse, output_dim = 1, input_length = 1, name = "UsageEmb") %>%
  layer_flatten(name = "Usage_flat") 

FleetEmb = Fleet %>%
  layer_embedding(input_dim = NFleet, output_dim = 1, input_length = 1, name = "FleetEmb") %>%
  layer_flatten(name = "Fleet_flat") 

PcEmb = PostalCode %>%
  layer_embedding(input_dim = NPc, output_dim = 3, input_length = 1, name = "PcEmb") %>%
  layer_flatten(name = "Pc_flat") 

q1 <- 30   
q2 <- 15
q3 <- 10

Network <- list(Design, CovEmb, SexEmb, FuelEmb, UsageEmb, FleetEmb, PcEmb) %>%
  layer_concatenate(name = 'concate') %>%
  layer_dense(units=q1, activation='tanh', name='hidden1') %>%
  layer_dense(units=q1, activation='tanh', name='hidden2') %>%
  layer_dense(units=q2, activation='tanh', name='hidden3') %>%
  layer_dense(units=q3, activation='tanh', name='hidden4') %>%
  layer_dense(units=1, activation='linear', name='Network')

Response <- list(Network, LogVol) %>% layer_add(name='Add') %>% 
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model_3 <- keras_model(inputs = c(Design, Coverage, Sex, Fuel, Usage, Fleet, PostalCode, LogVol),
                       outputs = c(Response))

model_3 %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')

fit_3 <- model_3 %>% fit(list(Xlearn, CovLearn, SexLearn, FuelLearn, UseLearn, FleetLearn, PcLearn, Vlearn),
                         Ylearn, 
                         epochs = 20, 
                         batch_size = 1718,
                         validation_split = 0.2)
fit_3$metrics$loss[10]
