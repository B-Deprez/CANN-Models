#### Setup ####

# Libraries needed
library(tidyverse)
library(readxl)
library(MASS)
library(readxl)
library(maidrr)
library(sp)
library(sf)
library(tmap)
library(mapview)
library(mgcv)

library(keras)

# Import all data needed
mtpl_be <- maidrr::mtpl_be #the MTPL data set used here
postal_codes <- read_excel("inspost.xls")

belgium_shape_sf <- st_read('./shape file Belgie postcodes/npc96_region_Project1.shp', quiet = TRUE)
belgium_shape_sf <- st_transform(belgium_shape_sf, CRS("+proj=longlat +datum=WGS84"))

# Poisson Deviance for comparison
Poisson.Deviance <- function(pred, obs){200*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)}

#### Fist small analysis why CANN, and not just NN ####
### Age
#GLM
GAM_age <- gam(nclaims ~ s(ageph), 
               data = mtpl_be, 
               offset = log(expo),
               family = poisson(link = "log"))
plot(GAM_age)

#NN
set.seed(1997)
Design <- layer_input(shape = c(1), dtype = "float32", name = "Design")
LogExpo <- layer_input(shape = c(1), dtype = "float32", name = "LogExpo")

Network <- Design %>%
  layer_batch_normalization(input_shape = c(1)) %>%
  layer_dense(units = 5, activation = 'tanh', name = 'hidden1') %>%
  layer_dense(units = 5, activation = 'tanh', name = 'hidden2') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network')

Response <- list(Network, LogExpo) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim=c(1,1)), array(0, dim = c(1))))

NN_age <- keras_model(inputs = c(Design, LogExpo), outputs = c(Response))

NN_age %>% compile(optimizer = optimizer_adam(), 
                    loss = "poisson")

Xfeat <- as.matrix(mtpl_be$ageph)
Xexpo <- as.matrix(log(mtpl_be$expo))
Ylearn <- as.matrix(mtpl_be$nclaims)

fit_nn_age <- NN_age %>% fit(list(Xfeat, Xexpo), Ylearn, 
                         epochs = 10, 
                         batch_size = 1718,
                         validation_split = 0.2, 
                         verbose = 0)

fit_nn_age$metrics$loss[10]
#CANN
set.seed(1997)
Vlearn2 <- as.matrix(predict(GAM_age) + log(mtpl_be$expo)) #You include exposure
LogGAM <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogGAM')

Design2 <- layer_input(shape = c(1), dtype = "float32", name = "Design2")

Network2 <- Design2 %>%
  layer_batch_normalization() %>%
  layer_dense(units = 5, activation = 'tanh', name = 'hidden2_1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network2')

Response2 <- list(Network2, LogGAM) %>%
  layer_add(name= "Add2") %>%
  layer_dense(units=1, activation="exponential", name = 'Response2', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

CANN_age <- keras_model(inputs = c(Design2, LogGAM), 
                        outputs = c(Response2))

CANN_age %>% compile(optimizer = optimizer_rmsprop(), 
                     loss = "poisson")

CANN_age%>% fit(list("Design2" = Xfeat, "LogGAM"=Vlearn2), Ylearn, 
                                epochs = 25, 
                                batch_size = 1718, 
                                verbose = 0)

#Compare GAM and NN+CANN
#Add all factor glm to it
factor_age <- glm(nclaims ~ factor(ageph)-1, offset = log(expo), data = mtpl_be, family = poisson(link="log"))
summary(factor_age)

XTest <- (min(mtpl_be$ageph)+1):max(mtpl_be$ageph)
ExpoTest <- rep(1, length(XTest))

y_fit_factor <- predict(factor_age, newdata = data.frame(ageph = XTest, expo = 1), 
                        type = "response")

y_fit_NN <- predict(NN_age, list(XTest, log(ExpoTest)))

y_fit_GAM <- predict(GAM_age, newdata = data.frame(ageph = XTest, expo = 1), 
                     type = "response")

y_fit_CANN <- predict(CANN_age, list(XTest, 
                                     predict(GAM_age, 
                                                    newdata = data.frame(ageph = XTest, expo = 1))))

Comparison <- tibble(Age = XTest, GAM = y_fit_GAM, NN = y_fit_NN, CANN = y_fit_CANN, factors = y_fit_factor)

ggplot(Comparison, aes(x = Age)) +
  geom_line(aes(y= GAM), color = "blue") +
  geom_line(aes(y= NN), color = "red") +
  geom_line(aes(y= CANN), color = "darkgreen")+
  geom_line(aes(y= factors))+
  ylab("Response") +
  ggtitle("Difference between GAM, Neural Network and CANN")+
  theme_bw()

#PD
y_test_factor <- factor_age %>% predict(newdata = mtpl_be, type = "response")

y_test_GAM <- GAM_age %>% predict(newdata = mtpl_be, type = "response")

y_test_NN <- NN_age %>% predict(list(mtpl_be$ageph, log(mtpl_be$expo)))

y_test_CANN <- CANN_age %>% predict(list(mtpl_be$ageph, log(y_test_GAM)))

Poisson.Deviance(y_test_factor, mtpl_be$nclaims)
Poisson.Deviance(y_test_GAM, mtpl_be$nclaims)
Poisson.Deviance(y_test_NN, mtpl_be$nclaims)
Poisson.Deviance(y_test_CANN, mtpl_be$nclaims)

#The GAM actually performed a bit better

#### Full modelling of frequency ####
str(mtpl_be)

## Plotting frequency for Belgium
post_claim <- mtpl_be %>% group_by(postcode) %>% summarize(num = n(), 
                                                            mean_claim = mean(nclaims))
belgium_shape_sf_grouped <- belgium_shape_sf %>% 
  mutate(codpos = as.factor(floor(POSTCODE/100))) %>%
  group_by(codpos) %>%
  summarise(geometry = st_union(geometry),
            Shape_Area = sum(Shape_Area)) %>%
  left_join(post_claim, 
            by = c("codpos" = "postcode"))

belgium_shape_sf_grouped$freq <- 
  belgium_shape_sf_grouped$mean_claim/belgium_shape_sf_grouped$Shape_Area
belgium_shape_sf_grouped$freq_class <- cut(belgium_shape_sf_grouped$freq, 
                                   breaks = quantile(belgium_shape_sf_grouped$freq, 
                                                     c(0,0.20, 0.4,0.6,0.8,1), 
                                                     na.rm = TRUE),
                                   right = FALSE, include.lowest = TRUE, 
                                   labels = c("low", "average-low", "average", 
                                              "average-high", "high"))
map_freq <- ggplot(belgium_shape_sf_grouped) +
  geom_sf(aes(fill = freq_class), size = 0) +
  ggtitle("Claim frequency data") +
  scale_fill_brewer(palette = "Blues", na.value = "white") + 
  labs(fill = "Relative\nclaim fequency") +
  theme_bw();map_freq

## Train-test split
source('./stratify_train_test.R')
stratified_train_test_split <- stratify_train_test(mtpl_be, c("nclaims"))
mtpl_train <- stratified_train_test_split[[1]]
mtpl_test <- stratified_train_test_split[[2]]

## GAM
GAM_full <- gam(nclaims ~ coverage + s(ageph) + sex + s(bm) + s(power) + 
                  s(agec) + fuel + use + fleet + postcode, 
                data = mtpl_train, 
                offset = log(expo), 
                family = poisson(link = "log"))

par(mfrow = c(2,2))
plot(GAM_full, scale = 0)

# Results from the GAM are our benchmark
y_test_fullGAM <- GAM_full %>% predict(newdata = mtpl_test, type = "response")
Poisson.Deviance(y_test_fullGAM, mtpl_test$nclaims)

## CANN
# We use embeddings for the factor varaibles
non_cat <- c(5, 7:9)
n0 <- length(non_cat)

Xlearn <- as.matrix(mtpl_train[,non_cat])
Vlearn <- as.matrix(predict(GAM_full) + log(mtpl_train$expo))

CovLearn <- as.matrix(as.integer(mtpl_train$coverage))-1
NCov <- length(unique(CovLearn))
SexLearn <- as.matrix(as.integer(mtpl_train$sex))-1
NSex <- length(unique(SexLearn))
FuelLearn <- as.matrix(as.integer(mtpl_train$fuel))-1
NFuel <- length(unique(FuelLearn))
UseLearn <- as.matrix(as.integer(mtpl_train$use))-1
NUse <- length(unique(UseLearn))
FleetLearn <- as.matrix(as.integer(mtpl_train$fleet))-1
NFleet <- length(unique(FleetLearn))
PcLearn <- as.matrix(as.integer(mtpl_train$postcode))-1
NPc <- length(unique(PcLearn))

XTest <- as.matrix(mtpl_test[,non_cat])
VTest <- as.matrix(predict(GAM_full, newdata = mtpl_test) + log(mtpl_test$expo))

CovTest <- as.matrix(as.integer(mtpl_test$coverage))-1
SexTest <- as.matrix(as.integer(mtpl_test$sex))-1
FuelTest <- as.matrix(as.integer(mtpl_test$fuel))-1
UseTest <- as.matrix(as.integer(mtpl_test$use))-1
FleetTest <- as.matrix(as.integer(mtpl_test$fleet))-1
PcTest <- as.matrix(as.integer(mtpl_test$postcode))-1

Ylearn <- as.matrix(mtpl_train$nclaims)

Design <- layer_input(shape = c(n0), dtype = 'float32', name = 'Design')

LogGAM <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogGAM')

Coverage <- layer_input(shape = c(1), dtype = 'int32', name = 'Coverage')
Sex <- layer_input(shape = c(1), dtype = 'int32', name = 'Sex')
Fuel <- layer_input(shape = c(1), dtype = 'int32', name = 'Fuel')
Usage <- layer_input(shape = c(1), dtype = 'int32', name = 'Usage')
Fleet <- layer_input(shape = c(1), dtype = 'int32', name = 'Fleet')
PostalCode <- layer_input(shape = c(1), dtype = 'int32', name = 'PostalCode')

num_layers <- c(1,3)
num_neurons <- c(5,10,15)
activ <- c("relu", "tanh")

grid_nn <- expand_grid(num_neurons, activ)

## With 1 layer

Tune_1 <- tibble(emb_dim = integer(), 
                 l_1 = integer(), 
                 pois_dev = numeric())

for(i in 1:2){ #Embedding dimensions
  # We repeat all layers, so they are initialised, and training doesn't start
  # in the optimal position of previous one
  CovEmb = Coverage %>%
    layer_embedding(input_dim = NCov, output_dim = 2, input_length = 1, name = "CovEmb") %>%
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
    layer_embedding(input_dim = NPc, output_dim = i, input_length = 1, name = "PcEmb") %>%
    layer_flatten(name = "Pc_flat") 
  
  for(l_1 in 1:(dim(grid_nn)[1])){
    Network <- list(Design, CovEmb, SexEmb, FuelEmb, UsageEmb, FleetEmb, PcEmb) %>%
      layer_concatenate(name = 'concate') %>%
      layer_batch_normalization() %>%
      layer_dense(units=as.numeric(grid_nn[l_1, 1]), 
                  activation=as.character(grid_nn[l_1,2]), 
                  name='hidden1')%>%
      layer_dense(units=1, activation='linear', name='Network')
    
    Response <- list(Network, LogGAM) %>% layer_add(name='Add') %>% 
      layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
                  weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))
    
    model_tune <- keras_model(inputs = c(Design, Coverage, Sex, Fuel, Usage, Fleet, PostalCode, LogGAM),
                         outputs = c(Response))
    
    model_tune %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')
    
    fit_tune <- model_tune %>% fit(list(Xlearn, CovLearn, SexLearn, FuelLearn, UseLearn, FleetLearn, PcLearn, Vlearn),
                    Ylearn, 
                    epochs = 20, 
                    batch_size = 1718, 
                    verbose = 0)
    
    y_test <- model_tune %>% predict(list(XTest, CovTest, SexTest, FuelTest, UseTest, FleetTest, PcTest, VTest))
    pois_dev <- Poisson.Deviance(y_test, mtpl_test$nclaims)
    print(pois_dev)
    Tune_1 <- Tune_1 %>% add_row(emb_dim = i, l_1 = l_1,pois_dev =pois_dev)
  }
}

Result1 <- Tune_1 %>% 
  filter(pois_dev == min(pois_dev)) %>%
  mutate(number_neurons = as.numeric(grid_nn[l_1, 1]),
         activation = as.character(grid_nn[l_1, 2])); Result1


# 3 hidden layers
Tune_3 <- tibble(emb_dim = integer(), 
                 l_1 = integer(), 
                 l_2 = integer(), 
                 l_3 = integer(), 
                 pois_dev = numeric())

for(i in 1:2){ #Embedding dimensions
  # We repeat all layers, so they are initialised, and training doesn't start
  # in the optimal position of previous one
  CovEmb = Coverage %>%
    layer_embedding(input_dim = NCov, output_dim = 2, input_length = 1, name = "CovEmb") %>%
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
    layer_embedding(input_dim = NPc, output_dim = i, input_length = 1, name = "PcEmb") %>%
    layer_flatten(name = "Pc_flat") 
  
  for(l_1 in 1:(length(num_neurons))){
    for(l_2 in 1:(length(num_neurons))){
      for(l_3 in 1:(length(num_neurons))){
        Network <- list(Design, CovEmb, SexEmb, FuelEmb, UsageEmb, FleetEmb, PcEmb) %>%
          layer_concatenate(name = 'concate') %>%
          layer_batch_normalization() %>%
          layer_dense(units=as.numeric(num_neurons[l_1]), 
                      activation="relu", 
                      name='hidden1')%>%
          layer_dense(units=as.numeric(num_neurons[l_2]), 
                      activation="relu", 
                      name='hidden2')%>%
          layer_dense(units=as.numeric(num_neurons[l_3]), 
                      activation="relu", 
                      name='hidden3')%>%
          layer_dense(units=1, activation='linear', name='Network')
        
        Response <- list(Network, LogGAM) %>% layer_add(name='Add') %>% 
          layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
                      weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))
        
        model_tune <- keras_model(inputs = c(Design, Coverage, Sex, Fuel, Usage, Fleet, PostalCode, LogGAM),
                                  outputs = c(Response))
        
        model_tune %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')
        
        fit_tune <- model_tune %>% fit(list(Xlearn, CovLearn, SexLearn, FuelLearn, UseLearn, FleetLearn, PcLearn, Vlearn),
                                       Ylearn, 
                                       epochs = 20, 
                                       batch_size = 1718, 
                                       verbose = 0)
        
        y_test <- model_tune %>% predict(list(XTest, CovTest, SexTest, FuelTest, UseTest, FleetTest, PcTest, VTest))
        pois_dev <- Poisson.Deviance(y_test, mtpl_test$nclaims)
        print(pois_dev)
        Tune_3 <- Tune_3 %>% add_row(emb_dim = i, 
                                     l_1 = l_1,
                                     l_2 = l_2, 
                                     l_3 = l_3, 
                                     pois_dev = pois_dev)
      }
    }
  }
}

Result3 <- Tune_3 %>% 
  filter(pois_dev == min(pois_dev)) %>%
  mutate(number_neurons1 = as.numeric(grid_nn[l_1, 1]),
         number_neurons2 = as.numeric(grid_nn[l_2, 1]),
         number_neurons3 = as.numeric(grid_nn[l_3, 1])); Result3

Result1$pois_dev
Result3$pois_dev


#### Total number of claims predicted ####
#First retrain the models with the resulting hyperparameters

## One layer network
CovEmb = Coverage %>%
  layer_embedding(input_dim = NCov, output_dim = 2, input_length = 1, name = "CovEmb") %>%
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
  layer_embedding(input_dim = NPc, output_dim = 2, input_length = 1, name = "PcEmb") %>%
  layer_flatten(name = "Pc_flat") 

Network <- list(Design, CovEmb, SexEmb, FuelEmb, UsageEmb, FleetEmb, PcEmb) %>%
  layer_concatenate(name = 'concate') %>%
  layer_batch_normalization() %>%
  layer_dense(units=5, 
              activation="relu", 
              name='hidden1')%>%
  layer_dense(units=1, activation='linear', name='Network')

Response <- list(Network, LogGAM) %>% layer_add(name='Add') %>% 
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model_1 <- keras_model(inputs = c(Design, Coverage, Sex, Fuel, Usage, Fleet, PostalCode, LogGAM),
                          outputs = c(Response))

model_1 %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')

fit_1 <- model_1 %>% fit(list(Xlearn, CovLearn, SexLearn, FuelLearn, UseLearn, FleetLearn, PcLearn, Vlearn),
                               Ylearn, 
                               epochs = 20, 
                               batch_size = 1718)

y_pred_1 <- model_1 %>% predict(list(XTest, CovTest, SexTest, FuelTest, UseTest, FleetTest, PcTest, VTest))
Poisson.Deviance(y_pred_1, mtpl_test$nclaims)

#Three layer network
Network <- list(Design, CovEmb, SexEmb, FuelEmb, UsageEmb, FleetEmb, PcEmb) %>%
  layer_concatenate(name = 'concate') %>%
  layer_batch_normalization() %>%
  layer_dense(units=as.numeric(num_neurons[l_1]), 
              activation="relu", 
              name='hidden1')%>%
  layer_dense(units=as.numeric(num_neurons[l_2]), 
              activation="relu", 
              name='hidden2')%>%
  layer_dense(units=as.numeric(num_neurons[l_3]), 
              activation="relu", 
              name='hidden3')%>%
  layer_dense(units=1, activation='linear', name='Network')

Response <- list(Network, LogGAM) %>% layer_add(name='Add') %>% 
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model_3 <- keras_model(inputs = c(Design, Coverage, Sex, Fuel, Usage, Fleet, PostalCode, LogGAM),
                          outputs = c(Response))

model_3 %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')

fit_3 <- model_3 %>% fit(list(Xlearn, CovLearn, SexLearn, FuelLearn, UseLearn, FleetLearn, PcLearn, Vlearn),
                               Ylearn, 
                               epochs = 25, 
                               batch_size = 1718)

y_pred_3 <- model_3 %>% predict(list(XTest, CovTest, SexTest, FuelTest, UseTest, FleetTest, PcTest, VTest))
Poisson.Deviance(y_pred_3, mtpl_test$nclaims)

# Total claims predicted
Total_claims <- sum(mtpl_test$nclaims)

Total_GAM <- sum(GAM_full %>% 
  predict(newdata= mtpl_train, type = "response"))

Total_1 <- sum(y_pred_1)

Total_3 <- sum(y_pred_3)

tibble(Total_claims, Total_GAM, Total_1, Total_3)
