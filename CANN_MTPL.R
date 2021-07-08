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
GAM_age <- gam(nclaims ~ s(ageph) + offset(log(expo)), 
               data = mtpl_be, 
               family = poisson(link = "log"))
plot(GAM_age)

#NN
set.seed(42)
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
                         validation_split = 0.2)

fit_nn_age$metrics$loss[10]
#CANN
set.seed(42)
Vlearn <- as.matrix(log(fitted(GAM_age)))
LogGAM <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogGLM')

Design2 <- layer_input(shape = c(1), dtype = "float32", name = "Design2")

Network2 <- Design2 %>%
  layer_batch_normalization() %>%
  layer_dense(units = 5, activation = 'tanh', name = 'hidden1') %>%
  layer_dense(units = 5, activation = 'tanh', name = 'hidden2') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network2')

Response2 <- list(Network2, LogGAM) %>%
  layer_add(name= "Add") %>%
  layer_dense(units=1, activation=k_exp, name = 'Response2', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

CANN_age <- keras_model(inputs = c(Design2, LogGAM), 
                        outputs = c(Response2))

CANN_age %>% compile(optimizer = optimizer_adam(), 
                     loss = "poisson")

fit_CANN_age <- CANN_age%>% fit(list(Xfeat, Vlearn), Ylearn, 
                                epochs = 25, 
                                batch_size = 1718,
                                validation_split = 0.2)

#Compare GAM and NN+CANN
XTest <- min(mtpl_be$ageph):max(mtpl_be$ageph)
ExpoTest <- rep(1, length(XTest))

y_fit_NN <- predict(NN_age, list(XTest, log(ExpoTest)))

y_fit_GAM <- predict(GAM_age, newdata = data.frame(ageph = XTest, expo = 1), 
                     type = "response")

y_fit_CANN <- predict(CANN_age, list(XTest, log(y_fit_GAM)))

Comparison <- tibble(Age = XTest, GAM = y_fit_GAM, NN = y_fit_NN, CANN = y_fit_CANN)

ggplot(Comparison, aes(x = Age)) +
  geom_line(aes(y= GAM), color = "blue") +
  geom_line(aes(y= NN), color = "red") +
  geom_line(aes(y= CANN), color = "darkgreen")+
  ylab("Response") +
  ggtitle("Difference between GAM and Neural Network")+
  theme_bw()

#PD
y_test_GAM <- GAM_age %>% predict(newdata = mtpl_be, type = "response")

y_test_NN <- NN_age %>% predict(list(mtpl_be$ageph, log(mtpl_be$expo)))

y_test_CANN <- CANN_age %>% predict(list(mtpl_be$ageph, log(y_test_GAM)))

Poisson.Deviance(y_test_GAM, mtpl_be$nclaims)
Poisson.Deviance(y_test_NN, mtpl_be$nclaims)
Poisson.Deviance(y_test_CANN, mtpl_be$nclaims)
#The GAM actually performed a bit better

#### Full modelling of frequency ####
str(mtpl_be)
