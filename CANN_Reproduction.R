library(maidrr)
library(tidyverse)
library(keras)
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

#GAM
model_2 <- gam(nclaims ~ coverage + s(ageph) + sex + s(bm) + s(power) + s(agec) +
                 fuel + use + fleet + postcode, 
               family = poisson(), 
               data = mtpl_be, 
               offset = log(expo))
summary(model_2)

#GBM