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

load("GAM.RData")


mtpl_be <- maidrr::mtpl_be #the MTPL data set used here

GAM_fun <- function(object, newdata)mean(predict(object, newdata, type = "response"))

GAM_full %>% insights(vars= c('ageph', 'bm', 'coverage', 'fuel', 'sex', 'fleet', 'use', 'postcode'),
                      data = mtpl_be,
                      interactions = "auto",
                      hcut = 0.75,
                      pred_fun = GAM_fun,
                      fx_in = NULL,
                      ncores = -1)



GAM_full %>% autotune(data = mtpl_be,
                      vars = c('ageph', 'bm', 'coverage', 'fuel', 'sex', 'fleet', 'use', 'postcode'),
                      target = 'nclaims',
                      hcut = 0.75,
                      pred_fun = GAM_fun,
                      lambdas = as.vector(outer(seq(1, 10, 1), 10^(-6:-2))),
                      nfolds = 5,
                      strat_vars = c('nclaims', 'expo'),
                      glm_par = alist(family = poisson(link = 'log'),
                                      offset = log(expo)),
                      err_fun = poi_dev,
                      ncores = -1)
