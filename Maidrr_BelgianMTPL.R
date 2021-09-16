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


model_CANN <- load_model_tf("CANN_3Layers")
mtpl_be <- maidrr::mtpl_be #the MTPL data set used here


df_model<- mtpl_be %>%
  mutate(
    coverage = as.integer(coverage)-1,
    sex = as.integer(sex)-1,
    fuel = as.integer(fuel)-1,
    use = as.integer(use)-1,
    fleet = as.integer(fleet)-1,
    postcode = as.integer(postcode)-1
    )

#GAM_full <- gam(nclaims ~ as.factor(coverage) + s(ageph) + as.factor(sex) + s(bm) + s(power) + 
#      s(agec) + as.factor(fuel) + as.factor(use) + as.factor(fleet) + as.factor(postcode), 
#    data = df_model, 
#    offset = log(expo), 
#    family = poisson(link = "log"))

GAM_full <- readRDS("GAM_CANN.rds")

CANN_fun <- function(object, newdata){
  non_cat <- c(5, 7:9)
  VTest <- as.matrix(mgcv::predict.gam(object[[1]], newdata = newdata) + log(newdata$expo))
  df_CANN <- list(
    as.matrix(newdata[,non_cat]), 
    as.matrix(newdata$coverage), 
    as.matrix(newdata$sex), 
    as.matrix(newdata$fuel), 
    as.matrix(newdata$use), 
    as.matrix(newdata$fleet), 
    as.matrix(newdata$postcode), 
    VTest
    )
  mean(predict(object[[2]], df_CANN))
}

model_list = list(GAM_full, model_CANN)

maidrr_model_data <- model_list %>% autotune(data = df_model,
                                             vars = c('ageph', 'sex'),
                                             target = 'nclaims',
                                             pred_fun = CANN_fun,
                                             strat_vars = c('nclaims', 'expo'),
                                             glm_par = alist(family = poisson(link = 'log'),
                                                             offset = log(expo)),
                                             err_fun = poi_dev,
                                             ncores = -1)



model_pd <- model_list %>% get_pd(var = "sex_ageph", 
                      grid = c("sex", "ageph")%>%get_grid(df_model),
                      data = df_model,
                      subsample = 1000,
                      fun = CANN_fun)

library(gbm)
data('mtpl_be')
features <- setdiff(names(mtpl_be), c('id', 'nclaims', 'expo', 'long', 'lat'))
set.seed(12345)
gbm_fit <- gbm::gbm(as.formula(paste('nclaims ~',
                                     paste(features, collapse = ' + '))),
                    distribution = 'poisson',
                    data = mtpl_be,
                    n.trees = 50,
                    interaction.depth = 3,
                    shrinkage = 0.1)

gbm_fun <- function(object, newdata) mean(predict(object, newdata, n.trees = object$n.trees, type = 'response'))
gbm_fit %>% autotune(data = mtpl_be,
                     vars = c('ageph', 'bm', 'coverage', 'fuel', 'sex', 'fleet', 'use'),
                     target = 'nclaims',
                     hcut = 0.75,
                     pred_fun = gbm_fun,
                     lambdas = as.vector(outer(seq(1, 10, 1), 10^(-6:-2))),
                     nfolds = 5,
                     strat_vars = c('nclaims', 'expo'),
                     glm_par = alist(family = poisson(link = 'log'),
                                     offset = log(expo)),
                     err_fun = poi_dev,
                     ncores = -1)
