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

pd_df <- model_list %>% 
  get_pd(var = "sex_ageph", 
                      grid = c("sex", "ageph")%>%get_grid(df_model),
                      data = df_model,
                      subsample = 1000,
                      fun = CANN_fun)

pd_df %>%
  mutate(x1 = ifelse(x1 == 0, "female", "male")) %>%
  rename(gender = x1, age = x2) %>%
  ggplot(aes(age, y, color = gender, group = gender))+
  geom_line()+
  theme_bw()

pd_age <- model_list %>%
  get_pd(var = "ageph", 
         grid = c("ageph")%>%get_grid(df_model),
         data = df_model,
         subsample = 1000,
         fun = CANN_fun)

age_levels <- c(rep("[18, 19]",2),
                rep("[20, 21]",2),
                rep("[22, 23]",2),
                rep("[24, 25]",2),
                rep("[26, 28]",3),
                rep("[29, 32]",4),
                rep("[33, 50]",18),
                rep("[51, 55]",5),
                rep("[56, 60]",5),
                rep("[61, 76]",16),
                rep("[77, 81]",5),
                rep("[82, 85]",4),
                rep("[86, 90]",5),
                rep("[91, 95]",5))

final_df <- pd_age %>%
  dplyr::select(x,y) %>%
  rename(age = x, pd = y) %>%
  mutate(glm = predict(output$best_surr, newdata = data.frame(ageph_=age_levels, expo = 1), type = "response"))

ggplot(final_df, aes(x=age))+
  geom_line(aes(y=glm), color = "blue")+
  theme_bw()
