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

#### Fist small analysis why CANN, and not just NN ####
# Age
GAM_age <- gam(nclaims ~ s(ageph) + offset(log(expo)), 
               data = mtpl_be, 
               family = poisson(link = "log"))
plot(GAM_age)

# For the NN we use an embedding layer