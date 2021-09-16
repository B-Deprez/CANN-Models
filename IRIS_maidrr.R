library(datasets)
library(gbm)
library(maidrr)
library(tidyverse)

data("iris")
summary(iris)

df_iris <- iris %>% mutate(Species = ifelse(Species =="versicolor", 1, 0))

indices <- sample(nrow(df_iris))
df_iris <- df_iris[indices, ]

features <- setdiff(names(df_iris), c('Species'))

gbm_fit <- gbm::gbm(as.formula(paste('Species ~',
                                     paste(features, collapse = ' + '))),
                    distribution = 'bernoulli',
                    data = df_iris,
                    n.trees = 20,
                    interaction.depth = 3,
                    shrinkage = 0.1,
                    cv.folds = 5,
                    train.fraction=0.9
                    )

gbm_fun <- function(object, newdata) mean(predict(object, newdata, n.trees = object$n.trees, type = 'response'))

set.seed(12345)
gbm_maidrr <- gbm_fit %>% autotune(data = df_iris,
                     vars = features,
                     target = 'Species',
                     hcut = 0.75,
                     pred_fun = gbm_fun,
                     lambdas = as.vector(outer(seq(1, 10, 1), 10^(-6:-2))),
                     nfolds = 5,
                     glm_par = alist(family = binomial),
                     err_fun = mse,
                     ncores = -1)

gbm_maidrr
surr_model <- gbm_maidrr$best_surr
