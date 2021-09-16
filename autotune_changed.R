mfit = model_list
# Copy of standard values
max_ngrps = 15
hcut = 0.75
ignr_intr = NULL
pred_fun = NULL
lambdas = as.vector(outer(seq(1, 10, 0.1), 10^(-7:3)))
nfolds = 5
strat_vars = NULL
glm_par = alist()
err_fun = mse
ncores = -1
out_pds = FALSE
# Specific parameters used here
data = df_model
vars = c('ageph', 'sex')
target = 'nclaims'
pred_fun = CANN_fun
strat_vars = c('nclaims', 'expo')
glm_par = alist(family = poisson(link = 'log'),
                offset = log(expo))
err_fun = poi_dev
ncores = -1

if (sum(grepl('_', vars)) > 0) stop('No underscores allowed in the variable names, these are interpreted as interactions in maidrr.')
if (! all(vars %in% names(data))) stop('All the variables needs to be present in the data.')
if (! target %in% names(data)) stop('The target variable needs to be present in the data.')

if (! ((hcut == -1) | (0 <= hcut & hcut <= 1))) stop('Invalid value specified for hcut, must equal -1 or lie within the range [0, 1].')

if (nfolds <= 1) stop('At least two folds are needed to perform K-fold cross-validation.')

if ((! is.null(strat_vars)) & (! all(strat_vars %in% names(data)))) stop('The stratification variables in strat_vars need to be present in the data.')

if ('formula' %in% names(glm_par)) {
  warning('The field "formula" is removed from glm_par as this is determined by the supplied target and automatic feature selection.')
  glm_par[['formula']] <- NULL
}

if(! all(paste0('y_', c('true', 'pred')) %in% names(formals(err_fun)))) stop('The error function must contain arguments y_true and y_pred.')
if (length(setdiff(names(formals(err_fun)), c('y_true', 'y_pred', 'w_case'))) > 0) stop('The error function can only contain arguments y_true, y_pred and w_case.')
if (('w_case' %in% names(formals(err_fun))) & (! 'weights' %in% names(glm_par))) stop('If w_case is an argument in err_fun, weights must be an argument in glm_par.')

if (ncores < -1 | ncores == 0) stop('The number of cores must be strictly positive, or equal to -1 for all available cores.')
if (ncores > parallel::detectCores()) warning('The asked number of cores is larger than parallel::detectCores, so there might be trouble ahead.')

# Function to calculate validation errors
Kfold_cross_valid <- function(data, target, nfolds, glm_par, err_fun, fixed_terms = NULL){
  val_err <- rep(NA, nfolds)
  for (f in seq_len(nfolds)) {
    # Get training fold
    trn_data <- dplyr::filter(data, fold != f)
    # Feature selection (only keep those with >1 group) and determine the GLM formula
    slct_feat <- names(which(unlist(lapply(trn_data[, grepl('_$', names(trn_data))], function(x) length(unique(x)) > 1))))
    glm_par[['formula']] <- as.formula(paste(target, '~', paste(c(1, fixed_terms, slct_feat), collapse = ' + ')))
    # Fit surrogate GLM on the training fold
    glm_fit <- maidrr::surrogate(trn_data, glm_par)
    # Get validation fold and calculate the prediction error
    val_data <- maidrr::rm_lvls(glm_fit, dplyr::filter(data, fold == f))
    y_pred <- suppressWarnings(predict(glm_fit, newdata = val_data, type = 'response'))
    y_true <- dplyr::pull(val_data, target)
    w_case <- val_data[[deparse(glm_fit$call$weights)]]
    val_err[f] <- ifelse(length(glm_fit$coefficients) > glm_fit$rank, Inf,
                         do.call(err_fun, setNames(lapply(names(formals(err_fun)), function(x) eval(as.name(x))), names(formals(err_fun)))))
  }
  return(val_err)
}

# Create the fold indicator via (stratified) sampling
data <- data %>% dplyr::sample_n(size = nrow(.), replace = FALSE) %>%
  dplyr::arrange(!!! rlang::syms(strat_vars)) %>%
  dplyr::mutate(fold = rep_len(seq_len(nfolds), nrow(.)))

# Generate the feature effects and the lambda grid for main effects

#%#%#%#% Goes wrong here: changed code untill next "#%#%#%#%" 
#--------------------------------------------------------------
interactions = 'user'



#--------------------------------------------------------------
#%#%#%#%

lmbd_main <- maidrr::lambda_grid(fx_vars = fx_main, lambda_range = lambdas, max_ngrps = max_ngrps)

# Set up a cluster and iterate over the lambda grid
switch(as.character(ncores),
       '-1' = doParallel::registerDoParallel(parallel::detectCores(logical = 'FALSE')),
       '1' = foreach::registerDoSEQ(),
       doParallel::registerDoParallel(ncores))
out_main <- foreach::foreach (l = seq_len(nrow(lmbd_main)), .combine = rbind) %dopar% {
  # Segmentation for current lambda values
  data_segm <- maidrr::segmentation(fx_vars = fx_main, data = data, type = 'ngroups',
                                    values =  unlist(dplyr::select(dplyr::slice(lmbd_main, l), vars), use.names = TRUE))
  # Calculate the validation errors
  Kfold_cross_valid(data = data_segm, target = target, nfolds = nfolds, glm_par = glm_par, err_fun = err_fun)
}
doParallel::stopImplicitCluster()

# Get the cross-validation error
if (! ('matrix' %in% class(out_main) & nrow(lmbd_main) > 1)) out_main <- out_main %>% as.matrix() %>% t()
out_main <- dplyr::bind_cols(lmbd_main, out_main %>% as.data.frame %>% setNames(seq_len(nfolds))) %>%
  dplyr::mutate(cv_err = rowMeans(dplyr::select(., as.character(seq_len(nfolds))), na.rm = TRUE)) %>% dplyr::arrange(cv_err)

# Get the selected features
slct_main <- out_main %>% dplyr::slice(1) %>% dplyr::select(!!!rlang::syms(vars)) %>%
  dplyr::select_if(~. > 1) %>% unlist(., use.names = TRUE)
if (length(slct_main) == 0) stop('Not a single feature was selected, please try again with lower values for lambda.')

# Segment the data based on the selected features and optimal number of groups
data_main <- maidrr::segmentation(fx_vars = fx_main[names(slct_main)], data = data, type = 'ngroups', values = slct_main)


# Return these results if no interactions are asked for
if (hcut == -1) {
  # Fit the optimal surrogate GLM
  glm_par[['formula']] <- as.formula(paste(target, '~', paste(c(1, paste0(names(slct_main), '_')), collapse = ' + ')))
  opt_surro <- maidrr::surrogate(data = data_main, par_list = glm_par)
  
  # Combine results in a list and return
  output <- list('slct_feat' = slct_main,
                 'best_surr' = opt_surro,
                 'tune_main' = out_main,
                 'tune_intr' = NULL)
  # Add PD effects if asked for
  if (out_pds) output$pd_fx <- fx_main[names(slct_main)]
  # Output
  return(output)
}


# Generate the feature effects and the lambda grid for interactions
fx_intr <- maidrr::insights(mfit = mfit,
                            vars = setdiff(names(slct_main), ignr_intr),
                            data = data,
                            interactions = 'auto',
                            hcut = hcut,
                            pred_fun = pred_fun,
                            fx_in = fx_main,
                            ncores = ncores)
fx_intr <- fx_intr[grepl('_', names(fx_intr))]
vars_intr <- names(fx_intr)
lmbd_intr <- maidrr::lambda_grid(fx_vars = fx_intr, lambda_range = lambdas, max_ngrps = max_ngrps)

# Set up a cluster and iterate over the lambda grid
switch(as.character(ncores),
       '-1' = doParallel::registerDoParallel(parallel::detectCores(logical = 'FALSE')),
       '1' = foreach::registerDoSEQ(),
       doParallel::registerDoParallel(ncores))
out_intr <- foreach::foreach (l = seq_len(nrow(lmbd_intr)), .combine = rbind) %dopar% {
  # Segmentation for current lambda values
  data_segm <- maidrr::segmentation(fx_vars = fx_intr, data = data_main, type = 'ngroups',
                                    values =  unlist(dplyr::select(dplyr::slice(lmbd_intr, l), vars_intr), use.names = TRUE))
  # Calculate the validation errors
  Kfold_cross_valid(data = data_segm, target = target, nfolds = nfolds, glm_par = glm_par, err_fun = err_fun, fixed_terms = paste0(names(slct_main), '_'))
}
doParallel::stopImplicitCluster()

# Get the cross-validation error
if (! ('matrix' %in% class(out_intr) & nrow(lmbd_intr) > 1)) out_intr <- out_intr %>% as.matrix() %>% t()
out_intr <- dplyr::bind_cols(lmbd_intr, out_intr %>% as.data.frame %>% setNames(seq_len(nfolds))) %>%
  dplyr::mutate(cv_err = rowMeans(dplyr::select(., as.character(seq_len(nfolds))), na.rm = TRUE)) %>% dplyr::arrange(cv_err)
if (all(out_intr$cv_err == Inf)) warning('All interaction combinations lead to rank-deficient GLM fits, try adding larger values for lambda to the grid.')

# Get the selected features
slct_intr <- out_intr %>% dplyr::slice(1) %>% dplyr::select(!!!rlang::syms(vars_intr)) %>%
  dplyr::select_if(~. > 1) %>% unlist(., use.names = TRUE)

# Check if interactions improve the fit with only main effects
if (min(out_intr$cv_err) < min(out_main$cv_err)) slct_feat <- c(slct_main, slct_intr) else slct_feat <- slct_main

# Segment the data based on the selected features and optimal number of groups
data_segm <- maidrr::segmentation(fx_vars = c(fx_main, fx_intr)[names(slct_feat)], data = data, type = 'ngroups', values = slct_feat)

# Fit the optimal surrogate GLM
glm_par[['formula']] <- as.formula(paste(target, '~', paste(c(1, paste0(names(slct_feat), '_')), collapse = ' + ')))
opt_surro <- maidrr::surrogate(data = data_segm, par_list = glm_par)

# Combine results in a list and return
output <- list('slct_feat' = slct_feat,
               'best_surr' = opt_surro,
               'tune_main' = out_main,
               'tune_intr' = out_intr)
# Add PD effects if asked for
if (out_pds) output$pd_fx <- c(fx_main, fx_intr)[names(slct_feat)]
# Output
return(output)