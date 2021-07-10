stratify_train_test <- function(df_input, cov, p=0.8, k = 10){
  #Set k and p to have a round amount of bins
  df_to_split <- df_input %>%
    dplyr::arrange(desc(df_input[cov]))
  l = ceiling(nrow(df_to_split)/k)
  for(i in 1:l){
    selected_rows <- df_to_split[((i-1)*k+1):min((i*k),nrow(df_to_split)),]
    permutation <- sample(min(k, nrow(df_to_split)-(i-1)*k))
    if(i == 1){ #First time through loop => initialise 
      df_strat_train<- selected_rows[permutation[1:(k*p)],]
      df_strat_test <- selected_rows[permutation[(k*p+1):length(permutation)],]
    }else if((k*p+1)<=length(permutation)){ 
      df_strat_train <- rbind(df_strat_train, selected_rows[permutation[1:(k*p)],])
      df_strat_test <- rbind(df_strat_test, selected_rows[permutation[(k*p+1):length(permutation)],])
    }else{ #If not enough observations in the end => put all in training set
      df_strat_train <- rbind(df_strat_train, selected_rows)
    }
  }
  return(list(df_strat_train, df_strat_test))
}