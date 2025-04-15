library(dplyr)
library(tibble)
library(readr)

# Set working directories and load initial data
setwd("C://Users//guest//Desktop//OneDrive_2025-01-04//Van Kerschaver & Xu - Benchmarking graph neural networks for churn prediction - shared folder//Data//Original data (R-format)//MobileVikings")

load("users.RData")
key <- data.frame(users = users, v2 = seq_along(users))

load("L_M1.RData")
usr <- L_M1$USR

setwd("C:\\Users\\guest\\Downloads\\granularData-20250408T162137Z-001\\granularData\\MobileVikings")

# Function to process core metrics for a specific month
process_month_metrics <- function(month_folder, variables_df, day_offset = 0) {
  max_days <- ifelse(month_folder == "October/", 31, 
                     ifelse(month_folder %in% c("December/", "January/"), 31, 30))
  
  for(i in max_days:1) {  # Process days in reverse order
    file <- paste0(month_folder, i, ".RData")
    if(file.exists(file)) {
      load(file)
      f <- as.data.frame(f) %>% as_tibble()
      f$X7 <- key[match(unlist(f[,1]), key[,1]), 2]
      
      f <- f %>% 
        group_by(X7) %>% 
        summarize(
          count = sum(count),
          length = sum(length)
        )
      
      current_days <- day_offset + (max_days - i)
      a <- usr %in% f$X7
      b <- f$X7 %in% usr
      
      # Update recency (minimum days)
      variables_df$R_on[a] <- pmin(variables_df$R_on[a], current_days)
      
      # Update appropriate time windows
      if(current_days < 30) {
        variables_df$M_30_on[a] <- f$length[b] + variables_df$M_30_on[a]
        variables_df$F_30_on[a] <- f$count[b] + variables_df$F_30_on[a]
      }
      if(current_days < 60) {
        variables_df$M_60_on[a] <- f$length[b] + variables_df$M_60_on[a]
        variables_df$F_60_on[a] <- f$count[b] + variables_df$F_60_on[a]
      }
    }
  }
  return(variables_df)
}

# Function to process network data for specific months
process_network_data <- function(month_folders) {
  all_data <- list()
  
  for(month in month_folders) {
    max_days <- ifelse(month == "October/", 31, 
                       ifelse(month %in% c("December/", "January/"), 31, 30))
    
    for(i in 1:max_days) {
      file_path <- paste0(month, i, ".RData")
      if(file.exists(file_path)) {
        load(file_path)
        f <- as.data.frame(f) %>% as_tibble() %>%
          mutate(
            X7 = key[match(unlist(.[[1]]), key[,1]), 2],
            X8 = key[match(unlist(.[[2]]), key[,1]), 2]
          )
        all_data[[length(all_data) + 1]] <- f
      }
    }
  }
  
  bind_rows(all_data) %>% 
    group_by(X7, X8) %>%
    summarize(
      count = sum(count, na.rm = TRUE),
      length = sum(length, na.rm = TRUE),
      .groups = "drop"
    )
}

# Function to create dataset for a specific period
create_period_dataset <- function(months) {
  # Initialize dataframe
  df <- data.frame(
    USR = usr,
    R_on = rep(1000, length(usr)),
    M_30_on = rep(0, length(usr)),
    M_60_on = rep(0, length(usr)),
    F_30_on = rep(0, length(usr)),
    F_60_on = rep(0, length(usr))
  )
  
  # Process metrics for each month with proper day offsets
  day_offset <- 0
  for(month in months) {
    df <- process_month_metrics(month, df, day_offset)
    max_days <- ifelse(month == "October/", 31, 
                       ifelse(month %in% c("December/", "January/"), 31, 30))
    day_offset <- day_offset + max_days
  }
  
  # Process network data
  network_data <- process_network_data(months)
  
  # Calculate network features (30-day uses only last month)
  last_month_data <- process_network_data(months[length(months)])
  
  # 30-day features (last month only)
  tmp_30_dialing <- last_month_data %>% 
    filter(X7 %in% usr) %>% 
    group_by(X7) %>% 
    summarize(numDialing_30_on = n_distinct(X8)) %>%
    rename(USR = X7)
  
  tmp_30_dialed <- last_month_data %>% 
    filter(X8 %in% usr) %>% 
    group_by(X8) %>% 
    summarize(numDialed_30_on = n_distinct(X7)) %>%
    rename(USR = X8)
  
  # 60-day features (all months)
  tmp_60_dialing <- network_data %>% 
    filter(X7 %in% usr) %>% 
    group_by(X7) %>% 
    summarize(numDialing_60_on = n_distinct(X8)) %>%
    rename(USR = X7)
  
  tmp_60_dialed <- network_data %>% 
    filter(X8 %in% usr) %>% 
    group_by(X8) %>% 
    summarize(numDialed_60_on = n_distinct(X7)) %>%
    rename(USR = X8)
  
  # Join all features
  df <- df %>%
    left_join(tmp_30_dialing, by = "USR") %>%
    left_join(tmp_30_dialed, by = "USR") %>%
    left_join(tmp_60_dialing, by = "USR") %>%
    left_join(tmp_60_dialed, by = "USR")
  
  # Handle NAs and add churn
  df[is.na(df)] <- 0
  df$churn <- L_M1$churn_m1
  
  return(df)
}

# Create all datasets
train_data <- create_period_dataset(c("November/", "October/"))
val_data   <- create_period_dataset(c("December/", "November/"))
test_data  <- create_period_dataset(c("January/", "December/"))

# Save as CSV files
write_csv(train_data, "train_rmf.csv")
write_csv(val_data, "val_rmf.csv")
write_csv(test_data, "test_rmf.csv")

# Also save as RData for compatibility
save(train_data, val_data, test_data, file = "all_periods_data.RData")
