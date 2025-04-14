library(dplyr)
library(tibble)

# Set working directories and load initial data
setwd("C://Users//guest//Desktop//OneDrive_2025-01-04//Van Kerschaver & Xu - Benchmarking graph neural networks for churn prediction - shared folder//Data//Original data (R-format)//MobileVikings")

load("users.RData")
key <- data.frame(users = users, v2 = seq_along(users))

load("L_M1.RData")
usr <- L_M1$USR

# Initialize variables
variables_inNet <- data.frame(
  USR = usr,
  R_on = rep(1000, length(usr)),
  M_30_on = rep(0, length(usr)),
  M_60_on = rep(0, length(usr)),
  M_90_on = rep(0, length(usr)),
  F_30_on = rep(0, length(usr)),
  F_60_on = rep(0, length(usr)),
  F_90_on = rep(0, length(usr))
)

setwd("C:\\Users\\guest\\Downloads\\granularData-20250408T162137Z-001\\granularData\\MobileVikings")

# Process November data for 30-day features (original calculation)
days <- 0
for(i in 30:1) {
  file <- paste0("November/", i, ".RData")
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
    
    a <- usr %in% f$X7
    b <- f$X7 %in% usr
    
    variables_inNet$R_on[a] <- pmin(variables_inNet$R_on[a], days)
    variables_inNet$M_30_on[a] <- f$length[b] + variables_inNet$M_30_on[a]
    variables_inNet$F_30_on[a] <- f$count[b] + variables_inNet$F_30_on[a]
  }
  days <- days + 1
}

# Process October data for 60-day features (additional calculation)
for(i in 31:1) {
  file <- paste0("October/", i, ".RData")
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
    
    a <- usr %in% f$X7
    b <- f$X7 %in% usr
    
    variables_inNet$R_on[a] <- pmin(variables_inNet$R_on[a], days)
    variables_inNet$M_60_on[a] <- f$length[b] + variables_inNet$M_60_on[a]
    variables_inNet$F_60_on[a] <- f$count[b] + variables_inNet$F_60_on[a]
  }
  days <- days + 1
}

# Network data processing function
process_network_month <- function(month_folder, usr, key) {
  max_days <- ifelse(month_folder == "October/", 31, 30)
  all_data <- list()
  
  for(i in 1:max_days) {
    file_path <- paste0(month_folder, i, ".RData")
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
  bind_rows(all_data)
}

# Process network data
november_network <- process_network_month("November/", usr, key)
october_network <- process_network_month("October/", usr, key)

# Calculate network features
calculate_network_features <- function(network_data, usr) {
  list(
    dialing = network_data %>%
      filter(X7 %in% usr) %>%
      group_by(X7) %>%
      summarize(value = n_distinct(X8)) %>%
      rename(USR = X7),
    
    dialed = network_data %>%
      filter(X8 %in% usr) %>%
      group_by(X8) %>%
      summarize(value = n_distinct(X7)) %>%
      rename(USR = X8)
  )
}

# 30-day features (November only)
net_30 <- calculate_network_features(november_network, usr)
variables_inNet$numDialing_30_on <- net_30$dialing$value[match(variables_inNet$USR, net_30$dialing$USR)]
variables_inNet$numDialed_30_on <- net_30$dialed$value[match(variables_inNet$USR, net_30$dialed$USR)]

# 60-day features (October + November)
full_network <- bind_rows(october_network, november_network)
net_60 <- calculate_network_features(full_network, usr)
variables_inNet$numDialing_60_on <- net_60$dialing$value[match(variables_inNet$USR, net_60$dialing$USR)]
variables_inNet$numDialed_60_on <- net_60$dialed$value[match(variables_inNet$USR, net_60$dialed$USR)]

# Final processing
variables_inNet[is.na(variables_inNet)] <- 0
variables_inNet$churn <- L_M1$churn_m1
variables_inNet <- as.data.frame(variables_inNet)

# Save results
save(variables_inNet, file = "test1404.RData")
