setwd("C://Users//guest//Desktop//OneDrive_2025-01-04//Van Kerschaver & Xu - Benchmarking graph neural networks for churn prediction - shared folder//Data//Original data (R-format)//MobileVikings")


load("users.RData")
key <- data.frame(users = users, v2 = seq_along(users))

load("L_M1.RData")
usr<-L_M1$USR

variables_inNet<-cbind(USR=usr,R_on=rep(1000,length(usr)),M_30_on=rep(0,length(usr)),M_60_on=rep(0,length(usr)),M_90_on=rep(0,length(usr)),F_30_on=rep(0,length(usr)),F_60_on=rep(0,length(usr)),F_90_on=rep(0,length(usr)))
#Start looking in march, from the back
variables_inNet<-as.data.frame(variables_inNet)

setwd("C:\\Users\\guest\\Downloads\\granularData-20250408T162137Z-001\\granularData\\MobileVikings")
days<-0
folder<-"November/"
for(i in c(30:1))
{  
  file<-paste(folder,i,".RData",sep="")
  load(file)
  f <- as.data.frame(f)        # First convert to base data.frame
  f <- tibble::as_tibble(f)    # Then convert to clean tibble
  f$X7<-key[match(unlist(f[,1]),key[,1]),2]
  f <- f %>% 
    group_by(X7) %>% 
    summarize(
      count = sum(count),
      length = sum(length)
      #.groups = "drop"  # Explicitly drop grouping after
    )
  a<-usr%in%f$X7
  b<-f$X7%in%usr
  variables_inNet$R_on[a]<-pmin(variables_inNet$R_on[a],days)#ifelse(a,day,NA)
  variables_inNet$M_30_on[which(a)]<-f$length[b]+variables_inNet$M_30_on[which(a)]
  variables_inNet$F_30_on[which(a)]<-f$count[b]+variables_inNet$F_30_on[which(a)]
  days<-days+1
}


# NumDialing and NumDialed code
# Improved code
# Initialize an empty list to store daily data (more memory-efficient)
daily_data <- list()

# Process November data (days 1-30)
for (i in 1:30) {
  # Safely load and validate data
  file_path <- paste0("November/", i, ".RData")
  if (!file.exists(file_path)) {
    warning(paste("File not found:", file_path))
    next
  }
  
  load(file_path)
  
  # Convert to tibble and validate structure
  f <- tryCatch(
    {
      f <- as.data.frame(f) %>% 
        tibble::as_tibble() %>%
        mutate(
          X7 = key[match(unlist(.[[1]]), key[,1]), 2],
          X8 = key[match(unlist(.[[2]]), key[,1]), 2]
        )
    },
    error = function(e) {
      stop(paste("Error processing file", file_path, ":", e$message))
    }
  )
  
  # Store in list instead of immediate binding
  daily_data[[i]] <- f
}

# Single aggregation (faster than incremental binding)
November_data <- bind_rows(daily_data) %>% 
  group_by(X7, X8) %>%
  summarize(
    count = sum(count, na.rm = TRUE),
    length = sum(length, na.rm = TRUE),
    .groups = "drop"
  )

f_m<-November_data

# Clean up
rm(daily_data, f)
gc()  # Optional: Free up memory


tmp<-f_m%>%group_by(X7)%>%summarize(numDialing_30_on=n_distinct(X8))
colnames(tmp)[1]<-"USR"
variables_inNet<-variables_inNet%>%left_join(tmp)
tmp<-f_m%>%group_by(X8)%>%summarize(numDialed_30_on=n_distinct(X7))
colnames(tmp)[1]<-"USR"
variables_inNet<-variables_inNet%>%left_join(tmp)


variables_inNet[is.na(variables_inNet)]<-0
variables_inNet$churn<-L_M1$churn_m1
variables_inNet<-as.data.frame(variables_inNet)
save(variables_inNet,file="train_rmf_pre.RData")

