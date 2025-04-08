setwd("C:\\Users\\guest\\Desktop\\OneDrive_2025-01-04\\Van Kerschaver & Xu - Benchmarking graph neural networks for churn prediction - shared folder\\Data\\Original data (R-format)\\ProximusPre")
library(dplyr)
install.packages("crayon")
library(crayon)
# train on oot
load("L_M1.RData") #labels come from april
usr<-L_M1$USR
# variables: resency, frequency, monetary (30,60,90 days) on/off net, #different numbers dialed on/off net, # different numbers dialling the number on/off net
key<-read.csv("Pre_IDKey.csv")
#################
## In network
################
variables_inNet<-cbind(USR=usr,R_on=rep(1000,length(usr)),M_30_on=rep(0,length(usr)),M_60_on=rep(0,length(usr)),M_90_on=rep(0,length(usr)),F_30_on=rep(0,length(usr)),F_60_on=rep(0,length(usr)),F_90_on=rep(0,length(usr)))
#Start looking in march, from the back
variables_inNet<-as.data.frame(variables_inNet)

setwd("C:\\Users\\guest\\Downloads\\granularData-20250408T162137Z-001\\granularData\\Prepaid")
days<-0
folder<-"May/"
for(i in c(30:1))
{  
  file<-paste(folder,i,".RData",sep="")
  load(file)
  f <- as.data.frame(f)        # First convert to base data.frame
  f <- tibble::as_tibble(f)    # Then convert to clean tibble
  f$A_NUMBER<-key[match(unlist(f[,1]),key[,1]),2]
  f <- f %>% 
    group_by(A_NUMBER) %>% 
    summarize(
      count_calls = sum(count_calls),
      seconds = sum(seconds)
      #.groups = "drop"  # Explicitly drop grouping after
    )
  a<-usr%in%f$A_NUMBER
  b<-f$A_NUMBER%in%usr
  variables_inNet$R_on[a]<-pmin(variables_inNet$R_on[a],days)#ifelse(a,day,NA)
  variables_inNet$M_30_on[which(a)]<-f$seconds[b]+variables_inNet$M_30_on[which(a)]
  variables_inNet$F_30_on[which(a)]<-f$count_calls[b]+variables_inNet$F_30_on[which(a)]
  days<-days+1
}


# NumDialing and NumDialed code
# Improved code
# Initialize an empty list to store daily data (more memory-efficient)
daily_data <- list()

# Process May data (days 1-30)
for (i in 1:30) {
  # Safely load and validate data
  file_path <- paste0("May/", i, ".RData")
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
          A_NUMBER = key[match(unlist(.[[1]]), key[,1]), 2],
          B_NUMBER = key[match(unlist(.[[2]]), key[,1]), 2]
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
may_data <- bind_rows(daily_data) %>% 
  group_by(A_NUMBER, B_NUMBER) %>%
  summarize(
    count_calls = sum(count_calls, na.rm = TRUE),
    seconds = sum(seconds, na.rm = TRUE),
    .groups = "drop"
  )

f_m<-may_data

# Clean up
rm(daily_data, f)
gc()  # Optional: Free up memory


tmp<-f_m%>%group_by(A_NUMBER)%>%summarize(numDialing_30_on=n_distinct(B_NUMBER))
colnames(tmp)[1]<-"USR"
variables_inNet<-variables_inNet%>%left_join(tmp)
tmp<-f_m%>%group_by(B_NUMBER)%>%summarize(numDialed_30_on=n_distinct(A_NUMBER))
colnames(tmp)[1]<-"USR"
variables_inNet<-variables_inNet%>%left_join(tmp)


variables_inNet[is.na(variables_inNet)]<-0
variables_inNet$churn<-L_M1$churn_m1
variables_inNet<-as.data.frame(variables_inNet)
save(variables_inNet,file="train_rmf_pre.RData")
