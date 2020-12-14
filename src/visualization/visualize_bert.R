library(tidyverse)


getTopDevAccResults <- function(singleRun){
  topEpoch <- which(singleRun$dev_acc == max(singleRun$dev_acc))
  return(singleRun[topEpoch,])
}

getResultsSingleTask <- function(dataset, task, useLen, useWR){
  # different types of experiments
  # 5 files for each task
  # print the test acc results, and generate a plot 
  # start with bert-False
  baseDir <- "../data/bert-True/"
  
  baseFolder <- paste(baseDir,dataset,"-", task, "-len-", useLen, "-wordrarity-", useWR,sep='')
  list_of_files <- list.files(baseFolder, pattern = "tracker.csv", recursive = T, full.names=T)
  list_results <- map(list_of_files, possibly(read.csv, otherwise = NULL))
  
  list_results <- list_results[!is.na(list_results)]
  
  # now that I have the files, what do I want to do? 
  # ID best dev acc, then get appropriate epoch and test acc
  
  best <- bind_rows(lapply(list_results, getTopDevAccResults)) %>%
    summarise_all(list(mean, sd), na.rm=T)
  
}

data <- list(
datasets = c("MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2"),
tasks = c("baseline", "naacl-linear", "naacl-root", "theta"),
lens = c("True", "False"),
wrs = c("True", "False") 
)

D <- cross_df(data) %>%
  filter(!(lens == "True" & wrs == "True")) %>%
  filter(!((lens != "False" | wrs != "False") & tasks == "theta")) 

Dvals <- bind_rows(mapply(getResultsSingleTask, D$datasets, D$tasks, D$lens, D$wrs))

D <- cbind(D, Dvals) %>%
  filter(!(!(lens == "True" & wrs == "False") & tasks == "baseline"))


D.plotting <- D %>%
  select(-c(
    test_acc_fn1,
    test_acc_fn2
  )
  ) 



