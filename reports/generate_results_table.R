# calculate accuracies for ddaclae and generate table
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(xtable)

setwd("~/code/cl-irt/reports/")


pullAccuracies <- function(directoryName){
  files <- list.files(
    directoryName,
    recursive = T,
    full.names=T) 
  files <- files[str_detect(files, 'preds')]
  # for each file, return accs
  outputs <- lapply(files, function(x){
    
    accData <- read_csv(x) %>%
      select(c(correct, prediction)) %>%
      mutate(response = correct == prediction) %>%
      summarise(accuracy = mean(response) * 100)
  })
  
  expName <- str_split_n(directoryName, "/",6)
  result <- data.frame(
    matrix(ncol = 1, nrow=length(outputs)))
  colnames(result) <- c("accuracies")
  result$accuracies <- unlist(outputs)
  result$experiment <- expName
  
  return(result)
}


writeResultsTable <- function(modelName){
  resultsDir <- str_glue("../src/results/{modelName}/")
  
  experiments <- list.dirs(resultsDir)[-c(1)]
  
  outputs <- lapply(experiments, pullAccuracies)
  
  outputsDF <- do.call(rbind, outputs)
  
  outputsDF %>%
    group_by(experiment) %>%
    summarise(meanAcc = mean(accuracies)) %>%
    print(n=Inf)
  
  outputTable <- outputsDF %>%
    mutate(experiment =  str_replace(experiment, "naacl-", "naacl_")) %>%
  mutate(experiment=str_replace(experiment,"SST-2","SST2")) %>%
      group_by(experiment) %>%
    mutate(
      meanAcc = mean(accuracies),
      sd = sd(accuracies),
      n=n()
    ) %>%
    mutate(
      me = qnorm(0.975) * sd/sqrt(n)
    ) %>%
    select(-c(accuracies,sd,n)) %>%
    unique() %>%
    separate(
      experiment, 
      c(
        "dataset",
        "exp",
        "dropping",
        "length_heuristic"
      ),
      sep='-'
    ) %>%
    select(-dropping) %>%
    filter(dataset != 'WNLI') %>%
    pivot_wider(
      names_from = dataset,
      values_from = c(meanAcc, me)
    )
  
  print(xtable(outputTable, type="latex"), file=str_glue("ddaclae_accuracies_{modelName}.tex"))
}

writeResultsTable("bert-True")
writeResultsTable("lstm-True")

writeResultsTable("bert-False")
writeResultsTable("lstm-False")
