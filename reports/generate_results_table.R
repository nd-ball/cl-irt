# calculate accuracies for ddaclae and generate table
library(readr)
library(dplyr)
library(stringr)

setwd("~/code/cl-irt/reports/")

resultsDir <- "../src/results/bert/"

experiments <- list.dirs(resultsDir)

pullAccuracies <- function(directoryName){
  files <- list.files(directoryName) %>%
    filter('preds' %in% files) 
  # for each file, return accs
  outputs <- lapply(files, function(x){
    accData <- read_csv(x) %>%
      select(c(correct, prediction)) %>%
      mutate(response = correct == prediction) %>%
      summarise(accuracy = mean(response))
  })
  
  splits <- str_split_n(directoryName, '-', 1)
  return(splits, outputs)
}


