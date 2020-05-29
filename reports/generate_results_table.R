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
  # for each file, return accs and epoch
  outputs <- lapply(files, function(x){
    accData <- read_csv(x, col_types=cols()) %>%
      select(c(correct, prediction)) %>%
      mutate(response = correct == prediction) %>%
      summarise(accuracy = mean(response) * 100)
  })
  
  epochs <- lapply(files, function(x){
    eData <- read_csv(x, col_types=cols()) %>%
      select(epoch) %>%
      slice(1) 
  })
  
  expName <- str_split_n(directoryName, "/",6)
  result <- data.frame(
    matrix(ncol = 1, nrow=length(outputs)))
  colnames(result) <- c("accuracies")
  result$accuracies <- unlist(outputs)
  result$experiment <- expName
  result$epoch <- unlist(epochs) 
  
  return(result)
}


writeResultsTable <- function(modelName){
  resultsDir <- str_glue("../src/results/{modelName}/")
  
  experiments <- list.dirs(resultsDir)[-c(1)]
  
  outputs <- lapply(experiments, pullAccuracies)
  
  outputsDF <- do.call(rbind, outputs)
  
  #outputsDF %>%
  #  group_by(experiment) %>%
  #  summarise(meanAcc = mean(accuracies)) %>%
  #  print(n=Inf)
  
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
    mutate(
      meanAcc = round(meanAcc, digits=2),
      me = round(me, digits=2)
    ) %>%
    select(-c(accuracies,sd,n,epoch)) %>%
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
    mutate(outFormat = str_glue("{meanAcc} [$\\pm${me}]")) %>%
    select(-c(meanAcc, me)) %>%
    pivot_wider(
      names_from = dataset,
      #values_from = c(meanAcc, me)
      values_from = outFormat
    )
  
  outputTable <- outputTable %>%
    mutate(
      ExpName = str_glue("{exp}-{length_heuristic}")
    ) %>%
    select(-c(exp, length_heuristic)) %>%
    mutate(
      ExpName = recode(
        ExpName,
        "baseline-True" = "Fully Supervised",
        "naacl_linear-False" = "CB Linear ($d_{irt}$)",
        "naacl_linear-True" = "CB Linear ($d_{length}$)",
        "naacl_root-False" = "CB Root ($d_{irt}$)",
        "naacl_root-True" = "CB Root ($d_{length}$)",
        "theta-False" = "DDaCLAE"
      )
    ) %>%
    select(ExpName, everything())
    
  result <- xtable(
    outputTable, 
    type = "latex",
    caption = c("dev set accuracy results, including 95\\% confidence intervals, for each task under consideration. During training, 10\\% of the training set was held out and used for early stopping. Highest overall accuracy is bolded. Highest accuracy among competence-based methods is underlined"),
    label = c(str_glue("tab:acc_{modelName}")),
    align = c("llcccccc")
)
  
  print(result, 
        file=str_glue("ddaclae_accuracies_{modelName}.tex"), 
        sanitize.text.function = function(x) {x},
        booktabs = TRUE,
        floating.environment = "table*",
        latex.environments = "center",
        size = "small",
        include.rownames=FALSE
  )
  
  # do a table of average epoch to convergence
  outputTable2 <- outputsDF %>%
    mutate(experiment =  str_replace(experiment, "naacl-", "naacl_")) %>%
    mutate(experiment=str_replace(experiment,"SST-2","SST2")) %>%
    group_by(experiment) %>%
    mutate(
      meanEpoch = mean(epoch),
      sd = sd(epoch),
      n=n()
    ) %>%
    mutate(
      me = qnorm(0.975) * sd/sqrt(n)
    ) %>%
    mutate(
      meanEpoch = round(meanEpoch, digits=2),
      me = round(me, digits=2)
    ) %>%
    select(-c(accuracies,sd,n,epoch)) %>%
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
    mutate(outFormat = str_glue("{meanEpoch} [$\\pm${me}]")) %>%
    select(-c(meanEpoch, me)) %>%
    pivot_wider(
      names_from = dataset,
      #values_from = c(meanEpoch, me)
      values_from = outFormat
    )
  
  outputTable2 <- outputTable2 %>%
    mutate(
      ExpName = str_glue("{exp}-{length_heuristic}")
    ) %>%
    select(-c(exp, length_heuristic)) %>%
    mutate(
      ExpName = recode(
        ExpName,
        "baseline-True" = "Fully Supervised",
        "naacl_linear-False" = "CB Linear ($d_{irt}$)",
        "naacl_linear-True" = "CB Linear ($d_{length}$)",
        "naacl_root-False" = "CB Root ($d_{irt}$)",
        "naacl_root-True" = "CB Root ($d_{length}$)",
        "theta-False" = "DDaCLAE"
      )
    ) %>%
    select(ExpName, everything())
  
  result <- xtable(
    outputTable2, 
    type = "latex",
    caption = c("Average epoch of convergence for each model, with 95\\% confidence intervals."),
    label = c(str_glue("tab:epoch_{modelName}")),
    align = c("llcccccc")
  )
  
  print(result,
        #xtable(outputTable2, type="latex"), 
        file=str_glue("ddaclae_avgEpoch_{modelName}.tex"), 
        sanitize.text.function = function(x) {x},
        floating.environment = "table*",
        latex.environments = "center",
        size = "small",
        include.rownames=FALSE
  )
}

writeResultsTable("bert-True")
writeResultsTable("lstm-True")

writeResultsTable("bert-False")
writeResultsTable("lstm-False")
