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



getSingleTopDevAccResults <- function(singleRun){
  topEpoch <- which(singleRun$dev_acc == max(singleRun$dev_acc))
  return(max(singleRun$dev_acc))
}

getBestResultSingleTask <- function(dataset, task, useLen, useWR){
  # different types of experiments
  # 5 files for each task
  # return the tracking code for the best single run by dev epoch
  baseDir <- "../data/bert-True/"
  
  baseFolder <- paste(baseDir,dataset,"-", task, "-len-", useLen, "-wordrarity-", useWR,sep='')
  list_of_files <- list.files(baseFolder, pattern = "tracker.csv", recursive = T, full.names=T)
  list_results <- map(list_of_files, possibly(read.csv, otherwise = NULL))
  
  list_results <- list_results[!is.na(list_results)]
  
  # now that I have the files, what do I want to do? 
  # ID best dev acc, then get appropriate epoch and test acc
  
  best <- lapply(list_results, getSingleTopDevAccResults)
  topval <- which(best == max(unlist(best)))[1]
  result <- list_results[[topval]] %>%
    mutate(num_training_examples = num_training_examples / max(num_training_examples)) %>%
    pivot_longer(-epoch,
                 names_to = "metric",
                 values_to = "value") %>%
    mutate(dataset = dataset)  %>%
    mutate(task = task) %>%
    mutate(useLen = useLen) %>%
    mutate(useWR = useWR)
  return(result)
}

data <- list(
  datasets = c("MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2"),
  tasks = c("baseline", "naacl-linear", "naacl-root", "theta"),
  lens = c("True", "False"),
  wrs = c("True", "False") 
)

D <- cross_df(data) %>%
  filter(!(lens == "True" & wrs == "True")) %>%
  filter(!((lens != "False" | wrs != "False") & tasks == "theta")) %>%
  filter(!(!(lens == "True" & wrs == "False") & tasks == "baseline"))

Dvals <- bind_rows(mapply(getBestResultSingleTask, D$datasets, D$tasks, D$lens, D$wrs, SIMPLIFY=F))


# test plot
# this works, next question is which tasks to include in the plot
# so that it isn't over-crowded
# definitely: baseline, DDaCLAE, IRT + CBCL 

p <- Dvals %>%
  pivot_wider(names_from = metric, values_from= value) %>%
  select(-test_acc) %>%
  pivot_longer(
    cols = c(num_training_examples, dev_acc),
    names_to="Metric") %>%
  mutate(Experiment = paste(task, useLen, useWR,sep="-")) %>%
  filter(Experiment %in% c(
    "baseline-True-False",
    #"naacl-linear-True-False",
    "naacl-root-True-False",
    "theta-False-False"
  )) %>%
  ggplot(aes(x=epoch)) + 
  geom_line(aes(y=value, linetype=Metric, color=Experiment)) + 
    facet_wrap(vars(dataset), ncol=2) +
  ggtitle("Training efficiency plot: BERT") + 
  theme_minimal() + 
  xlab("Training epoch")



ggsave("journal_plots/bert_balanced_data_plots.png", p, width=8, height=4)


###### BERT UNBALANCED #####
ggsave("journal_plots/bert_unbalanced_data_plots.png", p, width=8, height=4)
