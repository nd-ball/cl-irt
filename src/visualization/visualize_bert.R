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

# sstb
exp_type <- 'snli'
num_skip <- 17
D.baseline <- read_csv(paste(data_dir, exp_type,'-baseline.log',sep=''), 
                       col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                       skip=num_skip, n_max=200) 
D.baseline$epoch <- c(1:200) 
D.baseline$exp <- 'baseline'

D.irt <- read_csv(paste(data_dir, 'irt-cl-', exp_type, '-5000.log', sep=''),
                  col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                  skip=num_skip, n_max=200)
D.irt$epoch <- c(1:200)
D.irt$exp <- 'irt'

D.linear.easiest.irt <- read_csv(paste(data_dir,exp_type, '-naacl-linear-easiest-irt.log',sep=''), 
                                 col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                                 skip=num_skip, n_max=200) 
D.linear.easiest.irt$epoch <- c(1:200) 
D.linear.easiest.irt$exp <- 'naacl-linear-easiest-irt'

D.root.easiest.irt <- read_csv(paste(data_dir,exp_type, '-naacl-root-easiest-irt.log',sep=''), 
                               col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                               skip=num_skip, n_max=200) 
D.root.easiest.irt$epoch <- c(1:200) 
D.root.easiest.irt$exp <- 'naacl-root-easiest-irt'

D.linear.easiest.length <- read_csv(paste(data_dir,exp_type, '-naacl-linear-easiest-length.log',sep=''), 
                                    col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                                    skip=num_skip, n_max=200) 
D.linear.easiest.length$epoch <- c(1:200) 
D.linear.easiest.length$exp <- 'naacl-linear-easiest-length'

D.root.easiest.length <- read_csv(paste(data_dir,exp_type, '-naacl-root-easiest-length.log',sep=''), 
                                  col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                                  skip=num_skip, n_max=200) 
D.root.easiest.length$epoch <- c(1:200) 
D.root.easiest.length$exp <- 'naacl-root-easiest-length'



D <- rbind(D.baseline, D.irt, D.linear.easiest.irt,D.root.easiest.irt,D.linear.easiest.length,D.root.easiest.length)
filter <- D %>%
  group_by(exp) %>%
  summarize(max=max(dev_acc)) 

max_epochs <- merge(D,filter, by.x=c('exp','dev_acc'), by.y=c('exp','max'))

which(D$exp=='baseline' & D$epoch==23)
which(D$exp=='irt' & D$epoch==36)
which(D$exp=='naacl-linear-easiest-irt' & D$epoch==57)
which(D$exp=='naacl-linear-easiest-length' & D$epoch==55)
which(D$exp=='naacl-root-easiest-irt' & D$epoch==54)
which(D$exp=='naacl-root-easiest-length' & D$epoch==32)

cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

png("../../reports/figures/cl_irt_snli.png", width=800, height=400)
ggplot(D, aes(x=epoch, y=test_acc*100, color=exp))  + 
  geom_line(show.legend = T) + 
  geom_line(aes(x=epoch, y=train_size/5491.84, color=exp),D, linetype=2, show.legend = T) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(23,236,457,855,654,1032),], show.legend = T) + 
  theme_minimal() + 
  ggtitle("Comparison of CL Strategies: SNLI") + 
  ylab("Test accuracy") + 
  xlab("Epoch") + 
  ylim(50,100) +
  xlim(0, 100) +
  scale_color_manual(name='Experiment',
                       breaks=c('baseline', 'naacl-linear-easiest-irt', 'naacl-linear-easiest-length',
                                'irt', 'naacl-root-easiest-irt', 'naacl-root-easiest-length'),
                       labels=c('Baseline', 'CB-L-IRT', 'CB-L-Length', 
                                'DDaCLAE', 'CB-R-IRT', 'CB-R-Length'),
                       values=cbbPalette)
dev.off()

# new plot for acl 2020 submission 
D.acl <- D[which(D$exp %in% c('baseline', 'naacl-linear-easiest-irt', 
                              'naacl-root-easiest-irt', 'irt')),]

png("../../reports/figures/cl_irt_snli_acl2020.png", width=400, height=200)
ggplot(D.acl, aes(x=epoch, y=train_size/5491.84, color=exp, linetype=exp))  + 
  geom_line(show.legend = T) + 
  theme_minimal() + 
  ggtitle("Training Data Use: SNLI") + 
  ylab("Training data used (%)") + 
  xlab("Epoch") + 
  ylim(50,100) +
  xlim(0, 75) +
  scale_linetype_manual(
    name='Experiment',
    breaks=c('baseline', 'naacl-linear-easiest-irt', 
             'naacl-root-easiest-irt', 'irt'),
    labels=c('Baseline', 'CB-L',  
             'CB-R', 'DDaCLAE'),
    values=c(1,2,3,4)) + 
  scale_color_manual(name='Experiment',
                     breaks=c('baseline', 'naacl-linear-easiest-irt', 
                              'naacl-root-easiest-irt', 'irt'),
                     labels=c('Baseline', 'CB-L',  
                              'CB-R', 'DDaCLAE'),
                     values=cbbPalette)
dev.off()


######### Table to show how much data was required to get to best acc #################
sum(D[which(D$exp=='baseline'&D$epoch <= max_epochs[which(max_epochs$exp=='baseline'),]$epoch),]$train_size)
sum(D[which(D$exp=='irt'&D$epoch <= max_epochs[which(max_epochs$exp=='irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='naacl-linear-easiest-irt'&D$epoch <= max_epochs[which(max_epochs$exp=='naacl-linear-easiest-irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='naacl-root-easiest-irt'&D$epoch <= max_epochs[which(max_epochs$exp=='naacl-root-easiest-irt'),]$epoch),]$train_size)

