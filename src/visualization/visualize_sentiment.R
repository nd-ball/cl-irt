library(tidyverse)

# sstb
data_dir <- 'G:/My Drive/2019/research/projects/cl_irt/aaai_logs/aaai_run2/'
exp_type <- 'sstb'
num_skip <- 18
D.baseline <- read_csv(paste(data_dir, exp_type,'-baseline.log',sep=''), 
                       col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                       skip=num_skip, n_max=200) 
D.baseline$epoch <- c(1:200) 
D.baseline$exp <- 'baseline'

D.irt <- read_csv(paste(data_dir, 'irt-cl-', exp_type, '-1000.log', sep=''),
                  col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                  skip=num_skip, n_max=200)
D.irt$epoch <- c(1:200)
D.irt$exp <- 'irt'

D.linear.easiest <- read_csv(paste(data_dir,exp_type, '-naacl-linear-easiest.log',sep=''), 
                             col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                             skip=num_skip, n_max=200) 
D.linear.easiest$epoch <- c(1:200) 
D.linear.easiest$exp <- 'naacl-linear-easiest'

D.root.easiest <- read_csv(paste(data_dir,exp_type, '-naacl-root-easiest.log',sep=''), 
                           col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                           skip=num_skip, n_max=200) 
D.root.easiest$epoch <- c(1:200) 
D.root.easiest$exp <- 'naacl-root-easiest'



D <- rbind(D.baseline, D.irt, D.linear.easiest,D.root.easiest)
filter <- D %>%
  group_by(exp) %>%
  summarize(max=max(val_acc)) 

max_epochs <- merge(D,filter, by.x=c('exp','val_acc'), by.y=c('exp','max'))

which(D$exp=='baseline' & D$epoch==149)
which(D$exp=='naacl-linear-easiest' & D$epoch==162)
which(D$exp=='irt' & D$epoch==140)
which(D$exp=='naacl-root-easiest' & D$epoch==183)

png("../../reports/figures/cl_irt_sstb.png", width=1100, height=700)
ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/67348, color=exp),D, linetype=2) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(126,493,246,770,832),]) + 
  theme_minimal() + 
  ggtitle("Comaprison of CL Strategies: SSTB") + 
  ylab("Test accuracy") + 
  xlab("Epoch") + 
  scale_color_discrete(name='Experiment',
                       breaks=c('baseline', 'easiest', 'irt', 'middleout', 'ordered'),
                       labels=c('Baseline', 'EasyFirst', 'Theta', 'MiddleOut','Ordered'))
dev.off()


######### Table to show how much data was required to get to best acc #################
sum(D[which(D$exp=='baseline'&D$epoch <= max_epochs[which(max_epochs$exp=='baseline'),]$epoch),]$train_size)
sum(D[which(D$exp=='irt'&D$epoch <= max_epochs[which(max_epochs$exp=='irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='easiest'&D$epoch <= max_epochs[which(max_epochs$exp=='easiest'),]$epoch),]$train_size)
sum(D[which(D$exp=='middleout'&D$epoch <= max_epochs[which(max_epochs$exp=='middleout'),]$epoch),]$train_size)
sum(D[which(D$exp=='ordered'&D$epoch <= max_epochs[which(max_epochs$exp=='ordered'),]$epoch),]$train_size)
