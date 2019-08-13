library(tidyverse)

# sstb
data_dir <- 'G:/My Drive/2019/research/projects/cl_irt/sstb_run3/'
exp_type <- 'snli'
num_skip <- 17
D.baseline <- read_csv(paste(data_dir, exp_type,'_baseline.log',sep=''), 
                       col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                       skip=num_skip, n_max=200) 
D.baseline$epoch <- c(1:200) 
D.baseline$exp <- 'baseline'

D.irt <- read_csv(paste(data_dir, 'irt_cl_', exp_type, '_5000.log', sep=''),
                  col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                  skip=num_skip, n_max=200)
D.irt$epoch <- c(1:118)
D.irt$exp <- 'irt'

D.easiest <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-easiest.log',sep=''), 
                      col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                      skip=num_skip, n_max=200) 
D.easiest$epoch <- c(1:200) 
D.easiest$exp <- 'easiest'

D.ordered <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-ordered-easiest.log',sep=''), 
                      col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                      skip=num_skip, n_max=200) 
D.ordered$epoch <- c(1:200) 
D.ordered$exp <- 'ordered'


D.middleout <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-middleout.log',sep=''), 
                        col_names = c('a','b','train_size', 'train_acc', 'dev_acc', 'test_acc', 'theta'),
                        skip=num_skip, n_max=200) 
D.middleout$epoch <- c(1:200) 
D.middleout$exp <- 'middleout'

D <- rbind(D.baseline, D.irt, D.easiest, D.middleout,D.ordered)
filter <- D %>%
  group_by(exp) %>%
  summarize(max=max(dev_acc)) 

max_epochs <- merge(D,filter, by.x=c('exp','dev_acc'), by.y=c('exp','max'))

which(D$exp=='baseline' & D$epoch==20)
which(D$exp=='easiest' & D$epoch==97)
which(D$exp=='irt' & D$epoch==31)
which(D$exp=='middleout' & D$epoch==104)
which(D$exp=='ordered' & D$epoch==180)


png("../../reports/figures/cl_irt_snli.png", width=1100, height=700)
ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/549184, color=exp),D, linetype=2) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(20,415,231,622,898),]) + 
  theme_minimal() + 
  ggtitle("Comaprison of CL Strategies: SNLI") + 
  ylab("Test accuracy") + 
  xlab("Epoch") + 
  scale_color_discrete(name='Experiment',
                       breaks=c('baseline', 'easiest', 'irt', 'middleout', 'ordered'),
                       labels=c('Baseline', 'EasyFirst', 'Theta', 'MiddleOut', 'Ordered'))
dev.off()

######### Table to show how much data was required to get to best acc #################
sum(D[which(D$exp=='baseline'&D$epoch <= max_epochs[which(max_epochs$exp=='baseline'),]$epoch),]$train_size)
sum(D[which(D$exp=='irt'&D$epoch <= max_epochs[which(max_epochs$exp=='irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='easiest'&D$epoch <= max_epochs[which(max_epochs$exp=='easiest'),]$epoch),]$train_size)
sum(D[which(D$exp=='middleout'&D$epoch <= max_epochs[which(max_epochs$exp=='middleout'),]$epoch),]$train_size)
sum(D[which(D$exp=='ordered'&D$epoch <= max_epochs[which(max_epochs$exp=='ordered'),]$epoch),]$train_size)

