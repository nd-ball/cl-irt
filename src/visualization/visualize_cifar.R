library(tidyverse)

# sstb
data_dir <- 'G:/My Drive/data/curriculum_learning_irt/'
exp_type <- 'cifar'
num_skip <- 2
D.baseline <- read_csv(paste(data_dir, exp_type,'_baseline.log',sep=''), 
                       col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                       skip=num_skip, n_max=200) 
D.baseline$epoch <- c(1:200) 
D.baseline$exp <- 'baseline'

D.irt <- read_csv(paste(data_dir, 'irt_cl_', exp_type, '_5000.log', sep=''),
                  col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                  skip=num_skip, n_max=200)
D.irt$epoch <- c(1:200)
D.irt$exp <- 'irt'

D.easiest <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-easiest.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                      skip=num_skip, n_max=200) 
D.easiest$epoch <- c(1:200) 
D.easiest$exp <- 'easiest'

D.ordered <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-ordered-easiest.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                      skip=num_skip, n_max=200) 
D.ordered$epoch <- c(1:200) 
D.ordered$exp <- 'ordered'

D.middleout <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-middleout.log',sep=''), 
                        col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                        skip=num_skip, n_max=200) 
D.middleout$epoch <- c(1:200) 
D.middleout$exp <- 'middleout'

D <- rbind(D.baseline, D.irt, D.easiest, D.middleout, D.ordered)
filter <- D %>%
  group_by(exp) %>%
  summarize(max=max(test_acc)) 

max_epochs <- merge(D,filter, by.x=c('exp','test_acc'), by.y=c('exp','max'))

which(D$exp=='baseline' & D$epoch==135)
which(D$exp=='easiest' & D$epoch==188)
which(D$exp=='irt' & D$epoch==95)
which(D$exp=='middleout' & D$epoch==115)
which(D$exp=='ordered' & D$epoch==145)


png("../../reports/figures/cl_irt_cifar.png", width=1100, height=700)
ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/500, color=exp),D, linetype=2) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(135,588,295,715,945),]) + 
  theme_minimal() + 
  ggtitle("Comaprison of CL Strategies: CIFAR") + 
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

# load detailed RP data
rps_baseline <- read_csv(
  paste(data_dir, 'test_preds/cifar_baseline_False_easiest_False.csv',sep='')
)
rps_baseline$exp <- "baseline"

rps_irt <- read_csv(
  paste(data_dir, 'test_preds/cifar_theta_False_easiest_False.csv',sep='')
)
rps_irt$exp <- "Theta"

rps_easiest <- read_csv(
  paste(data_dir, 'test_preds/cifar_simple_False_easiest_False.csv',sep='')
)
rps_easiest$exp <- "Simple-Easiest"

rps_middleout <- read_csv(
  paste(data_dir, 'test_preds/cifar_simple_False_middleout_False.csv',sep='')
)
rps_middleout$exp <- "Simple-MiddleOut"

rps_ordered <- read_csv(
  paste(data_dir, 'test_preds/cifar_ordered_False_easiest_False.csv',sep='')
)
rps_ordered$exp <- "Ordered"


test_diffs <- read_csv(
  paste(data_dir, 'test_preds/cifar_diffs_test.csv',sep=''),
  col_names=c('pairid', 'diff')
)

rps_all <- rbind(rps_baseline, rps_irt, rps_easiest, rps_middleout, rps_ordered)

quantiles <- quantile(test_diffs$diff)

rps_all <- merge(rps_all, test_diffs, by.x='itemID', by.y='pairid')
rps_all$bin <- 1
rps_all[which(rps_all$diff >= quantiles[2]),]$bin <- 2
rps_all[which(rps_all$diff >= quantiles[3]),]$bin <- 3
rps_all[which(rps_all$diff >= quantiles[4]),]$bin <- 4

table(rps_all$bin)

Z <- rps_all %>% 
  group_by(epoch,bin,exp) %>%
  summarize(mean=mean(correct==pred))


ggplot(Z[which(Z$exp %in% c('baseline', 'Simple-MiddleOut')),],aes(x=epoch,y=mean,linetype=exp, color=as.factor(bin))) + 
  geom_line() 

Z[which(Z$exp=='baseline' & Z$epoch==135),]
Z[which(Z$exp=='Simple-Easiest' & Z$epoch==188),]
Z[which(Z$exp=='Theta' & Z$epoch==95),]
Z[which(Z$exp=='Simple-MiddleOut' & Z$epoch==115),]
Z[which(Z$exp=='Ordered' & Z$epoch==145),]
