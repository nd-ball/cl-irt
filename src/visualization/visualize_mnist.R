library(tidyverse)

# sstb
data_dir <- 'G:/My Drive/research/2019/projects/cl_irt/aaai_logs/aaai_run2/'
exp_type <- 'mnist'
num_skip <- 0
D.baseline <- read_csv(paste(data_dir, exp_type,'-baseline.log',sep=''), 
                       col_names = c('train_size', 'train_acc', 'val_acc', 'test_acc', 'theta'),
                       skip=num_skip, n_max=200) 
D.baseline$epoch <- c(1:200) 
D.baseline$exp <- 'baseline'

D.irt <- read_csv(paste(data_dir, 'irt-cl-', exp_type, '-1000.log', sep=''),
                  col_names = c('train_size', 'train_acc', 'val_acc', 'test_acc', 'theta'),
                  skip=num_skip, n_max=200)
D.irt$epoch <- c(1:200)
D.irt$exp <- 'irt'

D.linear.easiest <- read_csv(paste(data_dir,exp_type, '-naacl-linear-easiest.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'val_acc', 'test_acc', 'theta'),
                      skip=num_skip, n_max=200) 
D.linear.easiest$epoch <- c(1:200) 
D.linear.easiest$exp <- 'naacl-linear-easiest'

D.root.easiest <- read_csv(paste(data_dir,exp_type, '-naacl-root-easiest.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'val_acc', 'test_acc', 'theta'),
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

cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

p <-ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line(show.legend = T) + 
  geom_line(aes(x=epoch, y=train_size/500, color=exp),D, linetype=2, show.legend = T) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(149,562,340,783),], show.legend = T) + 
  theme_minimal() + 
  ggtitle("Comparison of CL Strategies: MNIST") + 
  ylab("Test accuracy (%)") + 
  xlab("Epoch") + 
  ylim(90,100) +
  scale_color_manual(name='Experiment',
                       breaks=c('baseline', 'naacl-linear-easiest', 'irt', 'naacl-root-easiest'),
                       labels=c('Baseline', 'CB-L', 'DDaCLAE', 'CB-R'),
                     values=cbbPalette)

ggsave("journal_plots/mnist_efficiency.png", p, width=7, height=3)

######### Table to show how much data was required to get to best acc #################
sum(D[which(D$exp=='baseline'&D$epoch <= max_epochs[which(max_epochs$exp=='baseline'),]$epoch),]$train_size)
sum(D[which(D$exp=='baseline'&D$epoch <= 149),]$train_size)
sum(D[which(D$exp=='irt'&D$epoch <= max_epochs[which(max_epochs$exp=='irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='naacl-linear-easiest'&D$epoch <= max_epochs[which(max_epochs$exp=='naacl-linear-easiest'),]$epoch),]$train_size)
sum(D[which(D$exp=='naacl-root-easiest'&D$epoch <= max_epochs[which(max_epochs$exp=='naacl-root-easiest'),]$epoch),]$train_size)


# load detailed RP data
rps_baseline <- read_csv(
  paste(data_dir, 'test_preds/mnist_baseline_False_easiest_False.csv',sep='')
)
rps_baseline$exp <- "baseline"

rps_irt <- read_csv(
  paste(data_dir, 'test_preds/mnist_theta_False_easiest_False.csv',sep='')
)
rps_irt$exp <- "Theta"

rps_easiest <- read_csv(
  paste(data_dir, 'test_preds/mnist_simple_False_easiest_False.csv',sep='')
)
rps_easiest$exp <- "Simple-Easiest"

rps_middleout <- read_csv(
  paste(data_dir, 'test_preds/mnist_simple_False_middleout_False.csv',sep='')
)
rps_middleout$exp <- "Simple-MiddleOut"

rps_ordered <- read_csv(
  paste(data_dir, 'test_preds/mnist_ordered_False_easiest_False.csv',sep='')
)
rps_ordered$exp <- "Ordered"


test_diffs <- read_csv(
  paste(data_dir, 'test_preds/mnist_diffs_test.csv',sep=''),
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


ggplot(Z,aes(x=epoch,y=mean,linetype=exp, color=as.factor(bin))) + 
  geom_line() 

Z[which(Z$exp=='baseline' & Z$epoch==172),]
Z[which(Z$exp=='Simple-Easiest' & Z$epoch==177),]
Z[which(Z$exp=='Theta' & Z$epoch==158),]
Z[which(Z$exp=='Simple-MiddleOut' & Z$epoch==157),]
Z[which(Z$exp=='Ordered' & Z$epoch==167),]


D.irt.hard <- read_csv(paste(data_dir, 'irt-cl-hard-', exp_type, '-1000.log', sep=''),
                  col_names = c('train_size', 'train_acc', 'val_acc', 'test_acc', 'theta'),
                  skip=num_skip, n_max=200)
D.irt.hard$epoch <- c(1:200)
D.irt.hard$exp <- 'irt.hard'
