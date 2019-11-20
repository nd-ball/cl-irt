library(tidyverse)

# sstb
data_dir <- 'G:/My Drive/2019/research/projects/cl_irt/sstb_run3/'
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

which(D$exp=='baseline' & D$epoch==96)
which(D$exp=='irt' & D$epoch==12)
which(D$exp=='naacl-linear-easiest-irt' & D$epoch==51)
which(D$exp=='naacl-linear-easiest-length' & D$epoch==66)
which(D$exp=='naacl-root-easiest-irt' & D$epoch==123)
which(D$exp=='naacl-root-easiest-length' & D$epoch==158)

cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

png("../../reports/figures/cl_irt_sstb.png", width=800, height=400)
ggplot(D, aes(x=epoch, y=test_acc*100, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/673.48, color=exp),D, linetype=2) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(96,212,451,866,723,1158),]) + 
  theme_minimal() + 
  ggtitle("Comaprison of CL Strategies: SSTB") + 
  ylab("Test accuracy") + 
  xlab("Epoch") + 
  ylim(60,100) +
  scale_color_manual(name='Experiment',
                       breaks=c('baseline', 'naacl-linear-easiest-irt', 'naacl-linear-easiest-length',
                                'irt', 'naacl-root-easiest-irt', 'naacl-root-easiest-length'),
                       labels=c('Baseline', 'CB-L-IRT', 'CB-L-Length', 
                                'DDaCLAE', 'CB-R-IRT', 'CB-R-Length'),
                     values=cbbPalette)
dev.off()


######### Table to show how much data was required to get to best acc #################
sum(D[which(D$exp=='baseline'&D$epoch <= max_epochs[which(max_epochs$exp=='baseline'),]$epoch),]$train_size)
sum(D[which(D$exp=='irt'&D$epoch <= max_epochs[which(max_epochs$exp=='irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='irt'&D$epoch <= 12),]$train_size)
sum(D[which(D$exp=='naacl-linear-easiest-irt'&D$epoch <= max_epochs[which(max_epochs$exp=='naacl-linear-easiest-irt'),]$epoch),]$train_size)
sum(D[which(D$exp=='naacl-root-easiest-irt'&D$epoch <= max_epochs[which(max_epochs$exp=='naacl-root-easiest-irt'),]$epoch),]$train_size)
