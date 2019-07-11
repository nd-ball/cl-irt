library(tidyverse)

data_dir <- 'G:/My Drive/data/curriculum_learning_irt/'
exp_type <- 'sstb'
num_skip <- 18
D.baseline <- read_csv(paste(data_dir, exp_type,'_baseline_easiest.log',sep=''), 
              col_names = c('a','b','train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
              skip=num_skip, n_max=100) 
D.baseline$epoch <- c(1:100) 
D.baseline$exp <- 'baseline'

D.irt <- read_csv(paste(data_dir, 'irt_cl_', exp_type, '.log', sep=''),
                  col_names = c('a','b','train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                  skip=num_skip, n_max=100)
D.irt$epoch <- c(1:100)
D.irt$exp <- 'irt'

D.easiest <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-easiest.log',sep=''), 
                      col_names = c('a','b','train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                      skip=num_skip, n_max=100) 
D.easiest$epoch <- c(1:100) 
D.easiest$exp <- 'easiest'

D.middleout <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-middleout.log',sep=''), 
                        col_names = c('a','b','train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                        skip=num_skip, n_max=100) 
D.middleout$epoch <- c(1:100) 
D.middleout$exp <- 'middleout'



D <- rbind(D.baseline, D.irt, D.easiest, D.middleout)

ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/67348, color=exp),D, linetype=2)


D.ordered <- read_csv(paste(data_dir,'cifar_cl_ordered_not_balanced-easiest.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
                      skip=2, n_max=20) 
D.ordered$epoch <- c(1:20) 
D.ordered$exp <- 'ordered'


# this is too messy, so break it down
ggplot(D.baseline, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(epoch, test_acc, color=exp),D.easiest) + 
  geom_line(aes(epoch, test_acc, color=exp), D.middleout) + 
  geom_line(aes(epoch, train_size / 500, color=exp), D.easiest)

D <- rbind(D.baseline, D.easiest, D.ordered)

ggplot(D, aes(x=epoch, y=test_acc / 100, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size / max(train_size)), D[which(D$exp == 'easiest'),], linetype=2) + 
  theme_minimal() +
  ggtitle('Initial CL Experiment: CIFAR') +
  ylab("Test Set Accuracy") + xlab('Epoch') + 
  scale_color_discrete(name='Experiment',
                       breaks=c('baseline', 'ordered', 'easiest'),
                       labels=c('Baseline', 'Ordered', 'SimpleCL'))
