library(tidyverse)

data_dir <- 'G:/My Drive/data/curriculum_learning_irt/'
D.baseline <- read_csv(paste(data_dir,'mnist_baseline.log',sep=''), 
              col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
              n_max=20) 
D.baseline$epoch <- c(1:20) 
D.baseline$exp <- 'baseline'


D.easiest <- read_csv(paste(data_dir,'mnist_cl_simple_not_balanced-easiest.log',sep=''), 
                       col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
                       n_max=20) 
D.easiest$epoch <- c(1:20) 
D.easiest$exp <- 'easiest'

D.middleout <- read_csv(paste(data_dir,'mnist_cl_simple_not_balanced-middleout.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
                      n_max=20) 
D.middleout$epoch <- c(1:20) 
D.middleout$exp <- 'middleout'

D.ordered <- read_csv(paste(data_dir,'mnist_cl_ordered_not_balanced-easiest.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
                      n_max=20) 
D.ordered$epoch <- c(1:20) 
D.ordered$exp <- 'ordered'


D <- rbind(D.baseline, D.easiest, D.ordered)

ggplot(D, aes(x=epoch, y=test_acc / 100, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size / max(train_size)), D[which(D$exp == 'easiest'),], linetype=2) + 
  theme_minimal() +
  ggtitle('Initial CL Experiment: MNIST') +
  ylab("Test Set Accuracy") + xlab('Epoch') + 
  scale_color_discrete(name='Experiment',
                       breaks=c('baseline', 'ordered', 'easiest'),
                       labels=c('Baseline', 'Ordered', 'SimpleCL'))
