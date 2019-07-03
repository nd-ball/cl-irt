library(tidyverse)

data_dir <- 'G:/My Drive/data/curriculum_learning_irt/'
D.baseline <- read_csv(paste(data_dir,'cifar_baseline.log',sep=''), 
              col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
              skip=2, n_max=20) 
D.baseline$epoch <- c(1:20) 
D.baseline$exp <- 'baseline'


D.easiest <- read_csv(paste(data_dir,'cifar_cl_simple_not_balanced-easiest.log',sep=''), 
                       col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
                       skip=2, n_max=20) 
D.easiest$epoch <- c(1:20) 
D.easiest$exp <- 'easiest'

D.middleout <- read_csv(paste(data_dir,'cifar_cl_simple_not_balanced-middleout.log',sep=''), 
                      col_names = c('train_size', 'train_acc', 'test_loss', 'test_acc'),
                      skip=2, n_max=20) 
D.middleout$epoch <- c(1:20) 
D.middleout$exp <- 'middleout'


# this is too messy, so break it down
ggplot(D.baseline, aes(x=epoch, y=test_acc, color='red'))  + 
  geom_line() + 
  geom_line(aes(epoch, test_acc, color='blue'),D.easiest) + 
  geom_line(aes(epoch, test_acc, color='green'), D.middleout) + 
  geom_line(aes(epoch, train_size / 500, color='blue'), D.easiest)


