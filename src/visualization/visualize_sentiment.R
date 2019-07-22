library(tidyverse)

# sstb
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

D.ordered <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-ordered-easiest.log',sep=''), 
                      col_names = c('a','b','train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                      skip=num_skip, n_max=100) 
D.ordered$epoch <- c(1:100) 
D.ordered$exp <- 'ordered'


D.middleout <- read_csv(paste(data_dir,exp_type, '_cl_not_balanced-simple-middleout.log',sep=''), 
                        col_names = c('a','b','train_size', 'train_acc', 'test_loss', 'test_acc', 'theta'),
                        skip=num_skip, n_max=100) 
D.middleout$epoch <- c(1:100) 
D.middleout$exp <- 'middleout'

D <- rbind(D.baseline, D.irt, D.easiest, D.middleout, D.ordered)
filter <- D %>%
  group_by(exp) %>%
  summarize(max=max(test_loss)) 

max_epochs <- merge(D,filter, by.x=c('exp','test_loss'), by.y=c('exp','max'))

png("../../reports/figures/cl_irt_sstb.png", width=1100, height=700)
ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/67348, color=exp),D, linetype=2) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(56,293,160,362,432),]) + 
  theme_minimal() + 
  ggtitle("Comaprison of CL Strategies: SSTB") + 
  ylab("Test accuracy") + 
  xlab("Epoch") + 
  scale_color_discrete(name='Experiment',
                       breaks=c('baseline', 'easiest', 'irt', 'middleout', 'ordered'),
                       labels=c('Baseline', 'EasyFirst', 'Theta', 'MiddleOut','Ordered'))
dev.off()


