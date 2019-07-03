library(tidyverse)

data_dir <- 'G:/My Drive/data/curriculum_learning_irt/'
D <- read_csv(paste(data_dir, 'sstb_results.csv', sep=''), 
              col_names = c('exp', 'epoch', 'train_size', 'acc_train', 'acc_dev', 'acc_test')) 

D.length <- read_csv(
  paste(data_dir, 'sstb_length_baseline.csv', sep=''), 
  col_names = c('exp', 'epoch', 'train_size', 'acc_train', 'acc_dev', 'acc_test')
)
D.length[which(D.length$exp=='simple_False_easiest_False'),]$exp <- 'bylength-easiest'
D.length[which(D.length$exp=='simple_False_hardest_False'),]$exp <- 'bylength-hardest'
D.length[which(D.length$exp=='simple_False_middleout_False'),]$exp <- 'bylength-midout'
# this is too messy, so break it down
ggplot(D.length, aes(x=epoch, y=acc_test, color=exp, linetype=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size / max(train_size)), D.length) +
  geom_line(aes(epoch, acc_test,color=exp),D.plot)

# baseline vs simple (balanced)
exp.include <- c('baseline_False_easiest_False', 'simple_True_easiest_False', 'simple_False_easiest_False', 
                 'simple_True_middleout_False', 'simple_False_middleout_False', 'simple_False_hardest_True')
exp.include <- c('baseline_False_easiest_False', 'simple_False_easiest_False', 
                  'simple_False_middleout_False', 'simple_False_hardest_True')
exp.include <- c('baseline_False_easiest_False', 'ordered_False_easiest_False', 
                 'simple_False_easiest_False')

D.plot <- D[which(D$exp %in% exp.include),]
ggplot(D.plot, aes(x=epoch, y=acc_dev, color=exp))  + 
  geom_line() + 
  #geom_hline(yintercept=max(D.plot[which(D.plot$exp == 'baseline_False_easiest_False'),]$acc_dev), color='black') + 
  geom_line(aes(x=epoch, y=train_size / max(train_size)), D.plot[which(D.plot$exp == 'simple_False_easiest_False'),], linetype=2) + 
  theme_minimal() +
  ggtitle('Initial CL Experiment: SSTB') +
  ylab("Dev Set Accuracy") + xlab('Epoch') + 
  scale_color_discrete(name='Experiment',
                       breaks=exp.include,
                       labels=c('Baseline', 'Ordered', 'SimpleCL'))
D[which(D$acc_dev == max(D$acc_dev)),]

z <- D.plot %>%
  group_by(exp) %>%
  summarize(max=max(acc_dev)) 

D[which(D$acc_dev %in% z$max ),]

inner_join(D.plot, z, by=c('exp', 'acc_dev' = 'max'))
