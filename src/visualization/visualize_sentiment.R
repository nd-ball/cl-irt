library(tidyverse)

D <- read_csv('../../data/interim/compiled_results.csv', 
              col_names = c('exp', 'epoch', 'train_size', 'acc_train', 'acc_dev', 'acc_test')) 

# this is too messy, so break it down
ggplot(D, aes(x=train_size, y=acc_test, color=exp))  + 
  geom_point()

# baseline vs simple (balanced)
exp.include <- c('baseline_False_easiest_False', 'simple_True_easiest_False', 'simple_False_easiest_False', 
                 'simple_True_middleout_False', 'simple_False_middleout_False')
D.plot <- D[which(D$exp %in% exp.include),]
ggplot(D.plot, aes(x=epoch, y=acc_dev, color=exp))  + 
  geom_line() + 
  geom_hline(yintercept=max(D.plot[which(D.plot$exp == 'baseline_False_easiest_False'),]$acc_dev), color='black')

D[which(D$acc_dev == max(D$acc_dev)),]

z <- D %>%
  group_by(exp) %>%
  summarize(max=max(acc_dev)) 

D[which(D$acc_dev %in% z$max ),]

inner_join(D, z, by=c('exp', 'acc_dev' = 'max'))
