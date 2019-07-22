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

D.irt <- read_csv(paste(data_dir, 'irt_cl_', exp_type, '.log', sep=''),
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


png("../../reports/figures/cl_irt_cifar.png", width=1100, height=700)
ggplot(D, aes(x=epoch, y=test_acc, color=exp))  + 
  geom_line() + 
  geom_line(aes(x=epoch, y=train_size/500, color=exp),D, linetype=2) + 
  geom_vline(aes(xintercept=epoch, color=exp ), D[c(135,588,343,715,945),]) + 
  theme_minimal() + 
  ggtitle("Comaprison of CL Strategies: CIFAR") + 
  ylab("Test accuracy") + 
  xlab("Epoch") + 
  scale_color_discrete(name='Experiment',
                       breaks=c('baseline', 'easiest', 'irt', 'middleout', 'ordered'),
                       labels=c('Baseline', 'EasyFirst', 'Theta', 'MiddleOut', 'Ordered'))
dev.off() 


# load detailed RP data
rps_baseline <- read_csv(
  paste(data_dir, 'test_preds/cifar_simple_False_easiest_True.csv',sep='')
)

rps_irt <- read_csv(
  paste(data_dir, 'test_preds/cifar_theta_False_easiest_False.csv',sep='')
)

test_diffs <- read_csv(
  paste(data_dir, 'test_preds/cifar_rp_test_snli.csv.diffs',sep=''),
  col_names=c('pairid', 'diff')
)

rps_irt <- merge(rps_irt, test_diffs, by.x='itemID', by.y='pairid')
rps_irt$bin <- 1
rps_irt[which(rps_irt$diff >= -1.2),]$bin <- 2
rps_irt[which(rps_irt$diff >= -0.14),]$bin <- 3
rps_irt[which(rps_irt$diff >= 0.98),]$bin <- 4

table(rps_irt$bin)

Z.irt <- rps_irt %>% 
  group_by(epoch,bin) %>%
  summarize(mean=mean(correct==pred))

rps_baseline <- merge(rps_baseline, test_diffs, by.x='itemID', by.y='pairid')
rps_baseline$bin <- 1
rps_baseline[which(rps_baseline$diff >= -1.2),]$bin <- 2
rps_baseline[which(rps_baseline$diff >= -0.14),]$bin <- 3
rps_baseline[which(rps_baseline$diff >= 0.98),]$bin <- 4

table(rps_baseline$bin)
table(rps_irt$bin)

Z.baseline <- rps_baseline %>% 
  group_by(epoch,bin) %>%
  summarize(mean=mean(correct==pred))

ggplot(Z.irt,aes(x=epoch,y=mean,color=as.factor(bin))) + 
  geom_line() + 
  ggtitle("DCL-IRT") + 
  geom_line(aes(x=epoch, y=mean, color=as.factor(bin)), Z.baseline, linetype=2)


ggplot(Z.baseline,aes(x=epoch,y=mean,color=as.factor(bin))) + 
  geom_line() + 
  ggtitle("Baseline")


z.baseline.all <- rps_baseline %>% 
  group_by(epoch) %>%
  summarize(mean=mean(correct==pred))
z.irt.all <- rps_irt %>% 
  group_by(epoch) %>%
  summarize(mean=mean(correct==pred))

ggplot(z.irt.all, aes(x=epoch,y=mean)) + 
  geom_line() + 
  geom_line(aes(x=epoch,y=mean), z.baseline.all, linetype=2)
