library(tidyverse)
library(xtable)

datadir <- "G:/My Drive/2019/research/projects/cl_irt/"

snli <- read_tsv(
  paste(
    datadir, "snli_1.0_train_diff.txt", sep=""
  )
)

snli$diff_rank <- rank(snli$difficulty)
snli$length_rank <- rank(nchar(snli$sentence1) + nchar(snli$sentence2))

snli$rank_difference <- rank(abs(snli$diff_rank - snli$length_rank))
ordering.snli <- order(-snli$rank_difference)

colnames(snli) <- c(
  "Label",
  "Premise",
  "Hypothesis",
  "PairID",
  "Difficulty",
  "RankD",
  "RankL",
  "Difference"
)

snli.output <- snli[ordering.snli[1:20],c(1,2,3,8)]
snli.output$Difference <- round(snli.output$Difference,0)
snli[order(-snli$Difference),]
print(
  xtable(
    snli.output[order(snli.output$Label),],
    caption="Examples from snli with the largest disparities",
    label="tab:differences_snli",
    digits=c(0,0,0,0,0),
    align=c("l","l","p{5cm}","p{5cm}","c")
  ), 
  include.rownames = FALSE
)



sstb <- read_tsv(
  paste(
    datadir, "sstb_train_diff.tsv", sep=""
  )
)

sstb$diff_rank <- rank(sstb$difficulty)
sstb$length_rank <- rank(nchar(sstb$sentence))

sstb$rank_difference <- rank(abs(sstb$diff_rank - sstb$length_rank))
ordering.sstb <- order(-sstb$rank_difference)

colnames(sstb) <- c(
  "Review",
  "Label",
  "PairID",
  "Difficulty",
  "Variance",
  "RankD",
  "RankL",
  "Difference"
  
)

sstb[which(sstb$Label==1),]$Label <- "Pos"
sstb[which(sstb$Label==0),]$Label <- "Neg"
sstb.output <- sstb[ordering.sstb[1:20],c(2,1,8)]
print(
  xtable(
    sstb.output[order(sstb.output$Label, decreasing = T),],
    caption="Examples from sstb with the largest disparities",
    label="tab:differences_sstb",
    digits=c(0,0,0,0),
    align=c("l","l","p{12cm}","c")
  ), 
  include.rownames = FALSE
)

