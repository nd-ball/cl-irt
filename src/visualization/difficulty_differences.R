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

snli[order(-snli$rank_difference),]
print(
  xtable(
    snli[ordering.snli[1:20],c(1,2,3,6,7,8)],
    caption="Examples from snli with the largest disparities",
    label="tab:differences_snli"
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

print(
  xtable(
    sstb[ordering.sstb[1:20],c(1,2,6,7,8)],
    caption="Examples from sstb with the largest disparities",
    label="tab:differences_sstb"
  ), 
  include.rownames = FALSE
)

