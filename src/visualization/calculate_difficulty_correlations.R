# calculate correlations

library(stringr)
library(readr)
library(dplyr)

library(tokenizers)
library(tidytext)

library(Hmisc)

# irt vs length
# irt vs wr
# wr vs length

datadir <- "G:/My Drive/research/data"

datasets <- c(
  "SST-2", 
  "MRPC",
  "QNLI",
  "QQP",
  "RTE",
  "MNLI"
)

colnames <- c(
  "sentence",
  "#1 String",
  "question",
  "question1",
  "sentence1",
  "sentence1"
)


names(colnames) <- datasets

# load IRT diffs (1 file per data set) 
# define a function here
loadIRTDiffs <- function(datasetName){
  if (datasetName == "SST-2"){
    datasetName <- "SST"
  }
  fname <- str_glue("{datadir}/glue_diffs/{datasetName}.rp.diffs")
  D <- read_csv(fname,
                col_names = c("id", "diff"))
  
  return(D)
  
}

# load training data set and grab the text (function here for 1 data set) 
loadTrainingData <- function(datasetName){
  fname <- str_glue("{datadir}/glue_data/{datasetName}/train.tsv")
  D <- read_tsv(fname, quote="")
  
  return(D)
  
}


# working example: MNLI
ds <- colnames[2]

calculateCorrelations <- function(ds){
  print(ds)
  dsname <- names(ds)[[1]]
D.data <- loadTrainingData(dsname)
D.diffs <- loadIRTDiffs(dsname)
cname <- as.String(ds)


my_cols <- c(cname, "diff")

D <- D.data %>%
  bind_cols(D.diffs) %>%
  select(one_of(my_cols))


names(D)[1] <- "sent"

D<- D %>%
  mutate(s_tok = tokenize_words(sent)) %>%
  mutate(id = row_number()) %>%
  rowwise() %>%
  mutate(s_length = length(s_tok))

# calculate sentence length diff


# calculate word rarity diff
word_counts <- D %>%
  unnest_tokens(word, sent) %>%
  count(word)

word_rarities <- word_counts %>%
  mutate(total = sum(word_counts$n)) %>%
  mutate(p = n/total)

sentence_words <- D %>%
  unnest_tokens(word, sent) %>%
  inner_join(word_rarities,
             by="word") %>%
  group_by(id) %>%
  mutate(wr = -1 * sum(log(p))) %>%
  slice(1) %>%
  select(diff, s_length, wr)


res <- rcorr(as.matrix(sentence_words[,-1]), type="spearman")
res
}

# calculate correlations


calculateCorrelations(colnames[1])
calculateCorrelations(colnames[2])
calculateCorrelations(colnames[3])
calculateCorrelations(colnames[4])
calculateCorrelations(colnames[5])
calculateCorrelations(colnames[6])

#do not run
#sapply(colnames, calculateCorrelations, USE.NAMES = T, simplify = F)
