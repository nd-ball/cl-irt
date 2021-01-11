# new difficulty histogram plots

# setup
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)

# load difficulties
data_dir <- 'G:/My Drive/research/data/glue_diffs/'

datasets <- c(
  "mnli",
  "mrpc",
  "qnli",
  "qqp",
  "rte",
  "sst"
)

load_data <- function(dname){
  D <- read_csv(
    str_glue("{data_dir}{dname}.rp.diffs"),
    col_names = c("id","diff")
  )
  D$dataset <- dname
  return(D)
}

datasets <- lapply(
  datasets, 
  load_data
)


D <- do.call("bind_rows", datasets)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

p <- D %>%
  ggplot(
    aes(
      diff, 
      color=dataset,
      linetype=dataset
    )
  ) + 
  geom_freqpoly(
    size=1,
    bins=100,
    aes(y = stat(width*density))
  ) + 
  xlab("Difficulty") + 
  ylab("Frequency (%age)") +
  ggtitle("Distribution of Dataset Difficulties: GLUE") +
  scale_colour_manual(values=cbPalette) +
  theme_minimal()


ggsave("journal_plots/glue_diffs.png", p, width=8, height=4)
