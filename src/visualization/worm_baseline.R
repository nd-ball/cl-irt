# load worm results on baseline model
library(readr)
library(dplyr)
library(stringr)

fnames  <- list.files(path="../results/",
                      pattern='tracker\\.csv',
                      recursive=TRUE,
                      full.names=TRUE)

fnames.matches  <- grep(
    pattern="baseline",
    fnames,
    value=TRUE
)

final.matches  <- grep(
    pattern="inf",
    fnames.matches,
    value=TRUE
)

# TODOS: 
# load all of the files 
# add ids to indicate data set and run
# identify best test via early stopping
# calculate correlation between initial theta and initial dev acc
# calculate correlation between initial theta and final test acc/dev acc

load_data  <- function(fname){
    D  <- read_csv(fname)
    # get id
    # initial theta
    firstTheta  <- D$theta_hat[1]
    # best test
    earlyStopping  <- which(D$dev_acc == max(D$dev_acc))[1]
    finalTest  <- D$test_acc[earlyStopping]
    finalDev <- D$dev_acc[earlyStopping]
    finalEpoch <- D$epoch[earlyStopping]
    # get necessary labels
    metadata <- str_split(fname, '-', simplify=T)
    dataset <- str_split(metadata[2], '\\/', simplify=T)[2]
    lbound <- metadata[9]
    ubound <- str_split(metadata[10], '\\/', simplify=T)[1]
    print(metadata)
    print(dataset)
    print(lbound)
    print(ubound)
    print(firstTheta)
    print(finalTest)
    result  <- data.frame(
        "dataset" = dataset,
        "lowerBound" = lbound,
        "upperBound" = ubound,
        "theta0" = firstTheta,
        "testAcc" = finalTest,
        "devAcc" = finalDev,
        "stopEpoch" = finalEpoch
    )
    return(result)
}


fnames[1]

load_data(fnames[1])

D <- lapply(final.matches, load_data) %>%
    bind_rows()

D %>%
    group_by(dataset) %>%
    summarize(c = cor(theta0, devAcc))

D %>%
    group_by(dataset) %>%
    summarize(c = cor(theta0, testAcc))

D %>%
    group_by(dataset) %>%
    summarize(c = cor(theta0, stopEpoch))
