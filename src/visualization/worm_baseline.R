# load worm results on baseline model


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





