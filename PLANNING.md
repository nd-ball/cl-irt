# JMLR reject and resubmit planning

## Intro 

DDaCLAE got a reject and resubmit decision from JMLR. 
The biggest items were the small-ish lift and the lack of external baselines. 

To improve the paper I need to put together a curriculum learning code base that implements all of the baselines together.
This can act as a test case for the upcoming all learning is curriculum learning paper. 

## What that means

This repository is getting re-vamped!

It will be built according to a few objects:

1. Data sets
  - every data set will have to include a difficulty attribute
2. Models
3. Trainers
  - curricula have to be implemented as part of the trainers, since certain ones like ddaclae require model access
  
  
What the process will look like:

1. Load a data set into a CLDataset object
2. Load a model
3. Initialize a CLTrainer

cltrainer.train(model, data) 

This way I can implement baselines as trainers:

- no CL
- CB-CL
- Graves's bandit approach
- MentorNet
- Self-paced learning 

Now they'll all be on the same page, and I can just run different models like Electra and bert-large 

### Reporting

Trainers need to log:

- clock time per epoch
- train acc per epoch
- dev acc per epoc
- test acc per epoch
- number of examples per epoch
- final clock time 

## Other work

I also really want to push the fact that learned difficulty is better than heuristics.

- correlations between different types of difficulties
- visual examples of "most difficult" cases according to each type 
