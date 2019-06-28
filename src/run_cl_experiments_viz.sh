# run curriculum learning experiments on gypsum
# setup

# MNIST 
# baseline (all data)
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_baseline.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy baseline"

# baseline (random curriculum)
# simple, balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_simple_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy simple --balanced --random"

# simple, not balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_simple_not_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy simple --random"

# ordered, balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_ordered_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy ordered --balanced --random"

# ordered, not balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_ordered_not_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy ordered --random"


for o in easiest middleout hardest
do

    # CL, simple, balanced
    sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_simple_balanced-$o.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy simple --balanced --ordering $o"

    # CL, simple, not balanced
    #sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_simple_not_balanced-$o.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy simple --ordering $o"

    # CL, ordered, balanced
    sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_ordered_balanced-$o.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy ordered --balanced --ordering $o"

    # CL, ordered, not balanced
    #sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_ordered_not_balanced-$o.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs 20 --strategy ordered --ordering $o"

done 

