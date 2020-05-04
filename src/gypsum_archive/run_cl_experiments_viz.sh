# run curriculum learning experiments on gypsum
# setup

NUMEPOCHS=200
# MNIST 
# baseline (all data)
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_baseline.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy baseline"

# baseline (random curriculum)
# simple, balanced
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_simple_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --balanced --random"

# simple, not balanced
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_simple_not_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --random"

# ordered, balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_ordered_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --balanced --random"

# ordered, not balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_ordered_not_balanced-random.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --random"


for o in easiest middleout hardest
do
    for s in simple ordered #balanced 
    do 

        # CL, simple, balanced
        sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_balanced-$s-$o.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --balanced --ordering $o"

        # CL, simple, not balanced
        sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/mnist_cl_not_balanced-$s-$o.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --ordering $o"

    done 

done 

# CIFAR
# baseline (all data)
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_baseline.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy baseline"

# baseline (random curriculum)
# simple, balanced
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_simple_balanced-random.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --balanced --random"

# simple, not balanced
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_simple_not_balanced-random.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --random"

# ordered, balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_ordered_balanced-random.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --balanced --random"

# ordered, not balanced
#sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_ordered_not_balanced-random.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --random"


for o in easiest middleout hardest
do
    for s in simple ordered #balanced 
    do 

        # CL, simple, balanced
        sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_balanced-$s-$o.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --balanced --ordering $o"

        # CL, simple, not balanced
        sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_not_balanced-$s-$o.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --ordering $o"
    done 
done 

# irt CL
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/irt_cl_mnist_1000.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000"

sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/irt_cl_cifar_1000.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000"