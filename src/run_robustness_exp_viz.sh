# run curriculum learning experiments on gypsum
# setup

NUMEPOCHS=200
# MNIST 


for k in 1 2 3 4 5
do 
  # baseline (all data)
  sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_baseline-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy baseline"

  # baseline (random curriculum)
  # simple, balanced
  sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_cl_simple_balanced-random-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --balanced --random"

  # simple, not balanced
  sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_cl_simple_not_balanced-random-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --random"

  # ordered, balanced
  sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_cl_ordered_balanced-random-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --balanced --random"

  # ordered, not balanced
  sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_cl_ordered_not_balanced-random-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --random"


    for o in easiest middleout 
    do
        for s in simple ordered #balanced 
        do 

            # CL, simple, balanced
            sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_cl_balanced-$s-$o-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --balanced --ordering $o"

            # CL, simple, not balanced
            sbatch -p 1080ti-short --gres=gpu:1 --mem=90gb --output=logs/robustness/mnist_cl_not_balanced-$s-$o-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --ordering $o"

        done 

    done 

    # CIFAR
    # baseline (all data)
    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_baseline-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy baseline"

    # baseline (random curriculum)
    # simple, balanced
    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_cl_simple_balanced-random-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --balanced --random"

    # simple, not balanced
    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_cl_simple_not_balanced-random-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --random"

    # ordered, balanced
    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_cl_ordered_balanced-random-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --balanced --random"

    # ordered, not balanced
    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_cl_ordered_not_balanced-random-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy ordered --random"


    for o in easiest middleout 
    do
        for s in simple ordered #balanced 
        do 

            # CL, simple, balanced
            sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_cl_balanced-$s-$o-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --balanced --ordering $o"

            # CL, simple, not balanced
            sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/cifar_cl_not_balanced-$s-$o-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $s --ordering $o"
        done 
    done 

    # irt CL
    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/irt_cl_mnist-$k.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000"

    sbatch -p 1080ti-long --gres=gpu:1 --mem=90gb --output=logs/robustness/irt_cl_cifar-$k.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000"
done 
