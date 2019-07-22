# run curriculum learning experiments on gypsum
# setup

# MNIST (k=0 to k=60000)
NUMEPOCHS=200

for i in `seq 0 1000 60000`
do 
    sbatch -p m40-short --gres=gpu:1 --mem=90gb --output=logs/k_exp/mnist_cl_not_balanced-simple-easiest-$i.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --ordering easiest --k $i"

    sbatch -p m40-short --gres=gpu:1 --mem=90gb --output=logs/k_exp/mnist_cl_balanced-simple-easiest-$i.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --balanced --ordering easiest --k $i"

done 


# CIFAR (k=0 to k=50000)
NUMEPOCHS=200

for i in `seq 0 1000 50000`
do 

    sbatch -p m40-short --gres=gpu:1 --mem=90gb --output=logs/k_exp/cifar_cl_balanced-simple-easiest-$i.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --balanced --ordering easiest --k $i"

    sbatch -p m40-short --gres=gpu:1 --mem=90gb --output=logs/cifar_cl_not_balanced-simple-easiest-$i.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy simple --ordering easiest --k $i"

done 