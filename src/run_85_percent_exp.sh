# setup

source activate dynet2.0

export LD_LIBRARY_PATH=/home/lalor/bin/dynet-base-py3/dynet/build/dynet/:$LD_LIBRARY_PATH
NUMEPOCHS=200

# irt CL

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_snli.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS --p-correct 0.85"

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_sstb.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS --p-correct 0.85"


# irt CL - hard

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_hard_snli.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta-hard --min-train-length 500 --num-epochs $NUMEPOCHS --p-correct 0.85"

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_hard_sstb.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta-hard --min-train-length 500 --num-epochs $NUMEPOCHS --p-correct 0.85"


# irt CL - viz 
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_mnist_1000.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000 --p-correct 0.85"

sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_cifar_1000.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000 --p-correct 0.85"


# irt CL - viz 
sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_mnist_hard_1000.log --wrap="python -u -m models.mnist --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta-hard --ordering easiest --min-train-length 1000 --p-correct 0.85"

sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/85percent/irt_cl_cifar_hard_1000.log --wrap="python -u -m models.cifar --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta-hard --ordering easiest --min-train-length 1000 --p-correct 0.85"

