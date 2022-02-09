from datasets.glue_dataset import GLUEDataset
from datasets.mnist_dataset import MNISTDataset

DATASETS = {
    "GLUE": GLUEDataset,
    "MNIST": MNISTDataset
}
