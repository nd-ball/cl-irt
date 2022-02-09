from models.bert_model import BERTModel
from models.resnet_model import ResnetModel
from models.simple_nn_model import SimpleNNModel
from models.vgg_model import VGGModel

MODELS = {
    "BERT": BERTModel,
    "Resnet": ResnetModel,
    "SimpleNN": SimpleNNModel,
    "VGG": VGGModel
}