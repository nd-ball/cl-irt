import torch
from models.abstract_model import AbstractModel
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleNNModel(AbstractModel):

    def __init__(self, config):
        self.config = config
        self.device = self.config["trainer"]["device"]
        self.model = Net()
        self.model.to(self.device) 

    def forward(self, inputs, labels=None):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(inputs)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        if labels is not None:
            outputs = self.model(**input_batch, labels=labels)
        else:
            outputs = self.model(**input_batch)
        return outputs
