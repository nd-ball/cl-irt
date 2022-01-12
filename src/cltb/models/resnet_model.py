import torch
from models.abstract_model import AbstractModel
from PIL import Image
from torchvision import transforms


class ResnetModel(AbstractModel):

    def __init__(self, config):
        self.config = config
        self.device = self.config["trainer"]["device"]
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
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
            outputs = self.model(**inputs, labels=labels)
        else:
            outputs = self.model(**inputs)
        return outputs

# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

