from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn
import torch
import math
from torch.nn.init import xavier_uniform_

from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from copy import deepcopy as cp
from transformers import AutoModel
# from load_bert import load_bert_model

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

class ActionPredictor(torch.nn.Module):
    def __init__(self, d_model=768, num_actions=6):
        super(ActionPredictor, self).__init__()
        self.action_predictor = nn.Linear(d_model, num_actions)
        #self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.action_predictor.bias.data.zero_()
        self.action_predictor.weight.data.uniform_(-initrange, initrange)

    def forward(self, state_tensor):
        actions = torch.nn.Softmax(-1)(self.action_predictor(state_tensor))
        return actions


class FineTunedModel(torch.nn.Module):
    def __init__(self, tasks, label_nums, config, pretrained_model_name='microsoft/deberta-v3-base', tf_checkpoint=None, \
                 dropout=0.1):
        super(FineTunedModel, self).__init__()

        self.config = config

        # Load DebertaV3 model
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(dropout)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = SequenceClassificationHead(self.encoder.config.hidden_size, label_nums[task.lower()])
            self.output_heads[task.lower()] = decoder

        self.state_predictor = nn.Linear(config.hidden_size, 1)

    def forward(self, task_name, src=None, mask=None, token_type_ids=None, pooled_output=None, discriminator=False,
                output_hidden_states=True):

        if pooled_output is None:
            outputs = self.encoder(
                input_ids=src,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states
            )

            if output_hidden_states:
                hidden_states = outputs.hidden_states
                pooled_output = hidden_states[-1][:, 0]  # Assuming DeBERTa's last layer pooled output
            else:
                pooled_output = outputs.last_hidden_state[:, 0]
        else:
            # If pooled_output is provided, we skip the encoder
            hidden_states = None

        out = self.output_heads[task_name.lower()](self.drop(pooled_output))

        if task_name == 'sts-b':
            out = nn.ReLU()(out)

        model_state = self.state_predictor(pooled_output).reshape(1, -1)

        features = None

        if output_hidden_states and not discriminator and hidden_states is not None:
            features = torch.cat(hidden_states[1:-1], dim=0).view(self.config.num_hidden_layers - 1,
                                                                  -1,
                                                                  src.size()[1],
                                                                  self.config.hidden_size)[:, :, 0]

        if discriminator:
            return (out, pooled_output, model_state, pooled_output)
        else:
            return (out, features, model_state, pooled_output)
