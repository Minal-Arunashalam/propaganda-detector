#this class loads a pretrained model and uses a custom multi label implentation 
from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch

class PropagandaModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0,:]  
        logits = self.classifier(cls)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss, logits

        return logits
