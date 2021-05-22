import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from transformers import BertModel


class BertFinetuner(pl.LightningModule):

    def __init__(self):
        super(BertFinetuner, self).__init__()

        self.bert = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, batch):
        outputs = self.bert(**batch["bert_inputs"])["pooler_output"]
        logits = self.linear(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["labels"]
        loss = self.criterion(logits, labels)

        _, preds = torch.max(logits, 1)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["labels"]

        _, preds = torch.max(logits, 1)
        self.valid_acc(preds, labels)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        logits = self(batch)
        _, preds = torch.max(logits, 1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

