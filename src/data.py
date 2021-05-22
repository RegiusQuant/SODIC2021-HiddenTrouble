import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


def load_data(train_path, test_path, valid_rate=0.2):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_data, valid_data, test_data = [], [], []

    for _, data in tqdm(train_df.iterrows(), desc="Load Train Data"):
        text_id, *text, label = data
        text = "\t".join(map(str, text))
        
        if random.random() > valid_rate:
            train_data.append((text_id, text, int(label)))
        else:
            valid_data.append((text_id, text, int(label)))

    for _, data in tqdm(test_df.iterrows(), desc="Load Test Data"):
        text_id, *text = data
        text = "\t".join(map(str, text))
        test_data.append((text_id, text, 0))

    return train_data, valid_data, test_data


class SodicDataset(Dataset):

    def __init__(self, input_data):
        self.input_data = input_data
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

    def __getitem__(self, idx):
        return self.input_data[idx]

    def __len__(self):
        return len(self.input_data)

    def collate_fn(self, batch):
        texts = [x[1] for x in batch]
        bert_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")

        labels = torch.LongTensor([x[2] for x in batch])

        return {
            "bert_inputs": bert_inputs,
            "labels": labels
        }
