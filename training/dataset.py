#this class loads the CSVs and tokenizes the text
import torch
from torch.utils.data import Dataset
import pandas as pd
import json

class PropagandaDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        labels = json.loads(self.df.loc[idx, "labels"])  
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float)
        }
