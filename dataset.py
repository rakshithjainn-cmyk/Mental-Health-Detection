# src/dataset.py
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class MHTextDataset(Dataset):
def __init__(self, texts, labels, model_name='bert-base-uncased', max_length=128):
self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
self.texts = texts
self.labels = labels
self.max_length = max_length


def __len__(self):
return len(self.texts)


def __getitem__(self, idx):
text = str(self.texts[idx])
inputs = self.tokenizer(
text,
add_special_tokens=True,
truncation=True,
padding='max_length',
max_length=self.max_length,
return_tensors='pt'
)
item = {k: v.squeeze(0) for k, v in inputs.items()}
if self.labels is not None:
item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
return item
