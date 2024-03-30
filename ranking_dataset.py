import os
import sys
import copy

import torch

from utils import flatten_concatenation

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer, args):
        self.data = input_data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            "input_text": self.data[index][0],
            "label": self.data[index][1],
        }
    
    def collate_fn(self, batch):
        # enable smart batching
        batch_text = [row["input_text"] for row in batch]
        batch_label = [row["label"] for row in batch]
        tokenized_input = self.tokenizer(
                                batch_text, 
                                padding=True, 
                                truncation="longest_first",
                                max_length=args.max_length, 
                                return_tensors="pt",
                                )
        return {
            "source": tokenized_input,
            "target": torch.tensor(batch_label),
        }

class LCEDatasetMaskedLM(torch.utils.data.Dataset):
    def __init__(self, collection, queries, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collection = collection
        self.queries = queries
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        line = self.dataset[index]
        input_pretokenized = []
        for pid in line[1:]:
            input_pretokenized.append(self.queries[line[0]]+self.sep_token+self.collection[pid])
        return input_pretokenized
    
    def collate_fn(self, batch):
        input_pretokenized = flatten_concatenation(batch)
        tokenized_input = self.tokenizer(input_pretokenized, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokenized_input

class LCEDatasetCausalLM(torch.utils.data.Dataset):
    def __init__(self, collection, queries, dataset, tokenizer, max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collection = collection
        self.queries = queries
        self.sep_token = "\n\n"

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        line = self.dataset[index]
        input_pretokenized = []
        for pid in line[1:]:
            document = self.tokenizer(self.collection[pid], truncation=True, max_length=self.max_length - 50)  # hardcoded
            truncated_document = self.tokenizer.decode(document.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            input_pretokenized.append(truncated_document+self.sep_token+self.queries[line[0]]+self.tokenizer.eos_token)
        return input_pretokenized
    
    def collate_fn(self, batch):
        input_pretokenized = flatten_concatenation(batch)
        tokenized_input = self.tokenizer(input_pretokenized, padding=True, truncation=False, return_tensors="pt")
        return tokenized_input

class LCEDatasetSeq2SeqLM(torch.utils.data.Dataset):
    def __init__(self, collection, queries, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collection = collection
        self.queries = queries
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token
        self.bos_token, self.bos_token_id = tokenizer.pad_token, tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        line = self.dataset[index]
        input_pretokenized = []
        for pid in line[1:]:
            input_pretokenized.append(self.bos_token+self.queries[line[0]]+self.sep_token+self.collection[pid])
        return input_pretokenized
    
    def collate_fn(self, batch):
        input_pretokenized = flatten_concatenation(batch)
        tokenized_input = self.tokenizer(input_pretokenized, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokenized_input