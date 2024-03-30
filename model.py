import os
import sys
import warnings

from typing import Optional, Tuple, Union

import torch
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

def pooling(model_output, attention_mask, mode="eos-pooling"):
    token_embeddings = model_output.last_hidden_state  # (bz, seq_len, hidden_dim)

    if mode == 'mean-pooling':
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    elif mode == 'cls-pooling':
        return token_embeddings[:, 0, :]
    elif mode == 'eos-pooling':
        return token_embeddings[:, -1, :]

def configure_lora_model(base_model, args):
    from transformers import PreTrainedModel
    assert isinstance(base_model, PreTrainedModel), "base model has to be a huggingface pretrained model"
    from peft import LoraModel, LoraConfig
    from peft import get_peft_model
    if "opt" in base_model.config._name_or_path.lower():
        target_modules = ["embed_tokens", "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif "pythia" in base_model.config._name_or_path.lower():
        target_modules = ["embed_in", "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "mamba" in base_model.config._name_or_path.lower():
        target_modules = ["embedding", "in_proj", "x_proj", "out_proj"]
    elif "t5" in base_model.config._name_or_path.lower():
        target_modules = ["embedding", "q", "k", "v", "o", "wi", "wo", "dense"]
    elif "deberta" in base_model.config._name_or_path.lower():
        target_modules = ["word_embeddings", "query_proj", "key_proj", "value_proj", "dense"]
    elif "bert" in base_model.config._name_or_path.lower():
        target_modules = ["embedding", "dense", "query", "key", "value"]
    elif "gpt2" in base_model.config._name_or_path.lower():
        target_modules = ["embedding", "c_attn", "c_proj", "c_fc", "dense"]
    else:
        raise Exception("base model for lora finetuning is not defined")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    lora_model = get_peft_model(base_model, config)
    return lora_model


def configure_opt_model(model_name_or_path, tokenizer, args):
    if args.flash_attention:
        base_model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    else:
        base_model = AutoModel.from_pretrained(model_name_or_path)

    if args.lora:  # this indicates you should be using flash attention 2
        base_model = configure_lora_model(base_model, args)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = SequenceRegressionModel(base_model=base_model, config=config, args=args)
    return model

def configure_pythia_model(model_name_or_path, tokenizer, args):
    if args.flash_attention:
        base_model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    else:
        base_model = AutoModel.from_pretrained(model_name_or_path)

    if args.lora:  # this indicates you should be using flash attention 2
        base_model = configure_lora_model(base_model, args)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = SequenceRegressionModel(base_model=base_model, config=config, args=args)
    return model

def configure_gpt2_model(model_name_or_path, tokenizer, args):
    base_model = AutoModel.from_pretrained(model_name_or_path)
    if args.lora:
        base_model = base_model.half()
        base_model = configure_lora_model(base_model, args)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = SequenceRegressionModel(base_model=base_model, config=config, args=args)
    return model

def configure_mamba_model(model_name_or_path, tokenizer, args):
    base_model = AutoModel.from_pretrained(model_name_or_path)
    if args.lora:
        base_model = configure_lora_model(base_model, args)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = SequenceRegressionModel(base_model=base_model, config=config, args=args)
    return model

def configure_t5_model(model_name_or_path, tokenizer, args):
    base_model = AutoModel.from_pretrained(model_name_or_path)
    if args.lora:
        base_model = base_model.to(torch.bfloat16)  # t5 model has to be in bfloat16 dtype
        base_model = configure_lora_model(base_model, args)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = T5RegressionModel(base_model=base_model, config=config, args=args)
    return model

def configure_bert_model(model_name_or_path, tokenizer, args):
    base_model = AutoModel.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.lora:
        base_model = base_model.to(torch.float16)  # we load the base model in fp16 format
        base_model = configure_lora_model(base_model, args)
    model = SequenceRegressionModel(base_model=base_model, config=config, args=args)
    return model

def configure_deberta_model(model_name_or_path, tokenizer, args):
    base_model = AutoModel.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.lora:
        base_model = base_model.to(torch.float16)
        base_model = configure_lora_model(base_model, args)
    model = SequenceRegressionModel(base_model=base_model, config=config, args=args)
    return model

class DistilBertRegressionHead(torch.nn.Module):
    """Head for sequence-level classification tasks"""
    def __init__(self, config, args):
        super().__init__()
        self.pre_classifier = torch.nn.Linear(config.dim, config.dim)
        self.classifier = torch.nn.Linear(config.dim, args.num_labels)
        self.dropout = torch.nn.Dropout(config.seq_classif_dropout)   

    def forward(self, features, **kwargs):
        """features (bx, dim)"""
        pooled_output = self.pre_classifier(features)  # (bs, dim)
        pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits

class BertRegressionHead(torch.nn.Module):
    """Head for sequence-level classification tasks"""
    def __init__(self, config, args):
        super().__init__()
        self.classifier = torch.nn.Linear(config.hidden_size, args.num_labels)
        dropout_rate = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, features, **kwargs):
        """features (bx, dim)"""
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)
        return logits

class DebertaRegressionHead(torch.nn.Module):
    """Head for sequence-level classification tasks"""
    def __init__(self, config, args):
        super().__init__()
        self.classifier = torch.nn.Linear(config.pooler_hidden_size, args.num_labels)
        dropout_rate = 0.1  # hardcoded, TODO: modify this as a hyperparameter
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, features, **kwargs):
        """features (bx, dim)"""
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)
        return logits

class T5RegressionHead(torch.nn.Module):
    """Head for sequence-level classification tasks"""
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.d_model, 1)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, features, **kwargs):
        """features (bx, dim)"""
        pooled_output = self.dropout(features)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits

class OPTRegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.word_embed_proj_dim, 1)
        self.dropout = torch.nn.Dropout(config.dropout)
    
    def forward(self, features, **kwargs):
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)
        return logits

class MambaRegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, features, **kwargs):
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)
        return logits

class PythiaRegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, features, **kwargs):
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)
        return logits

class GPT2RegressionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.n_embd, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, features, **kwargs):
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)
        return logits


class SequenceRegressionModel(torch.nn.Module):
    def __init__(self, base_model=None, config=None, args=None):
        super(SequenceRegressionModel, self).__init__()
        self.base_model = base_model
        self.config = config
        self.args = args

        if "distilbert" in args.model_name_or_path.lower():
            self.regressor = DistilBertRegressionHead(config, args)
        elif "deberta" in args.model_name_or_path.lower():
            self.regressor = DebertaRegressionHead(config, args)
        elif "bert" in args.model_name_or_path.lower():
            self.regressor = BertRegressionHead(config, args)
        elif "mamba" in args.model_name_or_path.lower():
            self.regressor = MambaRegressionHead(config)
        elif "opt" in args.model_name_or_path.lower():
            self.regressor = OPTRegressionHead(config)
        elif "pythia" in args.model_name_or_path.lower():
            self.regressor = PythiaRegressionHead(config)
        elif "gpt2" in args.model_name_or_path.lower():
            self.regressor = GPT2RegressionHead(config)
        else:
            raise Exception("model_name_or_path can not be recognized")

        self.device = self.base_model.device
        self.regressor.device = self.device
        self.regressor = self.regressor.to(self.regressor.device)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = pooling(outputs, attention_mask, self.args.pooling_method) # (bx, dim)
        logits = self.regressor.forward(pooled_output.to(self.regressor.classifier.weight.dtype))
        return logits

class T5RegressionModel(torch.nn.Module):
    def __init__(self, base_model=None, config=None, args=None):
        super(T5RegressionModel, self).__init__()
        self.base_model = base_model
        self.config = config
        self.args = args

        self.regressor = T5RegressionHead(config)
        self.device = self.base_model.device
        self.regressor.device = self.device
        self.regressor = self.regressor.to(self.regressor.device)
    
    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=torch.LongTensor([[0]] * input_ids.shape[0]).to(input_ids.device),
            return_dict=True
        )
        pooled_output = pooling(outputs, attention_mask, self.args.pooling_method) # (bx, dim)
        logits = self.regressor.forward(pooled_output.to(self.regressor.classifier.weight.dtype))
        return logits

class T5EncoderRegressionModel(torch.nn.Module):
    def __init__(self, base_model=None, config=None, args=None):
        super(T5EncoderRegressionModel, self).__init__()
        self.base_model = base_model
        self.config = config
        self.args = args

        self.regressor = T5RegressionHead(config)
        self.device = self.base_model.device
        self.regressor.device = self.device
        self.regressor = self.regressor.to(self.regressor.device)
    
    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = pooling(outputs, attention_mask, self.args.pooling_method) # (bx, dim)
        logits = self.regressor.forward(pooled_output.to(self.regressor.classifier.weight.dtype))
        return logits

if __name__ == "__main__":
    def configure_lora_model(base_model, args):
        from transformers import PreTrainedModel
        assert isinstance(base_model, PreTrainedModel), "base model has to be a huggingface pretrained model"
        from peft import LoraModel, LoraConfig
        from peft import get_peft_model
        if "opt" in base_model.config._name_or_path.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        elif "pythia" in base_model.config._name_or_path.lower():
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "mamba" in base_model.config._name_or_path.lower():
            target_modules = ["Conv1d", "in_proj", "x_proj", "dt_proj", "out_proj"]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        lora_model = get_peft_model(base_model, config)
        return lora_model
    def configure_model(model_name_or_path, tokenizer, args):
        if args.is_autoregressive:
            from model import SequenceRegressionModel, pooling
            from transformers import AutoModel
            base_model = AutoModel.from_pretrained(model_name_or_path)

            base_model = configure_lora_model(base_model, args)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path)
            from model import SequenceRegressionModel
            model = SequenceRegressionModel(tokenizer=tokenizer, base_model=base_model, config=config, args=args)
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)

        return model
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='state-spaces/mamba-130m-hf')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--is_autoregressive', action="store_true")
    parser.add_argument('--pooling_method', type=str, default="eos-pooling")

    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = configure_model(args.model_name_or_path, tokenizer, args)
    from train_document import print_trainable_parameters
    print_trainable_parameters(model)

    from datasets import load_dataset
    dataset = load_dataset("sst2")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fct = torch.nn.BCEWithLogitsLoss()
    import time
    for i, row in enumerate(dataset["train"]):
        input_pretokenized = row["sentence"]
        input_tokenized = tokenizer(input_pretokenized, return_tensors="pt")
        input_tokenized = {k: v.to(DEVICE) for k, v in input_tokenized.items()}
        logits = model.forward(**input_tokenized)
        loss = loss_fct(logits.squeeze(dim=1), torch.FloatTensor([row["label"]]).to(logits.device))
        loss.backward()
        print(loss.item())
        time.sleep(1.)
        optimizer.step()
        optimizer.zero_grad()