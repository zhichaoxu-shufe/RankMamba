import os
import sys
import json
import copy

import torch


def save_model(model, save_dest):
    if not os.path.isdir(save_dest):
        os.mkdir(save_dest)

    from transformers import PreTrainedModel
    if isinstance(model, PreTrainedModel):
        model.save_pretrained(save_dest)
    else:
        model_ = copy.deepcopy(model)
        from peft import PeftModel
        if isinstance(model.base_model, PeftModel):
            model_.base_model = model_.base_model.merge_and_unload()

        torch.save(model_.state_dict(), os.path.join(save_dest, 'pytorch_model.bin'))

def load_from_trained(args, initialized_model=None):
    from transformers import AutoTokenizer
    if "pytorch_model.bin" in args.model_ckpt:
        initialized_model.load_state_dict(torch.load(args.model_ckpt))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        return tokenizer, initialized_model
    else:
        del initialized_model
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        return tokenizer, model

def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

def load_lce_triples(fname):
    res = []
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip().split("\t")
            res.append(line)
    return res

def get_eval_batch(collection, batch_doc_ids):
    return [collection[i] for i in batch_doc_ids]

def read_validset(fname):
    res = []
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip().split("\t")
            res.append(line)
    return res

def read_ranklist(fname, cutoff=100):
    ranklist = {}
    with open(fname, "r") as fin:
        for line in fin:
            qid, pid, rank, score = line.strip().split("\t")
            if qid not in ranklist:
                ranklist[qid] = []
            if int(rank) > cutoff:
                continue
            else:
                ranklist[qid].append(pid)
    fin.close()
    return ranklist

def read_trec_dl_ranklist(fname, cutoff=100):
    ranklist = {}
    with open(fname, "r") as fin:
        for line in fin:
            qid, _, pid, rank, score, dummy = line.strip().split(" ")
            if qid not in ranklist:
                ranklist[qid] = []
            if int(rank) > cutoff:
                continue
            else:
                ranklist[qid].append(pid)
    fin.close()
    return ranklist

def read_qrels(fname):
    qrels_map = {}
    with open(fname, "r") as fin:
        for line in fin:
            qid, _, pid, relevance = line.strip().split("\t")
            if qid not in qrels_map:
                qrels_map[qid] = {}
            qrels_map[qid][pid] = int(relevance)
    fin.close()
    return qrels_map

def read_qrels_trec_dl(fname):
    qrels_map = {}
    with open(fname, "r") as fin:
        for line in fin:
            qid, _, pid, relevance = line.strip().split(" ")
            if qid not in qrels_map:
                qrels_map[qid] = {}
            qrels_map[qid][pid] = int(relevance)
    fin.close()
    return qrels_map

def read_queries(fname, q_prefix="[query]"):
    test_queries = {}
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip().split("\t")
            test_queries[line[0]] = q_prefix + line[1]
    fin.close()
    return test_queries

def configure_eval_dataset(name, is_autoregressive=False):
    if name == "dev":
        input_dir = "/home/zhichao/msmarco_document/"
        test_queries = os.path.join(input_dir, "queries.dev.tsv")
        bm25_ranklist = os.path.join(input_dir, "bm25-ranklist.dev.run")
        ranklist = read_ranklist(bm25_ranklist)
        if is_autoregressive:
            test_queries = read_queries(test_queries, "Query: ")
        else:
            test_queries = read_queries(test_queries, "[query]")

    elif name == "dl19":
        input_dir = "/home/zhichao/msmarco_document/trec_dl"
        qrels = os.path.join(input_dir, "2019qrels-docs.txt")
        judged = set()
        with open(qrels, "r") as fin:
            for line in fin:
                line = line.strip().split(" ")
                judged.add(line[0])
        test_queries = os.path.join(input_dir, "msmarco-test2019-queries.tsv")
        if is_autoregressive:
            test_queries = read_queries(test_queries, "Query: ")
        else:
            test_queries = read_queries(test_queries, "[query]")
        test_queries = {k: v for k, v in test_queries.items() if k in judged}
        bm25_ranklist = os.path.join(input_dir, "run.msmarco-v1-doc.bm25-doc-tuned.dl19.txt")
        ranklist = read_trec_dl_ranklist(bm25_ranklist)
    
    elif name == "dl20":
        input_dir = "/home/zhichao/msmarco_document/trec_dl"
        qrels = os.path.join(input_dir, "2020qrels-docs.txt")
        judged = set()
        with open(qrels, "r") as fin:
            for line in fin:
                line = line.strip().split(" ")
                judged.add(line[0])
        test_queries = os.path.join(input_dir, "msmarco-test2020-queries.tsv")
        if is_autoregressive:
            test_queries = read_queries(test_queries, "Query: ")
        else:
            test_queries = read_queries(test_queries, "[query]")
        test_queries = {k: v for k, v in test_queries.items() if k in judged}
        bm25_ranklist = os.path.join(input_dir, "run.msmarco-v1-doc.bm25-doc-tuned.dl20.txt")
        ranklist = read_trec_dl_ranklist(bm25_ranklist)
        ranklist = {k: v for k, v in ranklist.items() if k in judged}
    
    return test_queries, ranklist

