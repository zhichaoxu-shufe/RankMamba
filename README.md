This is the code base for our technical report <b>RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers</b>, https://arxiv.org/abs/2403.18276

#### Required packages
install required packages @requirements.txt, note `CUDA Version > 12.1` and install corresponding `torch` version

To run Mamba models, following instructions at the Mamba Repo https://github.com/state-spaces/mamba to install
```
mamba-ssm==1.2.0
causal-conv1d==1.2.0
```

Note the following packages
```
transformers >= 4.39.0  # required to use mamba models
flash-attn >= 2.5.6 # required to reproduce the flash attention experiments
pyserini >= 0.24.0 # required for evaluation with trectools
```

#### Dataset Preprocessing
Download the processed document collection, BM25 ranklist and sampled negatives from [this google drive link](https://drive.google.com/drive/folders/1-sHp9xTi5HdiVTHLcB7Uf00g3bpGN4F_?usp=sharing)


#### Training
Specify the dataset directory with `--input_dir`

Sample command to train encoder-only models
```
CUDA_VISIBLE_DEVICES=0 python train_document.py \
--model_name_or_path bert-base-uncased \
--tokenizer bert-base-uncased \
--triples train_samples_lce_2x.tsv \
--train_batch_size 8 \
--epochs 1 \
--fp16 \
--pooling_method cls-pooling \
--do_train 
```

Sample command to train decoder-only models
```
CUDA_VISIBLE_DEVICES=0 python train_document.py \
--model_name_or_path facebook/opt-125m \
--tokenizer facebook/opt-125m \
--triples train_samples_lce_2x.tsv \
--train_batch_size 8 \
--epochs 1 \
--flash_attention \
--pooling_method eos-pooling \
--do_train
```

Sample command to train mamba models 
```
CUDA_VISIBLE_DEVICES=0 python train_document.py \
--model_name_or_path state-spaces/mamba-130m-hf \
--tokenizer state-spaces/mamba-130m-hf \
--triples train_samples_lce_2x.tsv \
--train_batch_size 8 \
--epochs 1 \
--flash_attention \
--do_train
```


#### Load Trained Checkpoint and Inference
The current implementation is to initialize the un-trained model class (in `model.py`) and load the trained model weight from existing `pytorch_model.bin` file, change the file path accordingly in `utils.py` before trying to do inference
```
CUDA_VISIBLE_DEVICES=0 python train_document.py \
--model_name_or_path state-spaces/mamba-130m-hf \
--tokenizer state-spaces/mamba-130m-hf \
--load_from_trained \
--model_ckpt {path to your pytorch_model.bin} \
--do_eval \
--eval_dataset dl19,dl20,dev \
--ranklist firstp.run 
```
The saved ranklist will be named as `{model_name_or_path}_firstp.run`

We use Pyserini integrated evaluation
```
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-doc {your_dl19_ranklist}
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-doc {your_dl20_ranklist}
python -m pyserini.eval.trec_eval -c -M 100 -m recip_rank msmarco-doc-dev {your_dev_ranklist}
```

The trained model checkpoints can be downloaded at [this google drive link](https://drive.google.com/drive/folders/1nA4wz41t1smhdD9lNAK-pd06s6oyFiO6?usp=sharing)


#### Cite the Following Papers
```
@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
@article{xu2024rankmamba,
  title={RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers},
  author={Xu, Zhichao},
  journal={arXiv preprint arXiv:2403.18276},
  year={2024}
}
```
