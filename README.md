# README

This repository contains a implementation of our "WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding" .

## Environment Setup

1. Pytorch 1.12.1
2. Python 3.7.15

### Run

The implementation of embedding generate (```emb.py```)、node classification task(```nc.py```) and link prediction task (```lp.py```); 

## Example to run the codes

### step 1: fine-tune language model and generate embeddings

```python
python emb.py
```

### step 2: node classification task

```python
python nc.py
```

### step 3: link prediction task

```python
python lp.py
```

## Citation

If you find the code useful, please consider citing the following paper:

```
@inproceedings{tan2023walklm,
  title={WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding},
  author={Tan, Yanchao and Zhou, Zihao and Lv, Hang and Liu, Weiming and Yang, Carl},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

