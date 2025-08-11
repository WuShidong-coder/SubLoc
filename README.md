# Sequence to Location: Protein Subcellular Localization Driven by Deep Pretrained Language Model

This repository contains the source code and data used in our paper:  
**"Sequence to Location: Protein Subcellular Localization Driven by Deep Pretrained Language Model"**  
Submitted to *IEEE Transactions on Computational Biology and Bioinformatics (TCBB)*.

## Overview

We propose **SubLoc**, a novel deep learning framework that combines **pretrained protein language models (e.g., ProtT5-XL-U50)** with **BiGRU**, **attention mechanisms**, and **graph convolutional networks (GCN)** to predict protein subcellular localization. Our model leverages both sequence embeddings and residue-level structural contact maps derived from **AlphaFold2** to capture rich contextual and spatial information.


### 1. Get Protein Embeddings

Use ProtTrans (ProtT5) to generate sequence embeddings for your .fasta files.

Alternatively, precomputed embeddings can be provided.

### 2. Setup environment

If you are using conda, you can install all dependencies like this. Otherwise, look at the setup below.

```
conda env create -f environment.yml
conda activate bio
```

### 3.1 Training
```
python train.py --config configs/bio_attention.yaml
```


## Setup

Python 3 dependencies:

- pytorch
- biopython
- h5py
- matplotlib
- seaborn
- pandas
- pyaml
- torchvision
- sklearn

You can use the conda environment file to install all of the above dependencies. Create the conda environment `bio` by
running:
```
conda env create -f environment.yml
```
