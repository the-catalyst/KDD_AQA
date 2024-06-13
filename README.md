# AQA Document Retrieval

### CS 145 Final Project - Spring 2024

Team: Tomasz Jezak, Kiel Messinger, Daniel Wang,  John Reinker, Spencer Sang, Siddhant Kharbanda

Welcome to the AQA (Academic Question Answering) project. This project aims to develop a model that can retrieve the top 20 most relevant academic papers to answer technical questions. Below you will find detailed instructions on setting up the environment, training the model, and evaluating the results.

## Table of Contents
1. [Setup](#setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Training Instructions](#training-instructions)
   - [Stage 1 Training](#stage-1-training)
   - [Stage 2 Training](#stage-2-training)

# Setup

To get started, you need to set up the conda environment. This can be done by running the following command:

```bash
conda env create -n kdd_aqa --file=environment.yml
```

This will create a conda environment named kdd_aqa using the dependencies listed in the environment.yml file.

# Data Preprocessing

Before training the model, you need to preprocess the data. The preprocessing steps include cleaning the data and preparing it for training. All data files are inside the `./data` directory.

### Data Cleaning (`src/preprocessing.ipynb`)
1. Clean Data:

- Load and clean the training and validation data using BeautifulSoup to remove HTML tags and regular expressions to clean text.

- Store the cleaned questions, body text, and paper IDs for both training and validation sets.

2. Process Paper Data:

- Load the paper data and clean the titles and abstracts.

- Create mappings between paper IDs and their corresponding cleaned text.

3. Save Processed Data:

- Save the cleaned and processed data into appropriate files for later use in training and validation.

### Tokenization (`src/create_tokenized_data.py`)
1. Tokenize Data:

- Use the transformers library to tokenize the cleaned text data.

- Tokenize training questions and bodies, validation questions and bodies, and pretraining paper titles and abstracts.

2. Save Tokenized Data:

- Save the tokenized data into NumPy arrays for efficient loading during the training process.

# Training Instructions

## Stage 1 Training

For single GPU training in Stage 1, use the following command:
```bash
accelerate launch --config_file ./config.yaml --main_process_port 32342 --gpu_ids 0 --num_processes 1 src/main.py --task pretrain --version Stage1 --ep 50 --num-negs 0 --label-pool-size 500 --bs 256 --add-dual-loss --lr 5e-5 --cl-start 5 --cl-update 5 
```
### Explanation of parameters:

--config_file ./config.yaml: Path to the configuration file.

--main_process_port 32342: Port number for the main process.

--gpu_ids 0: Specifies the GPU ID to use.

--num_processes 1: Number of processes to run.

--task pretrain: Specifies the task for Stage 1 pretraining.

--version Stage1: Version identifier for this stage.

--ep 50: Number of epochs.

--num-negs 0: Number of negative samples.

--label-pool-size 500: Size of the label pool.

--bs 256: Batch size.

--add-dual-loss: Adds dual loss function.

--lr 5e-5: Learning rate.

--cl-start 5: Contrastive learning start epoch.

--cl-update 5: Contrastive learning update interval.


## Stage 2 Training
For single GPU training in Stage 2, use the following command:

```bash
accelerate launch --config_file ./config.yaml --main_process_port 32342 --gpu_ids 0 --num_processes 1 src/main.py --task train --version Stage2 --ep 50 --num-negs 1 --label-pool-size 210 --bs 64 --lr 5e-5 --cl-start 5 --cl-update 5 --fill-batch-gap 0 --lm model_stage1.pth --load-from-pt
```
### Explanation of additional parameters:

--task train: Specifies the task for Stage 2 training.

--version Stage2: Version identifier for this stage.

--num-negs 1: Number of negative samples.

--label-pool-size 210: Size of the label pool.

--fill-batch-gap 0: Parameter for batch gap filling.

--lm model_stage1.pth: Path to the model checkpoint from Stage 1.

--load-from-pt: Load model parameters from a pre-trained checkpoint.
