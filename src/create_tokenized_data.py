"""
For a dataset, create tokenized files in the folder {tokenizer-type}-{maxlen} folder inside the database folder
Sample usage: python -W ignore -u create_tokenized_files.py --data-dir /scratch/Workspace/data/LF-AmazonTitles-131K --tokenizer-type bert-base-uncased --max-length 32 --out_dir .
"""
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import os
import numpy as np
import time
import functools
import argparse


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])


def convert(corpus, tokenizer, max_len, num_threads, bsz=100000): 
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask

@timeit
def tokenize_dump(corpus, tokenization_dir,
                  tokenizer, max_len, prefix,
                  num_threads, batch_size=10000000):
    ind = np.zeros(shape=(len(corpus), max_len), dtype='int64')
    mask = np.zeros(shape=(len(corpus), max_len), dtype='int64')

    for i in range(0, len(corpus), batch_size):
        _ids, _mask = convert(
            corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ind[i: i + _ids.shape[0], :] = _ids
        mask[i: i + _ids.shape[0], :] = _mask
    
    np.save(f"{tokenization_dir}/{prefix}_input_ids.npy", ind)
    np.save(f"{tokenization_dir}/{prefix}_attention_mask.npy", mask)


def main(args):
    data_dir = args.data_dir
    max_train_len = 384
    max_abs_len, max_title_len = 256, 24

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type, do_lower_case=True)

    os.makedirs(f"{data_dir}/pretrain/", exist_ok=True)
    os.makedirs(f"{data_dir}/train/", exist_ok=True)
    os.makedirs(f"{data_dir}/valid/", exist_ok=True)

    train_ques = [x.strip() for x in open(f'{data_dir}/train_ques.raw.txt', "r", encoding="utf-8").readlines()]
    train_body = [x.strip() for x in open(f'{data_dir}/train_body.raw.txt', "r", encoding="utf-8").readlines()]

    print(f"Tokenizing Training Questions + Body with max length {max_train_len}...")
    train_qb_pair = [f"{q} [SEP] {b}" for q, b in zip(train_ques, train_body)]
    tokenize_dump(train_qb_pair, f"{data_dir}/train/", tokenizer, max_train_len, "ques", args.num_threads)

    valid_ques = [x.strip() for x in open(f'{data_dir}/valid_ques.raw.txt', "r", encoding="utf-8").readlines()]
    valid_body = [x.strip() for x in open(f'{data_dir}/valid_body.raw.txt', "r", encoding="utf-8").readlines()]
    
    print(f"Tokenizing Valid Questions + Body with max length {max_train_len}...")
    valid_qb_pair = [f"{q} [SEP] {b}" for q, b in zip(valid_ques, valid_body)]
    tokenize_dump(valid_qb_pair, f"{data_dir}/valid/", tokenizer, max_train_len, "ques", args.num_threads)

    print(len(train_qb_pair), len(valid_qb_pair))
    
    print(f"Tokenizing Pretraining Paper Title + Train Ques + Valid Ques with max length {max_title_len}...")
    pretrain_title = [x.strip() for x in open(f'{data_dir}/pretrain_title.raw.txt', "r", encoding="utf-8").readlines()]
    pretrain_title = pretrain_title + train_ques + valid_ques
    tokenize_dump(pretrain_title, f"{data_dir}/pretrain/", tokenizer, max_title_len, "ques", args.num_threads)

    del train_ques, valid_ques, pretrain_title

    print(f"Tokenizing Pretraining Paper Abstract + Train Body + Valid Body with max length {max_abs_len}...")
    pretrain_abs = [x.strip() for x in open(f'{data_dir}/pretrain_abstract.raw.txt', "r", encoding="utf-8").readlines()]
    pretrain_abs = pretrain_abs + train_body + valid_body
    tokenize_dump(pretrain_abs, f"{data_dir}/pretrain/", tokenizer, max_abs_len, "abstract", args.num_threads)
    
    del train_body, valid_body, pretrain_abs

    print(len(pretrain_title), len(pretrain_abs))

    print(f"Tokenizing Valid Paper Abstract with max length {max_train_len}...")
    valid_abstract = [x.strip() for x in open(f'{data_dir}/papers.raw.txt', "r", encoding="utf-8").readlines()]
    tokenize_dump(valid_abstract, f"{data_dir}/valid/", tokenizer, max_train_len, "abstract", args.num_threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data directory path")
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        help="Tokenizer to use",
        default="bert-base-uncased")
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Number of threads to use",
        default=24)

    args = parser.parse_args()
    main(args)
