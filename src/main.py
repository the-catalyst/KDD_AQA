import os
import math
import torch
import argparse
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from data_utils import *
from KDDDataset import *
from Runner import Runner
from DualEncoder import DualEncoder
from dense_clustering import cluster_dense_embs
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def load_data(args):
    trn_ques, trn_abs, val_ques, val_abs = {}, {}, {}, {}
            
    trn_ques['input_ids'] = torch.from_numpy(np.load(os.path.join(args.data_path, args.task, 'ques_input_ids.npy')))
    trn_ques['attention_mask'] = torch.from_numpy(np.load(os.path.join(args.data_path, args.task, 'ques_attention_mask.npy')))

    trn_abs['input_ids'] = torch.from_numpy(np.load(os.path.join(args.data_path, args.task, 'abstract_input_ids.npy')))
    trn_abs['attention_mask'] = torch.from_numpy(np.load(os.path.join(args.data_path, args.task, 'abstract_attention_mask.npy')))

    val_ques['input_ids'] = torch.from_numpy(np.load(os.path.join(args.data_path, 'valid', 'ques_input_ids.npy')))
    val_ques['attention_mask'] = torch.from_numpy(np.load(os.path.join(args.data_path, 'valid', 'ques_attention_mask.npy')))

    val_abs['input_ids'] = torch.from_numpy(np.load(os.path.join(args.data_path, 'valid', 'abstract_input_ids.npy')))
    val_abs['attention_mask'] = torch.from_numpy(np.load(os.path.join(args.data_path, 'valid', 'abstract_attention_mask.npy')))

    if params.task == "train":
        XY = sp.load_npz(f"{args.data_path}/train_Q_A.npz")
    elif params.task == "pretrain":
        len_X = trn_ques['input_ids'].shape[0]
        data, rows, cols = [1]*len_X, np.arange(len_X), np.arange(len_X)
        XY = sp.csr_matrix((data, (rows, cols)), shape=(len_X, len_X))

    return trn_ques, trn_abs, val_ques, val_abs, XY

def main(params):
    accelerator = Accelerator(even_batches=False, mixed_precision='bf16')
    
    params.model_dir = os.path.join(os.getcwd(), 'models/KDD_AQA', params.task, params.version)
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
    accelerator.print(f'Saving Model to: {params.model_dir}')
    params.device = accelerator.device

    set_seed(params.seed)

    accelerator.print(f"Initialized seed to {params.seed}")
    params.num_proc = accelerator.num_processes

    trn_ques, trn_abs, val_ques, val_abs, XY = load_data(params)

    train_dataset = KDDTrainDataset(trn_ques, trn_abs, XY, params)
    train_order_fname = f"{params.model_dir}/train_order.dat"
    train_bs_fname = f"{params.model_dir}/train_batch_size.dat"

    if accelerator.is_main_process:
        if os.path.exists(train_order_fname) and len(params.load_model) != 0:
            # If loading model, then it is either to continue training or do inference. Regardless, load the previous train_order
            accelerator.print("Loading existing training order")
            train_order = np.memmap(train_order_fname, dtype=np.int32, mode='r+', shape=(len(train_dataset),))
            train_bs = np.memmap(train_bs_fname, dtype=np.int32, mode='r+')
            try:
                cluster_mat = sp.load_npz(f"{params.model_dir}/cluster_mat.npz")
            except:
                cluster_mat = None
        
        else:
            # Only create a new one if you are not loading a model
            accelerator.print("Creating new training order")
            embs = torch.rand(trn_ques['input_ids'].shape[0], 64).to(params.device)
            # embs = torch.from_numpy(np.load(params.model_dir + '/doc_embs.npy')) #.to(DEVICE)
            cluster_mat = cluster_dense_embs(embs, device=params.device, tree_depth=int(math.log(trn_ques['input_ids'].shape[0]/params.batch_size, 2)))
            embs = embs.detach().cpu().numpy()
            del embs
            cmat = cluster_mat[np.random.permutation(cluster_mat.shape[0])]
            batch_size = [batch.nnz for batch in cmat]  
            train_bs = np.memmap(train_bs_fname, dtype=np.int32, mode='w+', shape=(len(batch_size),))
            train_bs[:] = batch_size
            train_order = np.memmap(train_order_fname, dtype=np.int32, mode='w+', shape=(len(train_dataset),))
            train_order[:] = cmat.indices

    else:
        train_order, train_bs, cluster_mat = None, None, None

    accelerator.wait_for_everyone()

    train_dl = DataLoader(dataset = train_dataset, num_workers=4, collate_fn=train_dataset.collate_fn, pin_memory=True, 
                                batch_sampler=BatchSampler(MySampler(train_dataset, train_order_fname), train_bs_fname, False))

    test_ds, ques_ds = DocDataset(val_ques), DocDataset(trn_ques)
    # abs_ds = DocDataset(trn_abs) if params.task == "pretrain" else DocDataset(val_abs)
    abs_ds = DocDataset(val_abs)

    test_dl = DataLoader(dataset = test_ds, batch_size=1024, shuffle=False, 
                                num_workers=4, collate_fn=test_ds.collate_fn, pin_memory=True)

    abs_bs = 1024
    abs_ds = DataLoader(dataset = abs_ds, batch_size=abs_bs, shuffle=False, 
                                num_workers=4, collate_fn=abs_ds.collate_fn, pin_memory=True)
    
    ques_bs = 8192 if params.task == "pretrain" else 1024
    ques_dl = DataLoader(dataset = ques_ds, batch_size=ques_bs, shuffle=False, 
                                num_workers=4, collate_fn=ques_ds.collate_fn, pin_memory=True)

    model = DualEncoder(params)

    runner = Runner([train_dl, test_dl, abs_ds, ques_dl], accelerator, [train_order, train_bs, cluster_mat], params)
    runner.train(model, params) 


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # ------------------------ Params -------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', type=str, default="./data", help='data directory')
    parser.add_argument('--task', type=str, default="pretrain", help='pretrain or train')
    parser.add_argument('--version', type=str, default='', help='model name')
    parser.add_argument('--lm', dest='load_model', type=str, default="", help='model to load')
    parser.add_argument('--test', action='store_true', help='Testing mode or training mode')
    parser.add_argument('--seed', type=int, default=29)

    parser.add_argument('--cd', dest='contrastive_dims', type=int, default=256, help='contrastive dimension')
    parser.add_argument('--loss-lambda', type=float, default=0.5)
    parser.add_argument('--bs', dest='batch_size', type=int, default=256)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-5, help='Learning Rate')
    parser.add_argument('--ep', dest='num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--eval-step', type=int, default=20, help='Evaluate after this many epochs')
    parser.add_argument('--temp', type=float, default = 18.)

    parser.add_argument('--fill-batch', action='store_true')
    parser.add_argument('--add-dual-loss', action='store_true')
    parser.add_argument('--shortlist-size', type=int, default = 20, help = 'ANNS Shortlist Size')

    #In-batch Clustering params
    parser.add_argument('--num-pos', type=int, default = 1)
    parser.add_argument('--update-pos', type=int, default = -1)
    parser.add_argument('--num-negs', type=int, default = 0)
    parser.add_argument('--rebatch', action = 'store_true')
    parser.add_argument('--re-hnm', action = 'store_true')
    parser.add_argument('--load-from-pt', action = 'store_true')
    parser.add_argument('--label-pool-size', type=int, default=1000, help='Learning Rate')
    parser.add_argument('--fill-batch-gap', type=int, default=10)

    parser.add_argument('--cl-start', type=int, default=5, help='Epoch to start using in-batch clustering')
    parser.add_argument('--cl-update', type=int, default=5, help='Num epochs between in-batch reclustering')

    params = parser.parse_args()
    main(params)

# python
