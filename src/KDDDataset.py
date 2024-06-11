import os
import torch
import numpy as np
import torch.sparse
from torch.utils.data import Dataset, Sampler
from xclib.data import data_utils as du
import scipy.sparse as sp
from xclib.utils.sparse import retain_topk
from typing import Union, Iterable, Sized, List, Iterator

class MySampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source, fname):
        self.data_source = data_source
        self.order = np.memmap(fname, dtype=np.int32, mode='r', shape=(len(data_source),))
        assert len(self.order) == len(self.data_source)

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.data_source)


class BatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: str, drop_last: bool) -> None:
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_sizes = np.memmap(batch_size, dtype=np.int32, mode='r')
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_sizes)]
                    yield batch
                except StopIteration:
                    break
        else:
            b_i = 0
            batch = [0] * self.batch_sizes[b_i]
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_sizes[b_i]:
                    yield batch
                    idx_in_batch = 0
                    b_i += 1
                    if b_i != len(self.batch_sizes):
                        batch = [0] * self.batch_sizes[b_i]
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        return len(self.batch_sizes)


class DocDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs
    
    def __len__(self):
        return len(self.docs['input_ids'])
    
    def __getitem__(self, idx):
        return idx 
    
    def collate_fn(self, batch):
        collated = {'docs': None}
        indices = np.array([x for x in batch])
        collated['docs'] = {'input_ids': self.docs['input_ids'][indices], 'attention_mask': self.docs['attention_mask'][indices]}
        collated['indices'] = indices
        
        return collated

class XMLTestDataset(DocDataset):
    def __init__(self, docs, XY, data_path):
        super(XMLTestDataset, self).__init__(docs)
        
        self.test_labels = [torch.from_numpy(x) for x in np.split(XY.indices, XY.indptr[1:-1])]
        
        filter_file = os.path.join(data_path, 'filter_labels_test.txt')

        if os.path.exists(filter_file):
            filter_test = np.loadtxt(filter_file).astype(np.int64)
            rows, cols, data = filter_test[:, 0], filter_test[:, 1], [1]*filter_test.shape[0]
            filter_test = sp.csr_matrix((data, (rows, cols)), shape=(XY.shape[0], XY.shape[1]))

            self.filter_test = {}
            for i in range(filter_test.shape[0]):
                if len(filter_test[i].indices):
                    self.filter_test[i] = torch.from_numpy(filter_test[i].indices)
            print("Loaded filter test file.")
        else:
            print("Filter test file not found in the dataset folder.")


class KDDTrainDataset(Dataset):
    def __init__(self, docs, lbls, XY, params):
        self.docs = docs
        self.lbls = lbls
        self.XY = XY
        if params.num_negs > 0:
            self.negatives_file = params.model_dir + '/hard_negatives.npy'
            if os.path.exists(self.negatives_file):
                print("Loading existing hard negatives")
                self.reload_negatives()
        self.num_negs = params.num_negs
        self.label_pool_size = params.label_pool_size
        self.neg_pool_size = (params.num_negs * params.cl_update)
        self.min_batch_gap = params.fill_batch_gap
    
    def __len__(self):
        return self.XY.shape[0]
    
    def __getitem__(self, idx):
        doc_idx = idx
        labels = self.XY[doc_idx].indices

        if hasattr(self, 'negatives'):
            neg_mask = ~np.isin(self.negatives[doc_idx], labels)
            neg_pool = self.negatives[doc_idx][neg_mask][:self.neg_pool_size]
            
            neg_lbls = np.random.choice(neg_pool, self.num_negs, replace=False) 

        else:
            neg_lbls = np.array([])

        out = {'doc_idx': doc_idx, 'pos_lbls': labels, 'neg_lbls': neg_lbls}
        
        return out

    def reload_negatives(self):
        self.negatives = np.load(self.negatives_file).astype(np.int32)
    
    def collate_fn(self, batch):
        batch_docs = np.array([x['doc_idx'] for x in batch])
        
        batch_labels = np.concatenate([x['pos_lbls'] for x in batch], axis = None)
        batch_labels, batch_stats = np.unique(batch_labels, return_counts = True)

        batch_gap = self.label_pool_size - len(batch_labels)

        if batch_gap < 0:
            pos_impt = np.argsort(-batch_stats)
            batch_labels = batch_labels[pos_impt][:batch_gap]
        
        elif hasattr(self, 'negatives') and batch_gap > self.min_batch_gap:
            neg_batch_lbls, neg_stats = np.unique(np.concatenate([x['neg_lbls'] for x in batch], axis = None), return_counts = True)
            neg_mask = np.isin(neg_batch_lbls, batch_labels, invert=True)
            neg_batch_lbls, neg_stats = neg_batch_lbls[neg_mask], neg_stats[neg_mask]
            if len(neg_batch_lbls) > batch_gap:
                neg_impt = np.argsort(-neg_stats)
                neg_batch_lbls = neg_batch_lbls[neg_impt][:batch_gap]
            batch_labels = np.concatenate((batch_labels, neg_batch_lbls))

        target_cnst = np.zeros((len(batch_docs), len(batch_labels)), dtype=np.float32)

        positive_labels = []
        
        for i, b in enumerate(batch):
            positive_labels.append(torch.tensor(b['pos_lbls']))
            target_cnst[i] = np.isin(batch_labels, b['pos_lbls']).astype(np.float32)

        lbls_per_doc = target_cnst.sum(1)
        doc_mask = lbls_per_doc > 0

        doc_ii = self.docs['input_ids'][batch_docs][doc_mask]
        doc_am = self.docs['attention_mask'][batch_docs][doc_mask]        

        target_cnst = torch.from_numpy(target_cnst[doc_mask]) 
        
        lbl_ii = self.lbls['input_ids'][batch_labels]
        lbl_am = self.lbls['attention_mask'][batch_labels]

        collated = {'target': {'target_cnst': target_cnst, 'lbls': positive_labels, 'cnst_labels': torch.from_numpy(batch_labels)}}

        collated['docs'] = {'input_ids': doc_ii, 'attention_mask': doc_am}
        collated['lbls'] = {'input_ids': lbl_ii, 'attention_mask': lbl_am}

        return collated