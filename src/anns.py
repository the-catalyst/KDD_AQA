import faiss
import torch
import time
import faiss.contrib.torch_utils
import torch.nn.functional as F

import hnswlib
import numpy as np
from tqdm import tqdm
import math

class FaissMIPSIndex():
    def __init__(self, device):
        self.device = device
        
    def search(self, query_batch, k = 1000):
        dists, keys = self.anns.search(query_batch, k)
        return keys, dists

    def build_index(self, embs):
        if hasattr(self, 'anns'):
            del self.anns
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.device
        resource = faiss.StandardGpuResources()
        self.anns = faiss.GpuIndexFlatIP(resource, embs.shape[1], cfg)
        # self.anns = faiss.GpuIndexFlatL2(resource, embs.shape[1], cfg)
        self.anns.add(embs)
        embs = embs.cpu()
        del embs

class HNSW(object):
    def __init__(self, M=110, efC=100, efS=1000, num_threads=90, device=0):
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS
        self.device = device

    def build_index(self, data, print_progress=True):
        if hasattr(self, 'anns'):
            del self.anns
        data = data.cpu().numpy()
        self.anns = hnswlib.Index(space='ip', dim=data.shape[1])
        self.anns.init_index(max_elements=data.shape[0], ef_construction=self.efC, M=self.M)
        data_labels = np.arange(data.shape[0]).astype(np.int64)
        self.anns.add_items(data, data_labels, num_threads=self.num_threads)
        del data

    def search(self, query_batch, k = 25):
        self.anns.set_ef(self.efS)
        k = min(k, self.efS)
        keys, dists = self.anns.knn_query(query_batch.cpu().numpy(), k=k)
        keys = keys.astype(np.int64())
        dists *= -1

        return torch.from_numpy(keys).to(self.device), torch.from_numpy(dists).to(self.device)