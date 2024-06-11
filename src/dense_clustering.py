import functools
import torch
import time
import operator
import numpy as np
import argparse
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import functools

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

def b_kmeans_dense(label_features, index, metric='cosine', tol=1e-4, leakage=None):
    with torch.no_grad():
        n = label_features.shape[0]
        if label_features.shape[0] == 1:
            return [index]
        cluster = np.random.randint(low=0, high=label_features.shape[0], size=(2))

        while cluster[0] == cluster[1]:
            cluster = np.random.randint(low=0, high=label_features.shape[0], size=(2))
        
        _centeroids = label_features[cluster]

        _similarity = torch.mm(label_features, _centeroids.T)
        old_sim, new_sim = -1000000, -2

        while new_sim - old_sim >= tol:
            clustered_lbs = torch.split(torch.argsort(_similarity[:, 1]-_similarity[:, 0]), (_similarity.shape[0]+1)//2)
            _centeroids = F.normalize(torch.vstack([torch.mean(label_features[x, :], axis=0) for x in clustered_lbs]))
            _similarity = torch.mm(label_features, _centeroids.T)
            old_sim, new_sim = new_sim, sum([torch.sum(_similarity[indx, i]) for i, indx in enumerate(clustered_lbs)]).item()/n
        
        del _similarity
        index = index.to(label_features.device)
        return list(map(lambda x: index[x], clustered_lbs))

def cluster_labels(labels, clusters, num_nodes, splitter):
    start = time.time()
    while len(clusters) < num_nodes:
        temp_cluster_list = functools.reduce(
            operator.iconcat,
            map(lambda x: splitter(labels[x], x), clusters), [])
        end = time.time()
        print(f"Total clusters {len(temp_cluster_list)}\tAvg. Cluster size {'%.2f'%(np.mean(list(map(len, temp_cluster_list))))}\tTotal time {'%.2f'%(end-start)} sec")
        clusters = temp_cluster_list
        del temp_cluster_list
    return clusters

@timeit
def cluster_dense_embs(embs, device='cpu', tree_depth = 9):
    print(f'device: {device}')

    if(embs.shape[0] >= 1000000):
        print(f"Num embeddings: {embs.shape[0]} - Using HalfTensor")
        clusters = cluster_labels(torch.Tensor.half(embs), [torch.arange(embs.shape[0])], 2**(tree_depth), b_kmeans_dense)
    else:
       clusters = cluster_labels(embs, [torch.arange(embs.shape[0])], 2**(tree_depth), b_kmeans_dense)
    
    # clusters = cluster_labels(embs.to(device), [torch.arange(embs.shape[0])], 2**(tree_depth), b_kmeans_dense)
    
    clustering_mat = csr_matrix((np.ones(sum([len(c) for c in clusters])), 
                                     torch.cat(clusters).cpu().numpy(),
                                     np.cumsum([0, *[len(c) for c in clusters]])),
                                 shape=(len(clusters), embs.shape[0]))
    return clustering_mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mn', type=str, help='model directory')
    parser.add_argument('--data', type=str, help='dataset name')
    params = parser.parse_args()

    embs = torch.rand(131073, 64).to('cuda:0')
    cluster_mat = cluster_dense_embs(embs, device='cuda:0', tree_depth=9)
    embs = embs.detach().cpu().numpy()
    del embs
#    train_order = np.memmap('/ads-nfs/skharbanda/EMEA_Nov_Data/train_order.dat', dtype=int, mode='w+', shape=(29828040,))
    cmat = cluster_mat[np.random.permutation(cluster_mat.shape[0])]
#    train_order[:] = cmat.indices
    batch_size = [batch.nnz for batch in cmat]  
    train_bs = np.memmap(f'./models/{params.mn}/{params.data}/train_batch_size.dat', dtype=np.int32, mode='w+', shape=(len(batch_size),))
    train_bs[:] = batch_size