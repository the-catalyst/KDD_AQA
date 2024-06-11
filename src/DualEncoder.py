import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from anns import FaissMIPSIndex
import torch.nn.functional as F
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

class DualEncoder(nn.Module):
    def __init__(self, params):
        super(DualEncoder, self).__init__()
        self.embed_dim = 768
        self.device = params.device
        self.num_proc = params.num_proc
        self.proc_id = int(str(params.device)[-1])
        self.shortlist_size = params.shortlist_size
        self.anns = FaissMIPSIndex(torch.cuda.current_device())

        self.contrastive_dims = params.contrastive_dims
        self.temp = nn.Parameter(torch.tensor(params.temp)) 
        self.init_classifiers()

        self.add_dual_loss = params.add_dual_loss

        print(f"Using temp = {self.temp.item()} for this model training.")

        #https://www.sbert.net/examples/applications/computing-embeddings/README.html
        self.bert = SentenceTransformer("msmarco-distilbert-base-v4")
        
        lr = params.lr
        no_decay = ['LayerNorm.weight']
        wd = 0.05
        
        self.dense_grouped_params = [
            {'params': [p for n, p in [*self.bert.named_parameters()] if not any(nd in n for nd in no_decay)], 'weight_decay': wd, 'lr': lr},
            {'params': [p for n, p in [*self.bert.named_parameters()] if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
            {'params': [*self.cnst_head[0].parameters()], 'weight_decay': wd, 'lr': lr*2}
        ]

    def init_classifiers(self):
        self.cnst_head = nn.Sequential(nn.Linear(self.embed_dim, self.contrastive_dims), nn.Tanh(), nn.Dropout(0.1))
        nn.init.xavier_uniform_(self.cnst_head[0].weight)
        nn.init.zeros_(self.cnst_head[0].bias)

    def encode(self, x, is_doc=False):
        input_ids, attn_mask = x['input_ids'], x['attention_mask']

        max_len = torch.max(torch.sum(attn_mask, dim=1))
        input_ids, attn_mask = input_ids[:, :max_len], attn_mask[:, :max_len] 

        bert_out = self.bert({'input_ids': input_ids, 'attention_mask': attn_mask})
        cnst_emb = self.cnst_head(bert_out['sentence_embedding'])
        
        cnst_emb = F.normalize(cnst_emb)
        return cnst_emb

    def get_dataset_embeddings(self, dataloader, is_doc=False, tqdm_disable=False):
        cnst_embeddings = torch.zeros((len(dataloader.dataset), self.contrastive_dims)).float().cuda()
        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=tqdm_disable):
                x = {k: v.cuda() for k, v in batch['docs'].items()}
                cnst_emb = self.encode(x)
                cnst_embeddings[batch["indices"]] = cnst_emb.detach()
        self.train()

        if self.num_proc > 1:
            dist.all_reduce(cnst_embeddings, op=dist.ReduceOp.SUM)

        if is_doc:
            return cnst_embeddings, None    
        return cnst_embeddings 

    def compute_loss(self, sim, target):
        num_pos = target.sum(-1)
                
        exp_sim = F.log_softmax(sim, dim=-1) 
        soft_logits = (target * exp_sim).sum(dim=-1)             
        loss = - (soft_logits/num_pos).mean()
        
        return loss

    def forward(self, docs, lbls = None, target = None, 
                doc_batch_size = None, lbl_batch_size = None, step=100):

        target = target['target_cnst']#, target['num_pos_labs']
        pos_lbl_idx = target.sum(0).nonzero().squeeze()

        doc_embs = self.encode(docs)
        lbl_embs = self.encode(lbls)

        if self.num_proc > 1:            
            global_doc_embs = torch.zeros(target.shape[0], doc_embs.shape[1]).to(self.device)
            global_doc_embs[doc_batch_size*self.proc_id : doc_batch_size*(self.proc_id + 1)] = doc_embs
            dist.all_reduce(global_doc_embs, op=dist.ReduceOp.SUM)
            doc_embs = global_doc_embs

            global_lbl_embs = torch.zeros(target.shape[1], lbl_embs.shape[1]).to(self.device)
            global_lbl_embs[lbl_batch_size*self.proc_id : lbl_batch_size*(self.proc_id + 1)] = lbl_embs
            dist.all_reduce(global_lbl_embs, op=dist.ReduceOp.SUM)
            lbl_embs = global_lbl_embs

        cos_sim = (doc_embs @ lbl_embs.T)*self.temp
        
        loss_doc = self.compute_loss(cos_sim, target)
        
        if self.add_dual_loss:
            loss_lbl = self.compute_loss(cos_sim.T[pos_lbl_idx], target.T[pos_lbl_idx])
        else:
            loss_lbl = torch.tensor(0.).to(self.device)

        
        return loss_doc, cos_sim.sigmoid(), loss_lbl, None, None