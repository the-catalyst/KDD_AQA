import os
import gc
import math
import time
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate.utils import broadcast, send_to_device
from transformers import AdamW, get_linear_schedule_with_warmup
from dense_clustering import cluster_dense_embs
from accelerate.logging import get_logger
import bitsandbytes as bnb

class Runner:
    def __init__(self, dataloaders, accelerator, query_clustering, params, top_k=20):
        self.train_dl = dataloaders[0]
        self.test_dl = dataloaders[1]
        self.test_abs_dl = dataloaders[2]
        self.ques_dl = dataloaders[3]
        self.abs_dl = dataloaders[4]
        self.accelerator = accelerator
        self.num_train, self.num_test = len(self.train_dl.dataset), len(self.test_dl.dataset)
        self.top_k = top_k
        self.num_proc = params.num_proc
        self.task = params.task
        self.model_dir = params.model_dir
        self.DEVICE = accelerator.device
        self.proc_id = int(str(accelerator.device)[-1])
        self.shortlist_size = params.shortlist_size
        self.train_order = query_clustering[0]
        self.train_bs = query_clustering[1]
        self.cluster_mat = query_clustering[2]
        self.loss_lambda = params.loss_lambda
        self.hard_min_negs = params.num_negs > 0
        self.latest_anns_ep = 0
        self.update_pos = params.update_pos
        self.label_pool_size = params.label_pool_size
        with open(f"{params.data_path}/all_pids.raw.txt") as fil:
            self.id_to_pid_map = np.array([pid.strip() for pid in fil.readlines()])

        # self.logger = self.logger.basicConfig(level=self.logger.INFO, filename=f"{params.model_dir}.log", filemode="a+", main_process_only=True)
        self.logger = get_logger(name=f"{params.version}.log", log_level="INFO")
    
        self.tree_depth = int(math.log(self.num_train/params.batch_size, 2))
        

    def create_hard_negatives(self, model, epoch, doc_embs=None):
        self.accelerator.print("Updating Hard Negatives")
        torch.cuda.empty_cache()
        
        if self.latest_anns_ep != epoch:
            lbl_enc_embs = model.get_dataset_embeddings(self.abs_dl, tqdm_disable = not self.accelerator.is_main_process)
            if self.accelerator.is_main_process:
                model.anns.build_index(lbl_enc_embs)
            
            del lbl_enc_embs
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
        
        if doc_embs == None:
            doc_embs = model.get_dataset_embeddings(self.ques_dl, tqdm_disable = not self.accelerator.is_main_process)
        
        if self.accelerator.is_main_process:
            all_preds = model.anns.search(doc_embs.float(), k = self.shortlist_size) 
            hard_negs = all_preds[0].cpu().numpy()
            np.save(self.model_dir + '/hard_negatives.npy', hard_negs)
            del model.anns.anns, all_preds, hard_negs
        
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        self.train_dl.dataset.reload_negatives()
        
        return doc_embs


    def update_clustered_batches(self, model, epoch, cl_start, cl_update, embs = None, force_rebatch = False):
        if epoch >= cl_start:
            if (epoch - cl_start) % cl_update == 0 or force_rebatch == True:
                if embs is None:
                    self.accelerator.print(f'Started creating updated query text embeddings at {time.ctime()}')
                    torch.cuda.empty_cache()
                    cluster_dl = self.ques_dl if self.task == "train" else self.abs_dl
                    embs = model.get_dataset_embeddings(cluster_dl, tqdm_disable = not self.accelerator.is_main_process)
                    self.accelerator.print(f'Query embeddings created at {time.ctime()}')
                
                if self.accelerator.is_main_process:
                    self.cluster_mat = cluster_dense_embs(embs, embs.device, tree_depth=self.tree_depth).tocsr()
                    sp.save_npz(f'{self.model_dir}/cluster_mat.npz', self.cluster_mat)
                
                embs = embs.detach().cpu().numpy()
                del embs

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                print('Updating clustered train order...\n')
                cmat = self.cluster_mat[np.random.permutation(self.cluster_mat.shape[0])]
                self.train_order[:] = cmat.indices
                self.train_bs[:] = np.array([b.nnz for b in cmat])
        
        elif self.accelerator.is_main_process:
            print('Shuffling train order...\n')
            self.train_order[:] = np.random.permutation(len(self.train_dl.dataset))

        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

    def predict(self, preds, y_true, ext, num=None, den=None):
        for pred, tr in zip(preds, y_true):
            match = torch.isin(pred, tr.cpu())
            ext += torch.cumsum(match, dim=0)/len(tr)

    def process_batch_for_multi_GPU_train(self, batch):

        def split_batch(x):
            batch_size = (x['input_ids'].shape[0]//self.num_proc)
            if x['input_ids'].shape[0] % batch_size > 0:
                batch_size += 1
            x['input_ids'] = x['input_ids'][batch_size * self.proc_id : batch_size * (self.proc_id + 1)]
            x['attention_mask'] = x['attention_mask'][batch_size * self.proc_id : batch_size * (self.proc_id + 1)]
            return x, batch_size

        num_lbls = torch.ones(1).to(self.DEVICE) * batch['lbls']['input_ids'].shape[0]
        num_lbls = broadcast(num_lbls)
        num_lbls = int(num_lbls.item())

        batch['docs'] = {k: v.to(self.DEVICE) for k, v in batch['docs'].items()}
        batch['docs'], batch['doc_batch_size'] = split_batch(batch['docs'])
        # print(self.DEVICE, batch['docs']['input_ids'].shape, batch['doc_batch_size'])
        
        batch['lbls'] = {k: v[:num_lbls].to(self.DEVICE) for k, v in batch['lbls'].items()}
        batch['lbls'] = self.accelerator.pad_across_processes(batch['lbls'])
        batch['lbls'] = broadcast(batch['lbls'])
        batch['lbls'], batch['lbl_batch_size'] = split_batch(batch['lbls'])
        # print(self.DEVICE, batch['lbls']['input_ids'].shape, batch['lbl_batch_size'])

        batch['target']['target_cnst'] = send_to_device(batch['target']['target_cnst'], self.DEVICE)
        batch['target']['target_cnst'] = self.accelerator.pad_across_processes(batch['target']['target_cnst'], dim=1)
        batch['target']['target_cnst'] = broadcast(batch['target']['target_cnst'])
        batch['target']['target_cnst'] = batch['target']['target_cnst'][:, :num_lbls]

        return batch

    def fit_one_epoch(self, model, epoch):
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        train_loss_clf, train_loss_cnst = 0.0, 0.0
        self.cos_score = torch.zeros(self.top_k)

        model.train()
        lf, df = 0., 0.
        loss_cnst, loss_clf = torch.tensor([0.]), torch.tensor([0.])

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)
        for step, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            batch = self.process_batch_for_multi_GPU_train(batch)

            if self.accelerator.is_main_process:
                target = batch['target']
                lf += torch.mean(torch.sum(target['target_cnst'], axis=1))
                df += target['target_cnst'].shape[1]
                pbar.set_postfix({'DF': target['target_cnst'].shape[1], 'ContL': loss_cnst.item(), 'ClfL': loss_clf.item()})

            loss_cnst, probs_cos, loss_clf, probs_dot, candidates = model(**batch, step=step)
            
            if self.loss_lambda != -1:
                loss_cnst = loss_cnst * self.loss_lambda
                loss_clf = loss_clf * (1 - self.loss_lambda)
            
            loss = loss_cnst + loss_clf            
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            
            if self.accelerator.is_main_process:
                train_loss_cnst += loss_cnst.detach()
                train_loss_clf += loss_clf.detach()

                if probs_cos is not None:
                    preds_cos = torch.topk(probs_cos, self.top_k)[1].to('cpu')
                    preds_cos = target['cnst_labels'][preds_cos].detach().cpu()
                    self.predict(preds_cos, target['lbls'], self.cos_score)

            self.accelerator.wait_for_everyone()


        if self.accelerator.is_main_process:
            train_loss_clf /= self.steps_per_epoch
            train_loss_cnst /= self.steps_per_epoch

            print(f"Epoch: {epoch}, LR: {[round(x, 6) for x in self.scheduler.get_last_lr()]},  Contr Loss:  {train_loss_cnst:.4f}, Classif Loss: {train_loss_clf:.4f}")
            print(f'Labels/Doc: {(lf/self.steps_per_epoch):.2f}, Labels/Batch: {(df//self.steps_per_epoch)}')
            self.logger.info(f"Epoch: {epoch}, Contr Loss:  {train_loss_cnst:.4f}, Avg. Labels/Doc: {(lf/self.steps_per_epoch):.2f}, Avg. Labels/batch: {(df/self.steps_per_epoch):.2f}", main_process_only=True)

            recall = self.cos_score.detach().cpu().numpy() * 100.0 / (self.num_train)
            print(f'ANNS Training Scores: R@1: {recall[0]:.2f}, R@5: {recall[4]:.2f}, R@10: {recall[9]:.2f}, R@20: {recall[19]:.2f}')

        self.accelerator.wait_for_everyone()

    def initialize_model(self, model, params):
        model_path = os.path.join(self.model_dir, params.load_model)
        self.accelerator.print(f'loading model from {model_path}')

        self.accelerator.load_state(model_path)
        if params.load_from_pt:
            init = 0
        else:
            # init = math.ceil(self.scheduler.state_dict()['last_epoch']/self.steps_per_epoch)
            init = 5

        if params.test:
            self.evaluate(model.module, params, init)
            exit()

        if self.update_pos != -1 and init >= params.cl_start:
            self.accelerator.print(f"\nUpdating number of sampled positive to {self.update_pos}.\n")
            self.train_dl.dataset.num_pos = self.update_pos

        self.accelerator.wait_for_everyone()

        doc_embs = None

        if params.re_hnm:
            doc_embs = self.create_hard_negatives(model.module, init)

        self.update_clustered_batches(model.module, init, params.cl_start, params.cl_update, doc_embs, params.rebatch)
        del doc_embs

        return init


    def train(self, model, params):

        pattern = "%"*100 + '\n'
        self.accelerator.print(model)
        self.logger.info(model, main_process_only=True)
        self.accelerator.print(pattern + str(params) + '\n' + pattern)
        self.logger.info(pattern + str(params) + '\n' + pattern, main_process_only=True)
        
        self.optimizer = AdamW(model.dense_grouped_params, lr=params.lr)
        
        self.steps_per_epoch = len(self.train_dl)
        init, last_batch = 0, -1

        model = self.accelerator.prepare(model)

        if params.load_from_pt:
            init = self.initialize_model(model, params)

        self.optimizer = self.accelerator.prepare(self.optimizer)

        self.test_dl, self.ques_dl, self.abs_dl = self.accelerator.prepare(self.test_dl, self.ques_dl, self.abs_dl)

        if len(params.load_model) and not params.load_from_pt:
            init = self.initialize_model(model, params)
            # last_batch = init * self.steps_per_epoch

        warm_up_steps = 2*self.steps_per_epoch if params.task == "train" else self.steps_per_epoch//2
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, last_epoch=last_batch,
                                                num_training_steps=params.num_epochs*self.steps_per_epoch,
                                                num_warmup_steps = warm_up_steps)
        
        cl_start, cl_update = params.cl_start, params.cl_update

        doc_embs = None

        for epoch in range(init + 1, params.num_epochs + 1):    
            gc.collect()
            torch.cuda.empty_cache()

            self.fit_one_epoch(model.module, epoch)

            self.accelerator.save_state(f'{params.model_dir}/model_latest.pth')

            if epoch == cl_start:
                self.accelerator.save_state(f'{params.model_dir}/model_{cl_start}.pth')
            
            if (epoch - cl_start) % cl_update == 0:
                if self.hard_min_negs and epoch >= cl_start:
                    doc_embs = self.create_hard_negatives(model.module, epoch)
                else:
                    doc_embs = None                    
                
                self.update_clustered_batches(model.module, epoch, cl_start, cl_update, doc_embs)                
                del doc_embs            
            else:
                self.update_clustered_batches(model.module, epoch, cl_start, cl_update)

        self.evaluate(model.module, params, epoch)


    def evaluate(self, model, params, epoch):    
        
        torch.cuda.empty_cache()

        ##Get Document Embeddings
        self.accelerator.print(f"Creating Test Document Embeddings at Epoch {epoch}")
        test_enc_embs = model.get_dataset_embeddings(self.test_dl, tqdm_disable = not self.accelerator.is_main_process)
        
        ##Get Encoder Label Embeddings
        lbl_enc_embs = model.get_dataset_embeddings(self.test_abs_dl, tqdm_disable = not self.accelerator.is_main_process)

        if self.accelerator.is_main_process:
            model.anns.build_index(lbl_enc_embs)

            model.eval()
            with torch.no_grad():
                candidates, probs = model.anns.search(test_enc_embs.float(), k=20)   
                candidates = candidates.detach().cpu()
                    
            preds = self.id_to_pid_map[candidates]
            preds = '\n'.join(','.join(pred) for pred in preds)
            with open(f"{self.model_dir}/pred_file.txt", "w") as fil:
                fil.write(preds)
        
        del test_enc_embs, lbl_enc_embs
        torch.cuda.empty_cache()
        self.latest_anns_ep = epoch
        self.accelerator.wait_for_everyone()