"""AdaGate-SGL: Adaptive Gating Enhanced Self-supervised Graph Learning

Architecture innovation - Adaptive Gating in GCN propagation:
    gate = sigmoid(W_gate @ h + b_gate)   # per-node, per-layer gate
    h_new = gate * (A @ h) + (1-gate) * h # gated blend of propagation and identity

This allows each node to adaptively control information flow based on
its current embedding state, preventing oversmoothing for high-degree
nodes while preserving identity for low-degree nodes.
"""

__author__ = "Yehaikun"
__all__ = ["AdaGateSGL"]

import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from data import PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice


class _GatedGCN(nn.Module):
    """LightGCN with adaptive per-node gating."""
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers, use_gate=True):
        super(_GatedGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.use_gate = use_gate
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # Adaptive gating parameters: per-layer scalar gate projection
        if use_gate:
            self.gate_proj = nn.ModuleList([
                nn.Linear(embed_dim, 1, bias=True) for _ in range(n_layers)
            ])

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)
        sup_logits = sup_pos_ratings - sup_neg_ratings

        pos_ratings_user = inner_product(user_embs1, user_embs2)
        pos_ratings_item = inner_product(item_embs1, item_embs2)
        tot_ratings_user = torch.matmul(user_embs1, user_embeddings2.T)
        tot_ratings_item = torch.matmul(item_embs1, item_embeddings2.T)

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]
        return sup_logits, ssl_logits_user, ssl_logits_item

    def _forward_gcn(self, norm_adj, return_all=False):
        ego = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = [ego]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                prop = torch_sp.mm(norm_adj[k], ego)
            else:
                prop = torch_sp.mm(norm_adj, ego)

            if self.use_gate and self.training:
                # Per-node adaptive gate: how much to blend propagated signal
                gate = torch.sigmoid(self.gate_proj[k](ego))  # [n_nodes, 1]
                ego = gate * prop + (1 - gate) * ego
            else:
                ego = prop
            all_emb.append(ego)

        if return_all:
            layer_list = []
            for emb in all_emb:
                u, i = torch.split(emb, [self.num_users, self.num_items], dim=0)
                layer_list.append((u, i))
            return layer_list

        all_emb = torch.stack(all_emb, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items], dim=0)
        return user_emb, item_emb

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_GatedGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class AdaGateSGL(AbstractRecommender):
    """AdaGate-SGL: SGL with adaptive gating mechanism."""
    def __init__(self, config):
        super(AdaGateSGL, self).__init__(config)
        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]
        self.n_layers = config['n_layers']

        self.ssl_aug_type = config["aug_type"].lower()
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_temp = config["ssl_temp"]
        self.ssl_warmup_epochs = config["ssl_warmup_epochs"] if "ssl_warmup_epochs" in config else 3
        self.mlc_start_layer = config["mlc_start_layer"] if "mlc_start_layer" in config else 0
        self.mlc_layer_weight = config["mlc_layer_weight"] if "mlc_layer_weight" in config else "uniform"
        self.ssl_degree_alpha = config["ssl_degree_alpha"] if "ssl_degree_alpha" in config else 0.5
        self.use_gate = bool(int(config["use_gate"])) if "use_gate" in config else True

        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)
        self.model_str = '#layers=%d-reg=%.0e/ratio=%.1f-temp=%.2f' % (self.n_layers, self.reg, self.ssl_ratio, self.ssl_temp)
        self.model_str += '/gate=%d' % int(self.use_gate)
        self.pretrain_flag = config["pretrain_flag"]
        self.save_flag = config["save_flag"]

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.num_ratings = self.dataset.num_train_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Degree weights (DGW)
        user_deg = np.array(self.dataset.train_csr_mat.sum(1)).reshape(-1).astype(np.float32)
        item_deg = np.array(self.dataset.train_csr_mat.sum(0)).reshape(-1).astype(np.float32)
        self.user_ssl_weights = np.power(user_deg + 1.0, -float(self.ssl_degree_alpha))
        self.item_ssl_weights = np.power(item_deg + 1.0, -float(self.ssl_degree_alpha))
        self.user_ssl_weights = self.user_ssl_weights / self.user_ssl_weights.mean()
        self.item_ssl_weights = self.item_ssl_weights / self.item_ssl_weights.mean()
        self.user_ssl_weights = torch.from_numpy(self.user_ssl_weights).float().to(self.device)
        self.item_ssl_weights = torch.from_numpy(self.item_ssl_weights).float().to(self.device)

        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        self.lightgcn = _GatedGCN(self.num_users, self.num_items, self.emb_size,
                                   adj_matrix, self.n_layers, use_gate=self.use_gate).to(self.device)
        self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]
        if is_subgraph and self.ssl_ratio > 0:
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                users_np, items_np = np.array(users_np)[keep_idx], np.array(items_np)[keep_idx]
                ratings = np.ones_like(users_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))
            elif aug_type == 'nd':
                du = randint_choice(self.num_users, size=int(self.num_users * self.ssl_ratio), replace=False)
                di = randint_choice(self.num_items, size=int(self.num_items * self.ssl_ratio), replace=False)
                iu = np.ones(self.num_users, dtype=np.float32); iu[du] = 0.
                ii = np.ones(self.num_items, dtype=np.float32); ii[di] = 0.
                R = sp.csr_matrix((np.ones_like(users_np, dtype=np.float32), (users_np, items_np)), shape=(self.num_users, self.num_items))
                R_prime = sp.diags(iu).dot(R).dot(sp.diags(ii))
                u_keep, i_keep = R_prime.nonzero()
                tmp_adj = sp.csr_matrix((R_prime.data, (u_keep, i_keep + self.num_users)), shape=(n_nodes, n_nodes))
        else:
            tmp_adj = sp.csr_matrix((np.ones_like(users_np, dtype=np.float32), (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5).flatten(); d_inv[np.isinf(d_inv)] = 0.
        return sp.diags(d_inv).dot(adj_mat).dot(sp.diags(d_inv))

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss = total_bpr = total_reg = 0.0
            start = time()
            if self.ssl_aug_type in ['nd', 'ed']:
                sg1 = sp_mat_to_sp_tensor(self.create_adj_mat(True, self.ssl_aug_type)).to(self.device)
                sg2 = sp_mat_to_sp_tensor(self.create_adj_mat(True, self.ssl_aug_type)).to(self.device)
            else:
                sg1, sg2 = [], []
                for _ in range(self.n_layers):
                    sg1.append(sp_mat_to_sp_tensor(self.create_adj_mat(True, self.ssl_aug_type)).to(self.device))
                    sg2.append(sp_mat_to_sp_tensor(self.create_adj_mat(True, self.ssl_aug_type)).to(self.device))
            self.lightgcn.train()
            for bu, bp, bn in data_iter:
                bu = torch.from_numpy(bu).long().to(self.device)
                bp = torch.from_numpy(bp).long().to(self.device)
                bn = torch.from_numpy(bn).long().to(self.device)
                sp_logits, sl_u, sl_i = self.lightgcn(sg1, sg2, bu, bp, bn)

                bpr = -torch.sum(F.logsigmoid(sp_logits))
                reg = l2_loss(self.lightgcn.user_embeddings(bu), self.lightgcn.item_embeddings(bp), self.lightgcn.item_embeddings(bn))
                cu = torch.logsumexp(sl_u / self.ssl_temp, dim=1)
                ci = torch.logsumexp(sl_i / self.ssl_temp, dim=1)
                ssl = torch.sum(cu + ci)

                # MLC + DGW
                lv1 = self.lightgcn._forward_gcn(sg1, return_all=True)
                lv2 = self.lightgcn._forward_gcn(sg2, return_all=True)
                if isinstance(sg1, list):
                    lv1 = self.lightgcn._forward_gcn(sg1, return_all=True)
                    lv2 = self.lightgcn._forward_gcn(sg2, return_all=True)
                mlc = 0.0
                nl = self.lightgcn.n_layers + 1
                al = list(range(min(max(int(self.mlc_start_layer), 0), nl - 1), nl))
                lw = np.ones(len(al), dtype=np.float32)
                if self.mlc_layer_weight == "linear":
                    lw = np.arange(1, len(al) + 1, dtype=np.float32)
                elif self.mlc_layer_weight == "deep":
                    lw = np.array(al, dtype=np.float32) + 1.0
                lw = lw / lw.sum()
                for w, li in zip(lw, al):
                    u1, i1 = lv1[li]; u2, i2 = lv2[li]
                    u1n = F.normalize(u1, dim=1); i1n = F.normalize(i1, dim=1)
                    u2n = F.normalize(u2, dim=1); i2n = F.normalize(i2, dim=1)
                    pu = inner_product(F.embedding(bu, u1n), F.embedding(bu, u2n))
                    pi = inner_product(F.embedding(bp, i1n), F.embedding(bp, i2n))
                    su = torch.matmul(F.embedding(bu, u1n), u2n.T) - pu[:, None]
                    si = torch.matmul(F.embedding(bp, i1n), i2n.T) - pi[:, None]
                    cu2 = torch.logsumexp(su / self.ssl_temp, dim=1)
                    ci2 = torch.logsumexp(si / self.ssl_temp, dim=1)
                    uw = torch.index_select(self.user_ssl_weights, 0, bu)
                    iw = torch.index_select(self.item_ssl_weights, 0, bp)
                    mlc += float(w) * torch.sum((cu2 * uw) + (ci2 * iw))
                ssl = ssl + mlc

                sw = min(1.0, float(epoch) / float(self.ssl_warmup_epochs)) if self.ssl_warmup_epochs > 0 else 1.0
                loss = bpr + self.ssl_reg * sw * ssl + self.reg * reg
                total_loss += loss; total_bpr += bpr; total_reg += self.reg * reg
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch, total_loss / self.num_ratings, total_bpr / self.num_ratings,
                (total_loss - total_bpr - total_reg) / self.num_ratings, total_reg / self.num_ratings, time() - start))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                res, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, res))
                if flag:
                    self.best_epoch = epoch; stopping_step = 0
                    self.logger.info("Find a better model.")
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping at epoch: %d" % epoch); break

        self.logger.info("best@epoch %d:\n\t\t%s" % (self.best_epoch, '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])))

    def evaluate_model(self):
        self.lightgcn.eval()
        cur, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < cur[1]:
            self.best_result = cur
            return buf, True
        return buf, False

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
