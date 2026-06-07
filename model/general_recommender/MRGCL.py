"""
MRGCL: Multi-Relation Graph Contrastive Learning for Recommendation

Extends SGL with:
- Social graph (U-U) contrastive learning
- Category graph (B-C) contrastive learning
- Cross-relation InfoNCE losses
"""

__author__ = "Yehaikun"
__email__ = "yehaikun@example.com"

__all__ = ["MRGCL"]

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
import os


class _LightGCN(nn.Module):
    """Standard LightGCN for User-Business interaction graph."""
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
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

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)    # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)    # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))        # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  # [batch_size, num_users]

        return sup_logits, ssl_logits_user, ssl_logits_item, user_embeddings, item_embeddings

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class _SocialGCN(nn.Module):
    """GCN for User-User social graph. Propagates user embeddings through social relations."""
    def __init__(self, num_users, embed_dim, uu_adj, n_layers):
        super(_SocialGCN, self).__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        self.uu_adj = uu_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self._user_embeddings_final = None

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            path = os.path.join(dir, 'user_embeddings.npy')
            if os.path.exists(path):
                pretrain = np.load(path)
                tensor = torch.FloatTensor(pretrain).cuda()
                self.user_embeddings = nn.Embedding.from_pretrained(tensor)
                return
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)

    def forward(self):
        return self._forward_gcn(self.uu_adj)

    def _forward_gcn(self, norm_adj):
        embeddings = self.user_embeddings.weight
        all_emb = [embeddings]
        for k in range(self.n_layers):
            embeddings = torch_sp.mm(norm_adj, embeddings)
            all_emb.append(embeddings)
        all_emb = torch.stack(all_emb, dim=1).mean(dim=1)
        return all_emb  # [num_users, dim]

    def eval(self):
        super(_SocialGCN, self).eval()
        self._user_embeddings_final = self._forward_gcn(self.uu_adj)


class _CategoryGCN(nn.Module):
    """GCN for Business-Category graph."""
    def __init__(self, num_items, num_categories, embed_dim, bc_adj, n_layers, cat_pretrain=None):
        super(_CategoryGCN, self).__init__()
        self.num_items = num_items
        self.num_categories = num_categories
        self.embed_dim = embed_dim
        self.bc_adj = bc_adj
        self.n_layers = n_layers
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        if cat_pretrain is not None:
            self.category_embeddings = nn.Embedding.from_pretrained(cat_pretrain)
        else:
            self.category_embeddings = nn.Embedding(self.num_categories, self.embed_dim)
        self._item_embeddings_final = None

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        init = get_initializer(init_method)
        init(self.item_embeddings.weight)
        if not hasattr(self.category_embeddings, 'weight') or self.category_embeddings.weight.requires_grad:
            pass  # category embeddings already pretrained or initialized

    def forward(self):
        return self._forward_gcn(self.bc_adj)

    def _forward_gcn(self, norm_adj):
        item_emb = self.item_embeddings.weight
        cat_emb = self.category_embeddings.weight
        embeddings = torch.cat([item_emb, cat_emb], dim=0)
        all_emb = [embeddings]
        for k in range(self.n_layers):
            embeddings = torch_sp.mm(norm_adj, embeddings)
            all_emb.append(embeddings)
        all_emb = torch.stack(all_emb, dim=1).mean(dim=1)
        item_embs = all_emb[:self.num_items]  # [num_items, dim]
        return item_embs

    def eval(self):
        super(_CategoryGCN, self).eval()
        self._item_embeddings_final = self._forward_gcn(self.bc_adj)


class MRGCL(AbstractRecommender):
    """Multi-Relation Graph Contrastive Learning.

    Uses three relation graphs:
    1. U-B interaction graph (standard)
    2. U-U social graph (new)
    3. B-C category graph (new)

    Losses: BPR + SSL + SocialNC + CatNC
    """
    def __init__(self, config):
        super(MRGCL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
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

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Hyper-parameters for cross-relation contrastive
        self.social_reg = float(config["social_reg"]) if "social_reg" in config else 0.1
        self.cat_reg = float(config["cat_reg"]) if "cat_reg" in config else 0.1
        self.cross_temp = float(config["cross_temp"]) if "cross_temp" in config else self.ssl_temp

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (self.n_layers, self.reg)
        self.model_str += '/ssl_ratio=%.1f-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio, self.ssl_temp, self.ssl_reg)
        self.model_str += '/social_reg=%.1f-cat_reg=%.1f' % (self.social_reg, self.cat_reg)

        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, self.model_name, self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, self.model_name, self.n_layers)

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.num_ratings = self.dataset.num_train_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ---- Load preprocessed.pt for social & category data ----
        self._load_aux_data()

        # ---- Build U-B main graph ----
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        self.lightgcn.reset_parameters(init_method=self.param_init)

        # ---- Build Social GCN (U-U graph) ----
        uu_adj_tensor = sp_mat_to_sp_tensor(self.uu_adj).to(self.device)
        self.social_gcn = _SocialGCN(self.num_users, self.emb_size,
                                     uu_adj_tensor, self.n_layers).to(self.device)
        self.social_gcn.reset_parameters(init_method=self.param_init)

        # ---- Build Category GCN (B-C graph) ----
        bc_adj_tensor = sp_mat_to_sp_tensor(self.bc_adj).to(self.device)
        self.category_gcn = _CategoryGCN(self.num_items, self.num_categories,
                                          self.emb_size, bc_adj_tensor,
                                          self.n_layers, cat_pretrain=self.cat_pretrain).to(self.device)

        # ---- Optimizer (all params) ----
        self.optimizer = torch.optim.Adam(
            list(self.lightgcn.parameters()) +
            list(self.social_gcn.parameters()) +
            list(self.category_gcn.parameters()),
            lr=self.lr
        )

    def _load_aux_data(self):
        """Load social and category data from preprocessed.pt."""
        pt_path = os.path.join(
            self.config.root_dir, 'dataset', self.dataset_name, 'preprocessed.pt')
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                "preprocessed.pt not found at %s. Please upload it first." % pt_path)

        data = torch.load(pt_path, map_location='cpu', weights_only=False)

        # Social edges (E_UU_idx): [2, num_edges] in unified node space
        if 'E_UU_idx' in data:
            e_uu = data['E_UU_idx'].numpy()
            # Both rows are user IDs (0 to num_users-1)
            row = e_uu[0]
            col = e_uu[1]
            vals = np.ones(len(row), dtype=np.float32)
            # Build U-U adjacency matrix
            uu = sp.csr_matrix((vals, (row, col)), shape=(self.num_users, self.num_users))
            uu = uu + uu.T  # symmetric
            # Symmetric normalization
            rowsum = np.array(uu.sum(1)).flatten()
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            self.uu_adj = d_mat_inv @ uu @ d_mat_inv
            self.logger.info("Social graph loaded: %d edges" % len(row))
        else:
            self.logger.warning("No social graph found (E_UU_idx missing), using zero adj")
            self.uu_adj = sp.identity(self.num_users, dtype=np.float32, format='csr')

        # Category edges (E_BC_idx): [2, num_edges]
        if 'E_BC_idx' in data and 'num_categories' in data:
            self.num_categories = int(data['num_categories'])
            e_bc = data['E_BC_idx'].numpy()
            # Map from unified node space to B-C space
            # Row: business IDs (offset by num_users)
            # Col: category IDs (offset by num_users + num_items)
            row = e_bc[0] - self.num_users
            col = e_bc[1] - self.num_users - self.num_items
            # Filter valid entries
            valid = (row >= 0) & (row < self.num_items) & (col >= 0) & (col < self.num_categories)
            row, col = row[valid], col[valid]
            vals = np.ones(len(row), dtype=np.float32)
            bc = sp.csr_matrix((vals, (row, col)),
                               shape=(self.num_items + self.num_categories,
                                      self.num_items + self.num_categories))
            bc = bc + bc.T
            rowsum = np.array(bc.sum(1)).flatten()
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            self.bc_adj = d_mat_inv @ bc @ d_mat_inv
            self.logger.info("Category graph loaded: %d edges, %d categories" % (len(row), self.num_categories))
        else:
            self.logger.warning("No category graph found, using dummy")
            self.num_categories = 1
            n = self.num_items + self.num_categories
            self.bc_adj = sp.identity(n, dtype=np.float32, format='csr')

        # Pre-trained category embeddings
        if 'E_c' in data:
            self.cat_pretrain = data['E_c'].float()
            self.logger.info("Category embeddings loaded: %s" % str(self.cat_pretrain.shape))
        else:
            self.cat_pretrain = None

    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        """Create U-B interaction adjacency matrix (same as SGL)."""
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
            else:
                # ND: node dropout
                drop_user_idx = randint_choice(self.num_users, size=int(self.num_users * self.ssl_ratio), replace=False)
                drop_item_idx = randint_choice(self.num_items, size=int(self.num_items * self.ssl_ratio), replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix((np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                                  shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                                        shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # Symmetric normalization
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                      batch_size=self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0

        # Pre-compute full embeddings for social and category GCNs (reused each epoch)
        # These are detached from the graph to save memory
        self.social_gcn.train()
        self.category_gcn.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_bpr_loss = 0.0
            total_reg_loss = 0.0
            total_ssl_loss = 0.0
            total_social_loss = 0.0
            total_cat_loss = 0.0
            training_start_time = time()

            # Create augmented sub-graphs for U-B SSL
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(self.n_layers):
                    tmp = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp).to(self.device))
                    tmp = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp).to(self.device))

            # Compute full graph embeddings for cross-relation contrastive
            with torch.no_grad():
                user_social_full = self.social_gcn()
                item_cat_full = self.category_gcn()
                user_social_full_norm = F.normalize(user_social_full, dim=1)
                item_cat_full_norm = F.normalize(item_cat_full, dim=1)

            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                # Forward through U-B main graph + SSL views
                sup_logits, ssl_logits_user, ssl_logits_item, user_emb_main, item_emb_main = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)

                # === BPR Loss ===
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # === Reg Loss ===
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # === SSL InfoNCE (standard) ===
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)

                # === Cross-relation: Social InfoNCE ===
                # u_batch_main: [batch, dim] from U-B graph
                u_batch_main = F.embedding(bat_users, user_emb_main)
                u_batch_social = F.embedding(bat_users, user_social_full_norm)
                pos_social = inner_product(u_batch_main, u_batch_social)  # [batch]
                # Negatives: all users' social embeddings
                neg_social = torch.matmul(u_batch_main, user_social_full_norm.T)  # [batch, num_users]
                su = neg_social - pos_social[:, None]
                cu = torch.logsumexp(su / self.cross_temp, dim=1)
                social_loss = torch.sum(cu)

                # === Cross-relation: Category InfoNCE ===
                i_batch_main = F.embedding(bat_pos_items, item_emb_main)
                i_batch_cat = F.embedding(bat_pos_items, item_cat_full_norm)
                pos_cat = inner_product(i_batch_main, i_batch_cat)  # [batch]
                neg_cat = torch.matmul(i_batch_main, item_cat_full_norm.T)  # [batch, num_items]
                si = neg_cat - pos_cat[:, None]
                ci = torch.logsumexp(si / self.cross_temp, dim=1)
                cat_loss = torch.sum(ci)

                # === Total Loss ===
                loss = (bpr_loss
                        + self.ssl_reg * infonce_loss
                        + self.social_reg * social_loss
                        + self.cat_reg * cat_loss
                        + self.reg * reg_loss)

                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                total_ssl_loss += infonce_loss
                total_social_loss += social_loss
                total_cat_loss += cat_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Logging
            self.logger.info(
                "[iter %d : loss : %.4f = bpr:%.4f + ssl:%.4f + social:%.4f + cat:%.4f + reg:%.4f, time: %f]" % (
                    epoch,
                    total_loss / self.num_ratings,
                    total_bpr_loss / self.num_ratings,
                    self.ssl_reg * total_ssl_loss / self.num_ratings,
                    self.social_reg * total_social_loss / self.num_ratings,
                    self.cat_reg * total_cat_loss / self.num_ratings,
                    total_reg_loss / self.num_ratings,
                    time() - training_start_time,
                ))

            # Evaluation
            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
