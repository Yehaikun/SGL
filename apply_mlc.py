"""Apply Multi-Layer Contrastive (MLC) to SGL."""
with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "r") as f:
    c = f.read()

# 1. Make _forward_gcn return per-layer embeddings when return_all=True
old_fwd = """    def _forward_gcn(self, norm_adj):
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

        return user_embeddings, item_embeddings"""

new_fwd = """    def _forward_gcn(self, norm_adj, return_all=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        if return_all:
            layer_list = []
            for emb in all_embeddings:
                u, i = torch.split(emb, [self.num_users, self.num_items], dim=0)
                layer_list.append((u, i))
            return layer_list

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings"""

c = c.replace(old_fwd, new_fwd)

# 2. Modify the SGL training loop forward to use multi-layer contrast
# The forward method already computes ssl_logits from view1 and view2
# We need to modify _LightGCN.forward to compute multi-layer SSL

# Add MLC loss after the existing InfoNCE loss in train_model
old_loss = "                infonce_loss = torch.sum(clogits_user + clogits_item)"

new_loss = """                infonce_loss = torch.sum(clogits_user + clogits_item)
                # MLC: Multi-Layer Contrastive
                layers1 = self.lightgcn._forward_gcn(self.lightgcn.norm_adj, return_all=True)
                layers2_v1 = self.lightgcn._forward_gcn(sub_graph1, return_all=True)
                layers2_v2 = self.lightgcn._forward_gcn(sub_graph2, return_all=True)
                if isinstance(sub_graph1, list):
                    layers2_v1 = self.lightgcn._forward_gcn(sub_graph1, return_all=True)
                    layers2_v2 = self.lightgcn._forward_gcn(sub_graph2, return_all=True)
                # For each layer, compute InfoNCE between view1 and view2 embeddings
                mlc_loss = 0.0
                n_layers = self.lightgcn.n_layers + 1
                for li in range(n_layers):
                    u1, i1 = layers2_v1[li]
                    u2, i2 = layers2_v2[li]
                    u1n, i1n = F.normalize(u1, dim=1), F.normalize(i1, dim=1)
                    u2n, i2n = F.normalize(u2, dim=1), F.normalize(i2, dim=1)
                    bu1 = F.embedding(bat_users, u1n)
                    bi1 = F.embedding(bat_pos_items, i1n)
                    bu2 = F.embedding(bat_users, u2n)
                    bi2 = F.embedding(bat_pos_items, i2n)
                    pu = inner_product(bu1, bu2)
                    pi = inner_product(bi1, bi2)
                    tu = torch.matmul(bu1, u2n.T)
                    ti = torch.matmul(bi1, i2n.T)
                    su = tu - pu[:, None]
                    si = ti - pi[:, None]
                    cu = torch.logsumexp(su / self.ssl_temp, dim=1)
                    ci = torch.logsumexp(si / self.ssl_temp, dim=1)
                    mlc_loss += torch.sum(cu + ci)
                mlc_loss /= n_layers
                infonce_loss = infonce_loss + mlc_loss"""

c = c.replace(old_loss, new_loss)

with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "w") as f:
    f.write(c)

import py_compile
try:
    py_compile.compile("/root/paper/SGL-Torch/model/general_recommender/SGL.py", doraise=True)
    print("OK")
except py_compile.PyCompileError as e:
    print(f"ERROR: {e}")
