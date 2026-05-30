"""Apply ProCL (Prototype Contrastive Learning) to SGL."""
with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "r") as f:
    content = f.read()

# 1. Add import
content = content.replace(
    "from util.pytorch import inner_product, l2_loss\nfrom util.pytorch import get_initializer",
    "from util.pytorch import inner_product, l2_loss\nfrom util.pytorch import get_initializer\nimport torch.nn.functional as F"
)

# 2. Add prototype params after optimizer
old_init_end = '        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)'
content = content.replace(
    old_init_end,
    old_init_end + '\n        # ProCL parameters\n        self.n_prototypes = 500\n        self.procl_beta = 0.1\n        self.procl_temp = 0.2\n        self.prototypes = nn.Embedding(self.n_prototypes, self.emb_size)\n        nn.init.normal_(self.prototypes.weight, std=0.1)'
)

# 3. Add _procl_loss method before train_model
old_start = '    def train_model(self):'
new_method = '''    def _procl_loss(self, emb_batch):
        """Prototype contrastive loss: pull emb toward its nearest prototype."""
        emb_norm = F.normalize(emb_batch, dim=-1)
        proto_norm = F.normalize(self.prototypes.weight, dim=-1)
        sim = torch.mm(emb_norm, proto_norm.T) / self.procl_temp
        labels = sim.argmax(dim=1).detach()
        return F.cross_entropy(sim, labels)

    def train_model(self):'''
content = content.replace(old_start, new_method)

# 4. Add ProCL loss after standard loss
old_loss = '                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss'
new_loss = '''                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                if self.procl_beta > 0:
                    u_main, i_main = self.lightgcn._forward_gcn(self.lightgcn.norm_adj)
                    u_b = F.embedding(bat_users, u_main)
                    i_b = F.embedding(bat_pos_items, i_main)
                    u_proto = self._procl_loss(u_b)
                    i_proto = self._procl_loss(i_b)
                    loss += self.procl_beta * (u_proto + i_proto)'''

content = content.replace(old_loss, new_loss)

with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "w") as f:
    f.write(content)

import py_compile
try:
    py_compile.compile("/root/paper/SGL-Torch/model/general_recommender/SGL.py", doraise=True)
    print("SYNTAX OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
