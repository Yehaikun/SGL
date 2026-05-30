import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(__file__))
from reckit.configurator import Configurator
from reckit import randint_choice
import numpy as np, torch

# Import and use MACL model
from model.general_recommender.SGL_MACL import SGL
import model.general_recommender.SGL_MACL as macl_mod

# Set mask_ratio
macl_mod.MASK_RATIO = 0.1

# Override SGL.__init__ to set mask_ratio
original_init = SGL.__init__
def patched_init(self, config):
    original_init(self, config)
    self.mask_ratio = 0.1
    self.lightgcn.mask_ratio = 0.1
SGL.__init__ = patched_init

# Also override _LightGCN.forward to add feature masking
original_forward = macl_mod._LightGCN.forward
def patched_forward(self, sg1, sg2, users, items, neg_items):
    # Call original to get standard outputs
    result = original_forward(self, sg1, sg2, users, items, neg_items)
    sup, ssl_u, ssl_i = result
    
    # Feature masking view
    u, i = self._forward_gcn(self.norm_adj)
    u1, i1 = self._forward_gcn(sg1)
    u1, i1 = F.normalize(u1, dim=1), F.normalize(i1, dim=1)
    
    if self.mask_ratio > 0:
        fm = torch.rand(u.shape[1], device=u.device) > self.mask_ratio
        u3 = F.normalize(u * fm.float(), dim=1)
        i3 = F.normalize(i * fm.float(), dim=1)
    else:
        u3, i3 = u1, i1
    
    u1_batch = F.embedding(users, u1)
    i1_batch = F.embedding(items, i1)
    u3_batch = F.embedding(users, u3)
    i3_batch = F.embedding(items, i3)
    
    return sup, ssl_u, ssl_i, u1_batch, i1_batch, u3_batch, i3_batch

import torch.nn.functional as F
from util.pytorch import inner_product
macl_mod._LightGCN.forward = patched_forward
