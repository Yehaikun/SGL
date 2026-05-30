"""Apply MACL (Multi-Augmentation Contrastive Learning) to SGL.
Adds feature masking as a third augmentation view and 3-way pairwise InfoNCE.
"""
import re

with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "r") as f:
    code = f.read()

# 1. Remove all hard_alpha code
code = code.replace(
    '        self.hard_alpha = config.get("hard_alpha", 0.0)',
    '        self.mask_ratio = config.get("mask_ratio", 0.0)'
)
code = re.sub(
    r'                if self\.hard_alpha > 0:.*?\n                else:\n                    clogits_item = torch\.logsumexp\(ssl_logits_item / self\.ssl_temp, dim=1\)',
    '                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)\n                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)',
    code, flags=re.DOTALL
)

# 2. In the forward method of _LightGCN, after view1 and view2, add view3 (feature masking)
# Find: user_embeddings2 = F.normalize(user_embeddings2, dim=1)
# And add feature masking after it
old = """        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)"""

new = """        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        # View 3: Feature masking
        if self.mask_ratio > 0:
            fmask = torch.rand(user_embeddings.shape[1], device=user_embeddings.device) > self.mask_ratio
            user_embeddings3 = F.normalize(user_embeddings * fmask.float(), dim=1)
            item_embeddings3 = F.normalize(item_embeddings * fmask.float(), dim=1)
        else:
            user_embeddings3, item_embeddings3 = user_embeddings1, item_embeddings1

        user_embs = F.embedding(users, user_embeddings)"""

code = code.replace(old, new)

# 3. Add mask_ratio to __init__ if not present
if "self.mask_ratio" not in code:
    code = code.replace(
        'self.ssl_temp = config["ssl_temp"]',
        'self.ssl_temp = config["ssl_temp"]\n        self.mask_ratio = config.get("mask_ratio", 0.0)'
    )

# 4. Return view3 from forward
code = code.replace(
    "return sup_logits, ssl_logits_user, ssl_logits_item",
    "return sup_logits, ssl_logits_user, ssl_logits_item, user_embeddings3, item_embeddings3"
)

# 5. Update the training loop forward call (5 returns instead of 3)
code = code.replace(
    "sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(",
    "sup_logits, ssl_logits_user, ssl_logits_item, u3, i3 = self.lightgcn("
)

# 6. Add 3rd view InfoNCE loss after the existing infonce_loss
old_loss = """                infonce_loss = torch.sum(clogits_user + clogits_item)"""

new_loss = """                infonce_loss = torch.sum(clogits_user + clogits_item)
                # MACL: 3rd view (feature masking) vs 1st view (edge dropout)
                if self.mask_ratio > 0:
                    u1_batch = user_embs1
                    i1_batch = item_embs1
                    u3_batch = F.embedding(bat_users, u3)
                    i3_batch = F.embedding(bat_pos_items, i3)
                    # User side
                    pos_u = inner_product(u1_batch, u3_batch)
                    tot_u = torch.matmul(u1_batch, u3.T)
                    ssl_u = tot_u - pos_u[:, None]
                    cu = torch.logsumexp(ssl_u / self.ssl_temp, dim=1)
                    # Item side
                    pos_i = inner_product(i1_batch, i3_batch)
                    tot_i = torch.matmul(i1_batch, i3.T)
                    ssl_i = tot_i - pos_i[:, None]
                    ci = torch.logsumexp(ssl_i / self.ssl_temp, dim=1)
                    infonce_loss += torch.sum(cu + ci)"""

code = code.replace(old_loss, new_loss)

with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "w") as f:
    f.write(code)

# Verify
import py_compile
try:
    py_compile.compile("/root/paper/SGL-Torch/model/general_recommender/SGL.py", doraise=True)
    print("SYNTAX OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")

# Count changes
for term in ["mask_ratio", "u3, i3", "user_embeddings3"]:
    count = code.count(term)
    print(f"  {term}: {count} occurrences")
