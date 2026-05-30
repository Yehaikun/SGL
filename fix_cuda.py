with open('/root/paper/SGL-Torch/model/general_recommender/SGL.py', 'r') as f:
    c = f.read()
c = c.replace(
    'u_main,i_main=self.lightgcn._forward_gcn(self.lightgcn.norm_adj)',
    'u_main,i_main=self.lightgcn._forward_gcn(self.lightgcn.norm_adj)\n                    u_main=u_main.cuda(); i_main=i_main.cuda()'
)
c = c.replace(
    'F.embedding(bat_users,u_main)',
    'F.embedding(bat_users,u_main.cuda())'
)
c = c.replace(
    'F.embedding(bat_pos_items,i_main)',
    'F.embedding(bat_pos_items,i_main.cuda())'
)
with open('/root/paper/SGL-Torch/model/general_recommender/SGL.py', 'w') as f:
    f.write(c)
import py_compile
py_compile.compile('/root/paper/SGL-Torch/model/general_recommender/SGL.py', doraise=True)
print('OK')
