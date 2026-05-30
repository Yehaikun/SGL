import math, py_compile

with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "r") as f:
    c = f.read()

# Add import math
c = c.replace(
    "from util.common import Reduction",
    "from util.common import Reduction\nimport math"
)

# Add curriculum augmentation
old = "for epoch in range(1, self.epochs + 1):"
new = """for epoch in range(1, self.epochs + 1):
            p = (epoch - 1) / self.epochs
            self.ssl_ratio = self.ssl_ratio * max(0.5, 1.0 - 0.5 * (1.0 + math.cos(p * math.pi)))"""

c = c.replace(old, new, 1)

with open("/root/paper/SGL-Torch/model/general_recommender/SGL.py", "w") as f:
    f.write(c)

py_compile.compile("/root/paper/SGL-Torch/model/general_recommender/SGL.py", doraise=True)
print("OK")
