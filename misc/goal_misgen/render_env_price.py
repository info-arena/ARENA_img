"""Render the env_price layout from the [2.3.2] 'everyone has a price' solution as a PNG.
Usage: python render_env_price.py <worktree root> <out.png>"""
import ast, sys
from pathlib import Path
import torch as t
from PIL import Image
root, out = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / "chapter2_rl/exercises"))
from part32_goal_misgen.potteryshop import Environment
src = (root / "infrastructure/chapters/chapter2_rl/master_2_3_2.py").read_text()
# the layout lives in a markdown code block in the solution: take it from there, so the picture can't drift
start = src.index("env_price = Environment(")
block = src[start:src.index("```", start)]
ns = {"t": t, "Environment": Environment}
exec(block, ns)
env = ns["env_price"]
img = env.render(env.reset(1))
Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(out, optimize=True)
print(img.shape, "->", out)
