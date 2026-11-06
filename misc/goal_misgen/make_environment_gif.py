"""Render a gif of a scripted robot cleaning a pottery shop, for the [2.3.2] intro.

The policy is hard-coded: repeatedly BFS to the nearest pile of shards (never
stepping on an urn), pick it up, BFS to the bin, put it down. Frames come from
the real `Environment.step` / `Environment.render`.

Usage: python make_env_gif.py <worktree root> <out.gif>
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image

root, out = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / "chapter2_rl/exercises"))
from part32_goal_misgen.potteryshop import Action, Environment, Item

SCALE = 4  # 48x56 px rendered -> 192x224, same size as the old environment.png

# an illustrative layout for the intro animation (not the notebook's `env`): the
# urns sit across the robot's routes, so it has to weave around them (the bin is
# reachable only from below). No shard sits directly above an urn, since the tall
# urn sprite would hide it
env = Environment(
    init_robot_pos=t.tensor((2, 4)),
    init_items_map=t.tensor(
        (
            (0, 0, 2, 0, 0, 0),
            (0, 2, 0, 0, 1, 0),
            (0, 0, 0, 2, 0, 0),
            (2, 0, 1, 0, 0, 2),
            (0, 0, 0, 2, 1, 0),
            (0, 1, 0, 2, 0, 0),
        )
    ),
    bin_pos=t.tensor((0, 0)),
)
MOVES = {Action.UP: (-1, 0), Action.LEFT: (0, -1), Action.DOWN: (1, 0), Action.RIGHT: (0, 1)}


def bfs(start, goal, items):
    """Shortest list of move actions from start to goal, avoiding urns."""
    ws = items.shape[0]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for a, (dr, dc) in MOVES.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < ws and 0 <= nxt[1] < ws and nxt not in prev and items[nxt] != Item.URN:
                prev[nxt] = (cur, a)
                q.append(nxt)
    path = []
    while prev[goal] is not None:
        goal, a = prev[goal]
        path.append(a)
    return path[::-1]


state = env.reset(1)
frames = [env.render(state)]


def act(a):
    global state
    state = env.step(state, t.tensor([int(a)]))
    frames.append(env.render(state))


while True:
    items = state.items_map[0].numpy()
    pos = tuple(state.robot_pos[0].tolist())
    shards = [tuple(p) for p in np.argwhere(items == Item.SHARDS) if tuple(p) != (0, 0)]
    if not shards:
        break
    target = min(shards, key=lambda p: len(bfs(pos, p, items)))
    for a in bfs(pos, target, items):
        act(a)
    act(Action.PICKUP)
    assert int(state.inventory[0]) == Item.SHARDS
    for a in bfs(target, tuple(state.bin_pos[0].tolist()), state.items_map[0].numpy()):
        act(a)
    act(Action.PUTDOWN)
    assert int(state.inventory[0]) == Item.EMPTY

assert not (state.items_map[0] == Item.SHARDS).any(), "all shards should be binned"
assert (state.items_map[0] == Item.URN).sum() == (env.init_items_map == Item.URN).sum(), "no urn broken"

# hold the first and last frames so the loop is readable
durations = [1200] + [220] * (len(frames) - 2) + [1800]
imgs = [Image.fromarray(f).resize((f.shape[1] * SCALE, f.shape[0] * SCALE), Image.NEAREST) for f in frames]
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
print(f"{len(frames)} frames, {imgs[0].size}, wrote {out}")
