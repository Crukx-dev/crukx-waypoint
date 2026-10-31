#!/usr/bin/env python3
"""Requires crukx_waypoint. Shared world demo: one save, two players, then a rewind that replays bit-exactly.

A player explores a Biome world, saves, and two players continue from that exact moment with their own
hands (different human-like control seeds). Then the save is loaded again and player 1's inputs are
replayed: every frame must hash identically to the first time. Output: a split-screen video
(player 1 | player 2) and a JSON record. Uses the shipped plug-in API (crukx_waypoint).

    fork_demo.py --model-dir MODEL --seed-image world.jpg --out DIR
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from world_engine import WorldEngine

from common import require_crukx, seed_frames
from controls import human_player

crukx = require_crukx()


def play(engine, controls, steps=None):
    """Play a fixed list of controls, or `steps` steps of a reacting human_player; returns frames,
    frame hashes and the controls actually sent (so a branch can be replayed exactly)."""
    frames, hashes, sent, f = [], [], [], None
    for i in range(len(controls) if steps is None else steps):
        ctrl = controls[i] if steps is None else controls.send(f) if i else next(controls)
        f = engine.gen_frame(ctrl=ctrl).cpu().numpy()
        frames.append(f)
        hashes.append(hashlib.sha256(f.tobytes()).hexdigest())
        sent.append(ctrl)
    return np.concatenate(frames), hashes, sent


def encode(frames, path, fps=30):
    h, w = frames.shape[1:3]
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                    '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(path)],
                   input=frames.tobytes(), check=True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--seed-image', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--explore', type=int, default=60, help='steps before the save')
    ap.add_argument('--branch', type=int, default=80, help='steps each player plays after it')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    engine = WorldEngine(args.model_dir, device='cuda', dtype=torch.bfloat16)
    report = crukx.accelerate(engine)
    saves = crukx.Saves(engine, args.out / 'saves')
    engine.reset()
    engine.append_frame(seed_frames(args.seed_image))

    intro, _, _ = play(engine, human_player(seed=1), args.explore)
    moment = saves.save()
    p1, p1_hashes, p1_controls = play(engine, human_player(seed=11), args.branch)
    saves.load(moment)
    p2, _, _ = play(engine, human_player(seed=22), args.branch)
    saves.load(moment)
    _, replay_hashes, _ = play(engine, p1_controls)  # rewind: player 1's exact inputs again

    world = Path(args.seed_image).stem
    encode(intro, args.out / f'{world}_intro.mp4')
    encode(np.concatenate([p1, p2], axis=2), args.out / f'{world}_two_players.mp4')
    record = {'world': world, 'gpu': torch.cuda.get_device_name(), 'crukx': report, 'save': moment,
              'rewind_replay_exact': replay_hashes == p1_hashes,
              'players_differ': not np.array_equal(p1, p2)}
    (args.out / f'{world}_fork.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
