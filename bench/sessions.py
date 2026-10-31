#!/usr/bin/env python3
"""Requires crukx_waypoint. Many players on one engine (crukx.Sessions): exactness, then how many fit and how fast they play.

    sessions.py --model-dir MODEL --seeds a.jpg b.jpg ... --out sessions.json

1. Exact: players 0 and 2 of three interleaved sessions must hash frame-for-frame like the same world played alone.
2. Scale: open sessions in steps (1, 2, 4, ...) until the GPU is full or a player falls below --floor fps,
   and at each step play round-robin rounds and record aggregate fps, per-player fps and GPU memory.
Exit status 0 only if the exact check passes.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from world_engine import CtrlInput, WorldEngine

from common import require_crukx, seed_frames

crukx = require_crukx()

GB = 2 ** 30


def ctrl(player, i):
    """Walk forward while looking around; each player turns differently."""
    return CtrlInput(button={87} | ({32} if i % 17 == 5 else set()),
                     mouse=(round(120 * math.sin(0.15 * i + player)), 0))


def digest(frames):
    return hashlib.sha256(frames.cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def solo(engine, frames, player, steps):
    torch.manual_seed(player)
    engine.reset()
    engine.append_frame(frames)
    return [digest(engine.gen_frame(ctrl=ctrl(player, i))) for i in range(steps)]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kernels', default='auto')
    ap.add_argument('--exact-steps', type=int, default=24)
    ap.add_argument('--rounds', type=int, default=12, help='round-robin rounds timed at each size')
    ap.add_argument('--floor', type=float, default=5.0)
    ap.add_argument('--max-players', type=int, default=512)
    args = ap.parse_args()
    seeds = [seed_frames(p) for p in args.seeds]

    engine = WorldEngine(args.model_dir, device='cuda', dtype=torch.bfloat16)
    report = crukx.accelerate(engine, kernels={'auto': 'auto', 'stock': False}[args.kernels])
    weights_gb = torch.cuda.memory_allocated() / GB

    # 1. exact: interleaved players match the same world played alone
    alone = {p: solo(engine, seeds[p % len(seeds)], p, args.exact_steps) for p in (0, 2)}
    sessions = crukx.Sessions(engine)
    for p in range(3):
        sessions.open(seeds[p % len(seeds)], seed=p)
    shared = {p: [] for p in range(3)}
    for i in range(args.exact_steps):
        for p in range(3):
            shared[p].append(digest(sessions.step(p, ctrl(p, i))))
    exact = all(shared[p] == alone[p] for p in alone)
    print(json.dumps({'exact': exact}), flush=True)

    # 2. scale: one engine, more and more players, until memory or speed runs out
    sessions = crukx.Sessions(engine)
    torch.cuda.empty_cache()
    rows, n, step = [], 0, 0
    sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    for target in (s for s in sizes if s <= args.max_players):
        try:
            while len(sessions) < target:
                sid = sessions.open(seeds[len(sessions) % len(seeds)], seed=len(sessions))
                for i in range(2):
                    sessions.step(sid, ctrl(sid, i))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            for r in range(args.rounds):
                for sid in range(len(sessions)):
                    sessions.step(sid, ctrl(sid, 2 + r)).cpu()
            wall = time.perf_counter() - t
        except torch.OutOfMemoryError:
            rows.append({'players': target, 'oom': True})
            print(json.dumps(rows[-1]), flush=True)
            break
        aggregate = 4 * len(sessions) * args.rounds / wall
        rows.append({'players': len(sessions), 'aggregate_fps': round(aggregate, 1),
                     'fps_per_player': round(aggregate / len(sessions), 1),
                     'gpu_allocated_gb': round(torch.cuda.memory_allocated() / GB, 1),
                     'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / GB, 1)})
        print(json.dumps(rows[-1]), flush=True)
        if rows[-1]['fps_per_player'] < args.floor:
            break

    total = torch.cuda.get_device_properties(0).total_memory / GB
    ok = [r for r in rows if not r.get('oom')]
    per_player = (ok[-1]['gpu_allocated_gb'] - weights_gb) / ok[-1]['players'] if ok else None
    result = {'gpu': torch.cuda.get_device_name(), 'gpu_total_gb': round(total, 1), 'model': args.model_dir,
              'crukx': report, 'exact_interleaved_vs_alone': exact, 'engine_gb': round(weights_gb, 2),
              'gb_per_player': round(per_player, 3) if per_player else None, 'scale': rows}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'scale'}), flush=True)
    raise SystemExit(0 if exact else 1)


if __name__ == '__main__':
    main()
