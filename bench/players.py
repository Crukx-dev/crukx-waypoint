#!/usr/bin/env python3
"""One player session per process, N processes on one GPU, all playing at the same time.

Each worker loads its own engine (as a hosted server would per session), seeds a Biome world, warms up,
waits at a file barrier until all N are ready, then plays human-like controls and reports its own fps.
players.sh runs N = 1, 2, 4, ... until a session drops below the floor or the GPU runs out of memory.

    players.py --model-dir MODEL --seed-image world.jpg --barrier DIR --rank 0 --n 4 --out r0.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from common import load_engine, seed_frames
from controls import human_controls


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--seed-image', required=True)
    ap.add_argument('--barrier', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--crukx', action='store_true')
    args = ap.parse_args()

    engine, report = load_engine(args.model_dir, crukx=args.crukx)
    engine.reset()
    engine.append_frame(seed_frames(args.seed_image))
    controls = human_controls(args.warmup + args.steps, seed=100 + args.rank)
    for ctrl in controls[:args.warmup]:
        engine.gen_frame(ctrl=ctrl)
    torch.cuda.synchronize()

    (args.barrier / f'ready{args.rank}').touch()
    while len(list(args.barrier.glob('ready*'))) < args.n:
        time.sleep(0.05)

    steps = []
    t0 = time.perf_counter()
    for ctrl in controls[args.warmup:]:
        s = time.perf_counter()
        engine.gen_frame(ctrl=ctrl).cpu()  # frames reach the host, as they would to a video encoder
        steps.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0
    steps.sort()
    p50 = steps[len(steps) // 2]
    result = {'rank': args.rank, 'n': args.n, 'gpu': torch.cuda.get_device_name(), 'crukx': report,
              'fps_p50': round(4 / p50, 1), 'fps_mean': round(4 * len(steps) / wall, 1),
              'step_ms_p95': round(steps[int(len(steps) * 0.95)] * 1000, 1),
              'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
    args.out.write_text(json.dumps(result) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
