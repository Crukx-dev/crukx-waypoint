#!/usr/bin/env python3
"""Key-press-to-pixels latency of the Waypoint loop, on the wall clock.

Inputs arrive at random times over the run. Each step samples the newest input that arrived before
it started; an input's latency is the time from its arrival until the host holds the first image
generated from it. Model side only: no display scan-out, OS input or network.

    latency.py --model-dir MODEL --out latency.json [--crukx] [--quant intw8a8]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import ctrl, load_engine, percentiles, workload


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--crukx', action='store_true')
    ap.add_argument('--quant')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--seed', type=int, default=73)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    engine, report = load_engine(args.model_dir, crukx=args.crukx, quant=args.quant)
    engine.reset()
    specs = workload('mixed', args.warmup + args.steps, args.seed)
    host = torch.empty(engine.gen_frame(ctrl=ctrl(specs[0])).shape, dtype=torch.uint8, pin_memory=True)
    for s in specs[1:args.warmup]:
        host.copy_(engine.gen_frame(ctrl=ctrl(s)))

    starts, readies = [], []
    for s in specs[args.warmup:]:
        starts.append(time.perf_counter())
        host.copy_(engine.gen_frame(ctrl=ctrl(s)))  # blocking: the frame is on the host
        readies.append(time.perf_counter())
    starts, readies = np.asarray(starts), np.asarray(readies)

    rng = random.Random(args.seed)
    arrivals = np.sort([rng.uniform(starts[0], starts[-2]) for _ in range(args.steps)])
    first = np.searchsorted(starts, arrivals, side='left')  # first step that starts after the input
    latency = (readies[first] - arrivals) * 1000
    period = np.diff(starts) * 1000
    result = {'gpu': torch.cuda.get_device_name(), 'crukx': report, 'quant': args.quant, 'steps': args.steps,
              'key_to_pixels_ms': percentiles(latency), 'step_period_ms': percentiles(period),
              'video_fps': round(4000 / float(np.median(period)), 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
