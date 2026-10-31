#!/usr/bin/env python3
"""Frames per second of the Waypoint loop, split into phases, after a warm compile.

One step is the engine's gen_frame: input prep, denoise pass, cache pass, decode (4 frames).
Reports the per-step wall time, the fps a player sees (median step), and unsynced throughput
(production style: no per-step sync or events).

    fps.py --model-dir MODEL --out fps.json [--crukx] [--quant intw8a8]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import ctrl, load_engine, percentiles, workload

PHASES = ('prep', 'denoise', 'cache', 'decode')


def timed_step(engine, spec, ev):
    ev[0].record()
    x = torch.randn(engine.frm_shape, device=engine.device, dtype=engine.dtype)
    inputs = engine.prep_inputs(x=x, ctrl=ctrl(spec))
    ev[1].record()
    x0 = engine._denoise_pass(x, inputs, engine.kv_cache).clone()
    ev[2].record()
    engine._cache_pass(x0, inputs, engine.kv_cache)
    ev[3].record()
    engine.vae.decode(x0.squeeze(1))
    ev[4].record()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--crukx', action='store_true')
    ap.add_argument('--quant', help="the engine's own quantization, e.g. intw8a8")
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--seed', type=int, default=73)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    engine, report = load_engine(args.model_dir, crukx=args.crukx, quant=args.quant)
    engine.reset()
    controls = workload('mixed', args.warmup + args.steps)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for spec in controls[:args.warmup]:
        timed_step(engine, spec, ev)
    torch.cuda.synchronize()

    wall, phases = [], {p: [] for p in PHASES}
    for spec in controls[args.warmup:]:
        t0 = time.perf_counter()
        timed_step(engine, spec, ev)
        torch.cuda.synchronize()
        wall.append(1000 * (time.perf_counter() - t0))
        for i, p in enumerate(PHASES):
            phases[p].append(ev[i].elapsed_time(ev[i + 1]))

    frames = engine.model_cfg.temporal_compression
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for spec in controls[args.warmup:]:
        engine.vae.decode(engine.gen_frame(ctrl(spec), return_img=False))
    torch.cuda.synchronize()
    throughput = frames * args.steps / (time.perf_counter() - t0)

    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'crukx': report, 'quant': args.quant,
              'step_wall_ms': percentiles(wall), 'phase_ms': {p: percentiles(v) for p, v in phases.items()},
              'displayed_fps_p50': round(1000 * frames / percentiles(wall)['p50'], 1),
              'throughput_fps_unsynced': round(throughput, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
