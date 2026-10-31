#!/usr/bin/env python3
"""How much would batching players into one forward pass save? An upper-bound probe, before building it.

Times the model's own linear layers (stock BF16 F.linear) at tokens_per_frame x B rows for B players, and
the real batch-1 step, so the matrix-multiply share of a step is known. Attention is per player (each attends
to its own cache), so only the matrix-multiply share can shrink with batching.

    batch_probe.py --model-dir MODEL --out probe.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from triton.testing import do_bench
from world_engine import CtrlInput, WorldEngine

from common import seed_frames


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--seed-image', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batches', default='1,2,4,8,16')
    args = ap.parse_args()
    cfg = yaml.safe_load((Path(args.model_dir) / 'config.yaml').read_text())
    tokens, passes = cfg['tokens_per_frame'], len(cfg['scheduler_sigmas']) - 1

    engine = WorldEngine(args.model_dir, device='cuda', dtype=torch.bfloat16)
    shapes = Counter((m.in_features, m.out_features) for n, m in engine.model.named_modules()
                     if isinstance(m, torch.nn.Linear) and '.blocks.' in f'.{n}.')
    engine.reset()
    engine.append_frame(seed_frames(args.seed_image))
    for _ in range(10):
        engine.gen_frame(ctrl=CtrlInput(button={87}))
    step_ms = do_bench(lambda: engine.gen_frame(ctrl=CtrlInput(button={87})), warmup=3, rep=300)

    weights = {s: torch.randn(s[1], s[0], device='cuda', dtype=torch.bfloat16) for s in shapes}
    rows = []
    for b in map(int, args.batches.split(',')):
        x = {s: torch.randn(tokens * b, s[0], device='cuda', dtype=torch.bfloat16) for s in shapes}
        gemm = passes * sum(n * do_bench(lambda s=s: F.linear(x[s], weights[s]), warmup=5, rep=50)
                            for s, n in shapes.items())
        rows.append({'players': b, 'gemm_ms_per_step': round(gemm, 2), 'gemm_ms_per_player': round(gemm / b, 2)})
        print(json.dumps(rows[-1]), flush=True)
    share = rows[0]['gemm_ms_per_step'] / step_ms
    for r in rows:  # step time per player if only the matrix-multiply share batches
        r['est_step_ms_per_player'] = round(step_ms * (1 - share) + r['gemm_ms_per_player'], 2)
        r['est_throughput_x'] = round(step_ms / r['est_step_ms_per_player'], 2)
    result = {'gpu': torch.cuda.get_device_name(), 'model': args.model_dir, 'tokens_per_frame': tokens,
              'passes_per_step': passes, 'linear_shapes': {f'{i}x{o}': n for (i, o), n in shapes.items()},
              'step_ms_batch1': round(step_ms, 2), 'gemm_share_batch1': round(share, 3), 'batches': rows}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'batches'}), flush=True)


if __name__ == '__main__':
    main()
