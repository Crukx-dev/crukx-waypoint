#!/usr/bin/env python3
"""Fidelity oracle: how far a Waypoint configuration is from a high-precision reference.

golden    FP32 free-running reference on a frozen workload: saves noise, controls and clean latents.
isolated  Before every step the candidate gets the exact canonical history (a BF16 cache built from
          the golden latents): one denoise step's error alone.
history   The candidate keeps its own cache, built by its own cache pass from the golden latents:
          how its cache numerics accumulate, as in play.
free      The candidate keeps its whole state on its own outputs: long-rollout stability.

Pixels are decoded from a copy of a canonical decoder (fed only golden latents), so decoder history
is not a confound. Run stock BF16 through the same test: its error is the budget. A configuration
meets BF16 fidelity when it is no worse than stock; only bit-identical results are lossless.

    fidelity.py golden  --model-dir MODEL --golden g.pt [--seed-image world.jpg]
    fidelity.py history --model-dir MODEL --golden g.pt --out stock.json
    fidelity.py history --model-dir MODEL --golden g.pt --out crukx.json --crukx
    fidelity.py history --model-dir MODEL --golden g.pt --out int8.json --quant intw8a8
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from common import ctrl, load_engine, psnr, rel_err, seed_frames, stats, workload


def start(engine, seed_image):
    engine.reset()
    if seed_image:
        engine.append_frame(seed_frames(seed_image))


@torch.inference_mode()
def golden(args):
    import world_engine.world_engine as we
    we.apply_inference_patches = lambda model: None  # BF16-only inference patches; FP32 runs the plain model
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine, _ = load_engine(args.model_dir, dtype=torch.float32)
    start(engine, args.seed_image)
    specs = workload(args.workload, args.steps, args.seed)
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    noise, x0s = [], []
    for spec in specs:
        x = torch.randn(engine.frm_shape, device='cuda', dtype=torch.float32, generator=gen)
        inputs = engine.prep_inputs(x=x, ctrl=ctrl(spec))
        x0 = engine._denoise_pass(x, inputs, engine.kv_cache).clone()
        engine._cache_pass(x0, inputs, engine.kv_cache)
        noise.append(x.cpu())
        x0s.append(x0.cpu())
    args.golden.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'workload': args.workload, 'seed': args.seed, 'specs': specs, 'seed_image': args.seed_image,
                'noise': torch.stack(noise), 'x0': torch.stack(x0s)}, args.golden)


@torch.inference_mode()
def compare(args):
    ref = torch.load(args.golden, weights_only=True)
    specs, world = ref['specs'], ref['seed_image']
    cand, report = load_engine(args.model_dir, crukx=args.crukx, quant=args.quant)
    start(cand, world)
    base = None
    if args.mode == 'isolated':
        base, _ = load_engine(args.model_dir)
        start(base, world)
    dec = copy.deepcopy(cand.vae)  # canonical decoder: only ever fed golden latents
    rows = []
    for i, spec in enumerate(specs):
        x = ref['noise'][i].to('cuda', torch.bfloat16)
        g = ref['x0'][i].to('cuda')
        g16 = g.to(torch.bfloat16)
        inputs = cand.prep_inputs(x=x, ctrl=ctrl(spec))
        if base is not None:  # hand the candidate the canonical history
            b_inputs = base.prep_inputs(x=x, ctrl=ctrl(spec))
            for lc, lb in zip(cand.kv_cache.layers, base.kv_cache.layers):
                lc.kv.copy_(lb.kv)
                lc.written.copy_(lb.written)
        xc = cand._denoise_pass(x, inputs, cand.kv_cache).clone()
        img_c = copy.deepcopy(dec).decode(xc.squeeze(1))
        img_g = dec.decode(g16.squeeze(1))
        rows.append({'latent_rel_err': rel_err(xc, g), 'pixel_psnr_db': psnr(img_c, img_g)})
        if base is not None:
            base._cache_pass(g16, b_inputs, base.kv_cache)
        elif args.mode == 'history':
            cand._cache_pass(g16, inputs, cand.kv_cache)
        else:
            cand._cache_pass(xc, inputs, cand.kv_cache)
    warm = rows[args.skip:]
    result = {'test': args.mode, 'gpu': torch.cuda.get_device_name(), 'crukx': report, 'quant': args.quant,
              'workload': ref['workload'], 'seed_image': world, 'steps': len(rows), 'skip': args.skip,
              'summary': {k: stats([r[k] for r in warm]) for k in warm[0]}, 'per_step': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('test', 'gpu', 'crukx', 'quant', 'summary')}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('golden', 'isolated', 'history', 'free'))
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--golden', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--crukx', action='store_true')
    ap.add_argument('--quant', help="the engine's own quantization, e.g. intw8a8")
    ap.add_argument('--workload', default='mixed', choices=('mixed', 'still', 'turn', 'abrupt'))
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--skip', type=int, default=16, help='ignore steps before the local window fills')
    ap.add_argument('--seed', type=int, default=73)
    ap.add_argument('--seed-image', help='golden: start the world from a Biome seed image')
    args = ap.parse_args()
    if args.mode == 'golden':
        golden(args)
    else:
        if args.out is None:
            ap.error('--out is required for comparisons')
        compare(args)


if __name__ == '__main__':
    main()
