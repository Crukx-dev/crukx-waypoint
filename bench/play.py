#!/usr/bin/env python3
"""Play Biome seed worlds like a player and record what they see, with startup and fps timings.

One process plays every listed world in turn with one engine, so the first world pays the compile
and autotune cost and later worlds show what a new world costs once kernels exist. Run it twice with
the same TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR to measure a start on a warm kernel cache.

    play.py --model-dir MODEL --seeds seeds/a.jpg seeds/b.jpg --tag 720_stock --out DIR [--crukx]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

T0 = time.perf_counter()  # process start, before torch is imported

import torch  # noqa: E402

from common import load_engine, seed_frames  # noqa: E402
from controls import human_controls  # noqa: E402


def sync_ms(t):
    torch.cuda.synchronize()
    return round((time.perf_counter() - t) * 1000, 1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--crukx', action='store_true')
    ap.add_argument('--steps', type=int, default=240)
    ap.add_argument('--seed', type=int, default=73)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    engine, report = load_engine(args.model_dir, crukx=args.crukx)
    result = {'tag': args.tag, 'gpu': torch.cuda.get_device_name(), 'crukx': report,
              'import_s': round(t - T0, 1), 'load_ms': sync_ms(t), 'worlds': []}
    controls = human_controls(args.steps)
    for path in args.seeds:
        name = Path(path).stem
        torch.manual_seed(args.seed)
        t = time.perf_counter()
        engine.reset()
        first = engine.append_frame(seed_frames(path))
        world = {'world': name, 'seed_ms': sync_ms(t)}
        h, w = first.shape[-3:-1]
        enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                '-s', f'{w}x{h}', '-r', '60', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                '-crf', '18', str(args.out / f'{name}_{args.tag}.mp4')], stdin=subprocess.PIPE)
        enc.stdin.write(first.cpu().numpy().tobytes())
        steps = []
        for i, c in enumerate(controls):
            s = time.perf_counter()
            frames = engine.gen_frame(ctrl=c)
            steps.append(sync_ms(s))
            if i == 0 and not result['worlds']:
                world['ready_to_first_frame_s'] = round(time.perf_counter() - T0, 1)
            enc.stdin.write(frames.cpu().numpy().tobytes())
        enc.stdin.close()
        enc.wait()
        p50 = sorted(steps[10:])[len(steps[10:]) // 2]
        world.update({'steps': len(steps), 'step_ms_p50': p50, 'fps_p50_with_decode': round(4000 / p50, 1)})
        result['worlds'].append(world)
        print(json.dumps(world), flush=True)
    (args.out / f'{args.tag}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
