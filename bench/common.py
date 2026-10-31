"""Shared helpers: the pinned engine, optional Crukx, controls, seed worlds and statistics."""
import math
import random

import numpy as np
import torch
from world_engine import CtrlInput, WorldEngine

ENGINE_COMMIT = '68b316390d576dd9f671f552803526bbf8eb2ed5'
CRUKX_MISSING = ('--crukx needs the crukx_waypoint package, which is distributed separately. '
                 'See https://crukx.dev or open an issue on this repository.')


def load_engine(model_dir, crukx=False, quant=None, dtype=torch.bfloat16, **kw):
    """The stock engine; with crukx=True, the same engine with crukx_waypoint.accelerate applied.

    accelerate() times its kernels against stock on this GPU and keeps stock where they do not win,
    so the returned report says what was actually enabled."""
    engine = WorldEngine(model_dir, device='cuda', dtype=dtype, quant=quant, **kw)
    report = None
    if crukx:
        try:
            import crukx_waypoint
        except ImportError:
            raise SystemExit(CRUKX_MISSING)
        report = crukx_waypoint.accelerate(engine)
    return engine, report


def require_crukx():
    try:
        import crukx_waypoint
    except ImportError:
        raise SystemExit(CRUKX_MISSING)
    return crukx_waypoint


def seed_frames(path):
    """A Biome seed image as a world's first 4 frames, as examples/gen_sample.py starts a world."""
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB').resize((1280, 720), Image.BICUBIC))
    return torch.from_numpy(np.repeat(arr[None], 4, axis=0))


def ctrl(spec):
    return CtrlInput(button=set(spec['button']), mouse=tuple(spec['mouse']), scroll_wheel=spec['scroll'])


def workload(name, n, seed=73):
    """Frozen control sequences, identical for every run."""
    rng = random.Random(seed)
    if name == 'mixed':
        return [{'button': sorted({(i % 7) + 1} if i % 3 else {(i % 7) + 1, 12}),
                 'mouse': (round(0.05 * (i % 5) - 0.1, 4), round(0.03 * (i % 3) - 0.03, 4)),
                 'scroll': (i % 3) - 1 if i % 4 == 0 else 0} for i in range(n)]
    if name == 'still':
        return [{'button': [], 'mouse': (0.0, 0.0), 'scroll': 0} for _ in range(n)]
    if name == 'turn':  # sustained camera turns reversing every 16 steps
        return [{'button': [], 'mouse': (0.4 if (i // 16) % 2 == 0 else -0.4, 0.0), 'scroll': 0} for i in range(n)]
    if name == 'abrupt':  # a new random control state every 4 steps
        out, cur = [], None
        for i in range(n):
            if i % 4 == 0:
                cur = {'button': sorted(rng.sample(range(1, 16), rng.randint(0, 2))),
                       'mouse': (round(rng.uniform(-0.5, 0.5), 3), round(rng.uniform(-0.3, 0.3), 3)),
                       'scroll': rng.choice((-1, 0, 0, 1))}
            out.append(cur)
        return out
    raise ValueError(name)


def percentiles(xs, qs=(50, 95, 99)):
    return {f'p{q}': round(float(np.percentile(xs, q)), 2) for q in qs}


def stats(xs):
    xs = np.asarray(xs, dtype=float)
    return dict(zip(('mean', 'p5', 'p50', 'p95'), [round(float(v), 4) for v in (xs.mean(), *np.percentile(xs, [5, 50, 95]))]))


def psnr(a, b):
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def rel_err(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float())).item()
