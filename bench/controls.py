"""Deterministic, human-like Waypoint controls: what a player at a keyboard and mouse sends through Biome.

Biome sends held keys as virtual-key codes and mouse look as raw pixel deltas (x sensitivity 1.8) summed
between frames; the engine feeds them to the model unscaled. Calibrated on Waypoint-1.5: about 200 per
step is a steady look turn, above ~600 the image smears, and below ~60 the camera barely moves.

A player does things in stretches, not per frame: walk while looking around, stop and scan, sprint,
strafe past something, jump. Mouse speed eases toward each new target with a little hand tremor.
The same seed always gives the same sequence, so stock and Crukx are driven identically.
human_player() also reacts like a player: facing a dark or featureless view (a wall), it turns away.
"""
import random

import numpy as np

from world_engine import CtrlInput

W, A, S, D, SPACE, SHIFT, E = 87, 65, 83, 68, 32, 16, 69
MOUSE_LEFT, MOUSE_RIGHT = 1, 2

# (name, weight, keys held, look speed range (px/step, x), duration range (steps))
_ACTIVITIES = (
    ('walk and look', 5, {W}, (-120, 120), (14, 30)),
    ('stop and scan', 2, set(), (-220, 220), (8, 16)),
    ('sprint', 2, {W, SHIFT}, (-60, 60), (12, 24)),
    ('strafe left', 1, {A}, (-80, 40), (6, 12)),
    ('strafe right', 1, {D}, (-40, 80), (6, 12)),
    ('turn around', 1, set(), (180, 260), (8, 12)),
    ('aim', 1, {MOUSE_RIGHT}, (-50, 50), (6, 12)),
    ('step back', 1, {S}, (-60, 60), (4, 8)),
)


def _blocked(frames):
    """A view a player would turn away from: very dark, or one surface filling the screen (a wall).

    Open views have large-scale contrast (sky, street, buildings); a wall up close is one texture, so the
    spread of 16x16 coarse cell means is low. Threshold 28 calibrated on 720p Biome worlds
    (walls 19-26, open views 35-53)."""
    gray = np.asarray(frames[-1], dtype=np.float32).mean(-1)
    h, w = (gray.shape[0] // 16) * 16, (gray.shape[1] // 16) * 16
    cells = gray[:h, :w].reshape(16, h // 16, 16, w // 16).mean((1, 3))
    return gray.mean() < 35 or cells.std() < 28


def human_player(seed=7):
    """Generator of CtrlInputs; send() the frames just generated (or None) to let the player react."""
    rng = random.Random(seed)
    vx = vy = 0.0
    while True:
        name, _, keys, (lo, hi), (dmin, dmax) = rng.choices(_ACTIVITIES, [a[1] for a in _ACTIVITIES])[0]
        target_x = rng.uniform(lo, hi) * (-1 if name == 'turn around' and rng.random() < 0.5 else 1)
        target_y = rng.uniform(-25, 25)
        jump_at = rng.randrange(dmin) if W in keys and rng.random() < 0.3 else -1
        for i in range(rng.randint(dmin, dmax)):
            vx += 0.35 * (target_x - vx) + rng.gauss(0, 6)  # ease toward the target, with tremor
            vy += 0.35 * (target_y - vy) + rng.gauss(0, 3)
            vy *= 0.9  # players re-center vertical look
            held = keys | ({SPACE} if i == jump_at else set())
            frames = yield CtrlInput(button=held, mouse=(round(vx), round(vy)))
            if frames is not None and _blocked(frames) and name != 'turn around':
                break  # next stretch: pick a new activity, biased to turn away
        if frames is not None and _blocked(frames):
            target = rng.choice((-1, 1)) * rng.uniform(200, 260)
            for _ in range(rng.randint(6, 10)):
                vx += 0.35 * (target - vx) + rng.gauss(0, 6)
                frames = yield CtrlInput(button={S} if rng.random() < 0.5 else set(), mouse=(round(vx), 0))


def human_controls(steps, seed=7):
    """A fixed list of `steps` CtrlInputs (open loop: identical for every run)."""
    player = human_player(seed)
    return [next(player) for _ in range(steps)]


if __name__ == '__main__':
    seq = human_controls(400)
    assert seq == human_controls(400) and len(seq) == 400  # deterministic
    assert max(abs(c.mouse[0]) for c in seq) < 600  # inside the calibrated range
    assert sum(W in c.button for c in seq) > 100  # mostly moving through the world
    print('ok', sum(SHIFT in c.button for c in seq), 'sprint steps,', sum(SPACE in c.button for c in seq), 'jumps')
