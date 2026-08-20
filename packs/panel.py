"""Open / close cues for the launcher's sliding panels, for the four Playhook sound sets.

**Two references, and they disagree about almost everything.** Both are shipped popup sounds:
the Xbox 360's and Steam Big Picture's.

| | peak | audible | onset | centroid | rolloff85 | flatness |
|---|---|---|---|---|---|---|
| Xbox open | -10.5 dBFS | 563 ms | 153 ms | 1570 Hz | 1227 Hz | 0.130 |
| Xbox close | -11.9 | 612 ms | 205 ms | 1652 Hz | 1529 Hz | 0.143 |
| Big Picture open | **-17.6** | 693 ms | 97 ms | **493 Hz** | **538 Hz** | **0.038** |
| Big Picture close | **-21.4** | 634 ms | 116 ms | **469 Hz** | **495 Hz** | **0.035** |

Big Picture's panel is seven to ten decibels quieter, three times darker and four times more
tonal than the Xbox's. Its loudest partial is at **43 Hz**, with 37-41% of its energy under
200 Hz and 85% under 540 Hz — a soft low swell you feel more than hear, where the Xbox's is a
bright whoosh. This pack follows Big Picture: a panel opening is not an event, and a sound that
announces one every time a menu appears is what makes an interface tiring.

**Construction, in order of how much of the sound each layer is.** The balance is the whole
point, and it is inverted from the first attempt here, which led with noise and measured
0.044-0.143 flatness at -12 dBFS — the thing that was too loud and too hissy:

1. a **low anchor**, a sine at the set's sub note, carrying the weight the way the reference's
   43 Hz partial does;
2. a **tonal core** — the set's root and the fifth above it, the fifth fading in as a panel
   opens and out as it closes, so the interval itself opens and closes;
3. a **narrow swept band** of noise, quiet, as texture only. Narrow on purpose: `Q_WIDTH` at
   2.4 rather than the 3.1 of the noisy version, because a wide band of noise is exactly the
   hiss that had to go;
4. a **wideband floor** at 0.8% of that band. Inaudible on its own, and the only way to reach
   the reference's flatness: at 0.4% the pack measured 0.025 against its 0.038, at 4% it shot
   to 0.129 and dragged the centroid from 559 Hz to 1613. That layer is the single most
   sensitive number in this file.

**The envelope** is the references' shared shape — both swell rather than strike: level climbs
for around 100 ms, holds, then falls away. Neither has a transient anywhere.

**One deliberate departure.** In both references, open and close are nearly the same sound
(Xbox's close is even a whole tone higher) — fine when a panel is visibly moving on screen.
Here they must be told apart by ear, so direction carries the meaning: **open rises, close
falls**, and close ends on a soft latch, quiet enough to be a settling rather than a knock.
Close is also 3.5 dB below open, following Big Picture's own gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from sfxsmith.engine import (SR, bp, env_ahd, env_perc, hz, lp, mix, noise, sine, soft_clip,
                             step, sweep_filter, voice)

# The reference's own levels, and its own gap between the two: Big Picture's popup sits far
# below anything else in its interface. Against Playhook's ladder (move -9, play -1) that puts
# a panel roughly eight decibels under a single step of navigation, which is the intent.
PEAK_DB = {
    'popup-open': -17.5,
    'popup-close': -21.0,
    # Abyss is lifted because peak level is not loudness. At the same peak as its siblings it
    # measured 7.7 dB quieter A-weighted (-47.1 dB against Cartridge's -39.6) — its energy sits
    # on a 41 Hz fundamental, where the ear is far less sensitive. These two values put it level
    # with Aurora by that measure while keeping it the deepest of the four.
    'playhook-abyss/popup-open': -13.2,
    'playhook-abyss/popup-close': -16.9,
}

# Width of the swept band either side of its centre. Narrow: this layer is texture, not the
# sound. At 3.1 it measured as hiss.
Q_WIDTH = 2.4

# How loud each layer is relative to the anchor. The anchor leads, which is what makes the cue
# read as deep rather than as bright.
ANCHOR_AMP = 0.38
CORE_AMP = 0.45
BAND_AMP = 0.70
LATCH_AMP = 0.15

# A wideband floor under everything, at 0.8% of the swept band. Not audible as hiss at these
# levels — it exists because flatness is a geometric mean, and a sound confined below 1.5 kHz
# has thousands of near-empty bins above it dragging the figure to a tenth of the reference's.
FLOOR_AMP = 0.008


@dataclass(frozen=True)
class Profile:
    """One set's panel pair, anchored to that set's own `move`.

    `sub` carries the weight, `root` the pitch; `low`/`high` are the ends of the filter sweep
    in Hz, bottom to top on open and reversed on close. `dur` scales with how fast that set
    moves in general.
    """

    sub: str
    root: str
    low: float
    high: float
    dur: float
    tilt: float
    width: float
    rt60: float
    seed: int


PROFILES: dict[str, Profile] = {
    # move: centroid 552 Hz, root E3, 1021 ms — the slowest and darkest set, so the deepest
    # anchor of the four and the slowest panel.
    'playhook-abyss': Profile('E1', 'E3', 130, 430, 0.68, 0.42, 0.34, 0.55, 501),
    # move: centroid 632 Hz, root E4. Bell material, so its tonal core carries further inside
    # the noise than the others do.
    'playhook-aurora': Profile('E1', 'E4', 200, 560, 0.62, 0.36, 0.4, 0.45, 503),
    # move: centroid 1639 Hz, root E5 — the brightest set, so the highest anchor and the widest
    # sweep. Still nowhere near where this pack's first version sat.
    'playhook-cartridge': Profile('E2', 'E4', 300, 820, 0.56, 0.34, 0.3, 0.38, 505),
    # move: centroid 1164 Hz, 334 ms — the fastest set, so the shortest panel of the four.
    'playhook-tactile': Profile('E2', 'E4', 260, 700, 0.46, 0.36, 0.22, 0.3, 507),
}


def swell(dur: float, opening: bool) -> np.ndarray:
    """The references' shared shape: a rise, a short plateau, then a fall — never a strike.

    Open rises fastest and lets go; close takes longer to build and then shuts. That asymmetry
    is in both references (onsets of 97 against 116 ms in Big Picture, 153 against 205 in the
    Xbox), so it survives here even though everything around it changed.
    """
    if opening:
        return env_ahd(dur, dur * 0.12, dur * 0.18, dur * 0.34, curve=2.6)
    return env_ahd(dur, dur * 0.16, dur * 0.20, dur * 0.26, curve=3.2)


def anchor(p: Profile, dur: float) -> np.ndarray:
    """The low swell that carries the cue: a sub sine with a touch of its own octave.

    This is the layer that answers "deeper". The reference's loudest partial is at 43 Hz and it
    holds 37-41% of the energy below 200 Hz; a panel built from mid-band material alone cannot
    get there however dark its filter is set.
    """
    f = hz(p.sub)
    return (sine(f, dur) + sine(f * 2, dur) * 0.4) * ANCHOR_AMP


def core(p: Profile, dur: float, opening: bool) -> np.ndarray:
    """The pitch centre: the set's root and the fifth above, the fifth fading in on open and
    out on close, so the interval opens and closes with the panel."""
    root = hz(p.root)
    fifth = np.linspace(0.1, 1.0, int(dur * SR))
    if not opening:
        fifth = fifth[::-1]
    return (sine(root, dur) * 0.8 + sine(step(root, 7), dur) * 0.5 * fifth) * CORE_AMP


def band(p: Profile, dur: float, opening: bool) -> np.ndarray:
    """A narrow band of noise swept between the profile's two centres — texture, nothing more.

    Low-passed after sweeping: `sweep_filter` works in blocks, and the block edges leave a
    little wideband energy up top that is inaudible on its own but reads as hiss once the rest
    of the sound is this quiet.
    """
    f0, f1 = (p.low, p.high) if opening else (p.high, p.low)
    swept = lp(sweep_filter(noise(dur, seed=p.seed), f0, f1, q_width=Q_WIDTH), p.high * 3.0)
    floor = bp(noise(dur, seed=p.seed + 5), p.high, 12000) * FLOOR_AMP
    return swept * BAND_AMP + floor


def latch(p: Profile, dur: float) -> np.ndarray:
    """The soft settle that ends a close, placed where the swell drops.

    Neither reference has one — it is what replaces the visual confirmation that a panel has
    finished moving, given that these two sounds have to be distinguishable without looking at
    the screen. Kept low and quiet so it settles rather than knocks.
    """
    n = int(dur * SR)
    hit = bp(noise(dur, seed=p.seed + 9), p.low * 0.5, p.high) * env_perc(dur, 0.004, 0.05)
    placed = np.zeros(n)
    off = int(dur * 0.76 * SR)
    placed[off:] = hit[:n - off]
    return placed * LATCH_AMP


def panel(p: Profile, opening: bool) -> np.ndarray:
    """One panel cue: a low swell with a tonal core, a breath of noise, and a latch on close."""
    dur = p.dur
    env = swell(dur, opening)
    layers = [anchor(p, dur) * env, core(p, dur, opening) * env, band(p, dur, opening) * env]
    if not opening:
        layers.append(latch(p, dur))
    return voice(soft_clip(mix(*layers), 1.1) * 0.55,
                 width=p.width, rt60=p.rt60, seed=p.seed + 41, tilt=p.tilt)


PACKS = {name: {'popup-open': partial(panel, profile, True),
                'popup-close': partial(panel, profile, False)}
         for name, profile in PROFILES.items()}
