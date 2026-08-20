"""The dead-end cue for the four Playhook sound sets: navigation pushed against a wall.

**What this sound has to survive.** It is the one cue in a launcher that fires in bursts — hold
a stick against the end of a row and it repeats several times a second. Every design decision
here follows from that, and they all point the opposite way from the notification cue:

- **quiet**, at -12 dBFS against `move`'s -9 and `play`'s -1, so a held stick never shouts;
- **short**, about half that set's own `move`, so repeats do not overlap into a drone. The
  exception is Tactile, whose `move` is only 139 ms to begin with — there the cue lands at 83%
  of it, because below ~100 ms the bend stops being audible as a pitch at all;
- **dry**, with the reverb roughly halved: a blocked move should not open up space, it should
  stop. This is the part that reads as a wall rather than as a quieter step;
- **darker than `move`**, because a repeating bright sound is what makes an interface feel like
  it is nagging.

**Why it reads as "no".** The body is the set's own `move` material, bent DOWN a whole tone over
the first 45-90 ms and then held there. Downward is the whole vocabulary: nothing in a working
interface bends down, so the ear does not need to learn what it means. A whole tone rather than
a semitone on purpose — a semitone reads as the right note played badly, which is a worse thing
to say about the interface than "there is nothing here".

Under it sits a short band-limited thud, the impact itself, tuned per set: 60 ms for Abyss's
slow low material, 18 ms for Tactile, which is over almost before it starts.

**Rooted where the set's own `move` is rooted**, measured rather than assumed: E3 for Abyss,
E4 for Aurora, E5 for Cartridge and Tactile. The cue is the same instrument as the step it
refuses, which is what keeps it from sounding like an error from another program.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from sfxsmith.engine import (SR, bp, env_ar, env_perc, glide, hz, inharmonic, mix, noise,
                             sine, soft_clip, step, voice)

PEAK_DB = {'limit': -12.0}

# How far the body bends down, in semitones. A whole tone: far enough to be unambiguous, close
# enough that it still belongs to the set's own pitch material.
BEND = 2.0

SHIMMER_RATIOS = [1.0, 1.51, 2.13, 2.79]


@dataclass(frozen=True)
class Profile:
    """One set's dead-end cue, in the terms of that set's own `move`.

    `root` is where its `move` measures; `fall` is how long the bend takes; `decay`, `tilt`,
    `width` and `rt60` are all pulled in from `move`'s values — shorter, darker and drier.
    """

    root: str
    fall: float
    decay: float
    thud: float
    thud_lo: float
    thud_hi: float
    tilt: float
    width: float
    rt60: float
    seed: int
    shimmer: bool = False
    sub: str | None = None
    sub_amp: float = 0.4


PROFILES: dict[str, Profile] = {
    # move: E3 root, 1914 ms, centroid 696 Hz, 36% under 200 Hz, correlation 0.35. The slowest
    # set, so the slowest bend and the longest thud — but a third of the length.
    'playhook-abyss': Profile('E3', 0.09, 0.24, 0.32, 90, 700, 0.62, 0.22, 0.7, 401,
                              sub='E2', sub_amp=0.55),
    # move: E4 with a G#4 above it, 823 ms, centroid 729 Hz. Bell material, so the bend is
    # audible as a pitch rather than as a bump.
    'playhook-aurora': Profile('E4', 0.075, 0.12, 0.24, 140, 1100, 0.56, 0.24, 0.35, 403,
                               sub='E3', sub_amp=0.3),
    # move: E5, 686 ms, centroid 1639 Hz, onset 3 ms. Keeps the metal-plate shimmer, damped to
    # a quarter of its notification length — the plate is struck and stopped.
    'playhook-cartridge': Profile('E5', 0.055, 0.09, 0.24, 260, 2000, 0.46, 0.12, 0.28, 405,
                                  shimmer=True, sub='E4', sub_amp=0.22),
    # move: E5/F5, 334 ms, onset 2.5 ms — the shortest in the family, so this is the shortest
    # cue here: bend, thud, gone.
    'playhook-tactile': Profile('E5', 0.04, 0.07, 0.28, 200, 1800, 0.5, 0.08, 0.22, 407,
                                sub='E2', sub_amp=0.5),
}

PARTIALS = [(1.0, 1.0, 1.0), (2.0, 0.3, 0.55), (3.0, 0.12, 0.35), (4.4, 0.05, 0.22)]


def drop(f0: float, dur: float, fall: float) -> np.ndarray:
    """Per-sample frequency that falls `BEND` semitones over `fall` seconds, then holds.

    Held rather than left falling: a glide that never lands reads as a slide, and a slide is a
    transition. This one arrives somewhere wrong and stays there.
    """
    n = int(dur * SR)
    k = min(int(fall * SR), n)
    target = step(f0, -BEND)
    return np.concatenate([glide(f0, target, fall)[:k], np.full(n - k, target)])[:n]


def bent_body(f0: float, dur: float, p: Profile) -> np.ndarray:
    """The set's tonal material, every partial bending down together."""
    out = np.zeros(int(dur * SR))
    for ratio, amp, dm in PARTIALS:
        out += sine(drop(f0 * ratio, dur, p.fall), dur) * amp * env_ar(
            dur, 0.004, p.decay * dm, curve=3.6)
    return out


def thud(dur: float, p: Profile) -> np.ndarray:
    """The impact: a band-limited burst, the width of the set's own low-mid weight."""
    burst = bp(noise(dur, seed=p.seed), p.thud_lo, p.thud_hi)
    return burst * env_perc(dur, 0.002, p.decay * 0.28) * p.thud


def limit(p: Profile) -> np.ndarray:
    """One dead-end cue: bent body, impact under it, and nothing left ringing."""
    dur = p.decay * 3.0 + p.fall
    root = hz(p.root)
    layers = [bent_body(root, dur, p), thud(dur, p)]

    if p.shimmer:
        plate = inharmonic(root * 2, dur, SHIMMER_RATIOS, amp=0.12, seed=p.seed + 5,
                           decay=p.decay * 0.22)
        layers.append(plate)
    if p.sub is not None:
        low = sine(drop(hz(p.sub), dur, p.fall * 1.4), dur)
        layers.append(low * env_ar(dur, 0.006, p.decay * 0.7, curve=3.2) * p.sub_amp)

    return voice(soft_clip(mix(*layers), 1.3) * 0.55,
                 width=p.width, rt60=p.rt60, seed=p.seed + 41, tilt=p.tilt)


PACKS = {name: {'limit': partial(limit, profile)} for name, profile in PROFILES.items()}
