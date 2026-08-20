"""The keystroke cue for the four Playhook sound sets: one key on the on-screen keyboard.

**The fastest-repeating sound in the launcher.** Typing a game name on a gamepad keyboard fires
this several times a second, dozens of times in a row, and every constraint follows from that:

- **shortest of the family** — 40-110 ms against that set's own `move` of 139-1021 ms. A key is
  the smallest event an interface has, and anything longer starts overlapping its own repeats;
- **quietest of the family** — around -15 dBFS peak, below `limit`'s -12 and `move`'s -9;
- **almost dry** — reverb at rt60 0.1-0.2 s and a narrow image. Tails are what actually ruin a
  repeating sound: at 0.5 s the wash from ten keystrokes is continuous, whatever each one
  sounds like alone;
- **noise-led, not pitch-led.** This is the important one. A clearly pitched click repeated
  thirty times reads as a monotone drone, so the tonal part is quiet, high and damped hard —
  it is there to say which sound set you are in, not to be a note. The transient carries the
  sound, which is also how a real keyboard works.

**A single file is a real limitation.** A physical keyboard never repeats exactly; this does,
because the launcher plays one file per slot and clones it per press. That is the one thing
here a measurement cannot fix — it needs several files and a consumer that round-robins them.
If that ever exists, the `seed` in each profile is the only thing that has to change to render
alternates.

**Rooted an octave above the set's `move`**, so a keystroke reads as a smaller object than a
navigation step in the same set — the same instrument, struck lighter and higher.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from sfxsmith.engine import (bp, env_ar, env_perc, hz, mix, noise, sine, soft_clip, voice)

PEAK_DB = {
    'typing': -15.0,
    # Abyss is lifted for the reason documented in packs/panel.py: peak level is not loudness,
    # and this set's material sits low enough that matching its siblings' peak leaves it several
    # decibels quieter to the ear.
    'playhook-abyss/typing': -11.5,
}


@dataclass(frozen=True)
class Profile:
    """One set's keystroke, in the terms of that set's own `move`.

    `tap_lo`/`tap_hi` bound the transient's band; `body` is the note the damped tonal part
    rings at; `thock` is the optional low knock under it, which only the sets with real sub
    content get.
    """

    body: str
    tap_lo: float
    tap_hi: float
    tap_decay: float
    tap_amp: float
    ring: float
    ring_amp: float
    tilt: float
    width: float
    rt60: float
    seed: int
    thock: str | None = None
    thock_amp: float = 0.3


PROFILES: dict[str, Profile] = {
    # move: centroid 552 Hz, root E3 — the darkest set, so the lowest and widest tap band and
    # the only keystroke here with a real knock under it.
    'playhook-abyss': Profile('E4', 160, 1400, 0.011, 0.55, 0.040, 0.30, 0.60, 0.10, 0.20, 601,
                              thock='E2', thock_amp=0.42),
    # move: centroid 632 Hz, root E4. Bell material, so its ring is the most audible of the
    # four — but still damped to 35 ms, an order below the set's own 823 ms move.
    'playhook-aurora': Profile('E5', 260, 2200, 0.009, 0.5, 0.028, 0.34, 0.50, 0.12, 0.18, 603,
                               thock='E3', thock_amp=0.22),
    # move: centroid 1639 Hz, onset 3 ms — the brightest and snappiest set, so the tightest tap.
    'playhook-cartridge': Profile('E6', 480, 2900, 0.006, 0.50, 0.024, 0.38, 0.40, 0.09, 0.14, 605),
    # move: 139 ms, onset 2.5 ms. The shortest sound in the whole family lands here.
    'playhook-tactile': Profile('E6', 420, 3100, 0.004, 0.62, 0.015, 0.28, 0.44, 0.06, 0.10, 607,
                                thock='E2', thock_amp=0.35),
}


def tap(p: Profile, dur: float) -> np.ndarray:
    """The transient: band-limited noise, a few milliseconds long, carrying the sound.

    Band-passed rather than high-passed, for the reason this repo keeps relearning — an open
    top end turns a click into static. The band is wide here because a keystroke is broadband
    by nature; it is the DECAY that keeps it from reading as noise.
    """
    return bp(noise(dur, seed=p.seed), p.tap_lo, p.tap_hi) * env_perc(dur, 0.0006, p.tap_decay) * p.tap_amp


def ring(p: Profile, dur: float) -> np.ndarray:
    """The damped tonal part: which sound set this is, in as few milliseconds as possible."""
    f = hz(p.body)
    tone = sine(f, dur) + sine(f * 2.0, dur) * 0.22 + sine(f * 3.1, dur) * 0.08
    return tone * env_ar(dur, 0.0008, p.ring, curve=4.4) * p.ring_amp


def thock(p: Profile, dur: float) -> np.ndarray:
    """The low knock under the click — the key bottoming out, for the sets that have low end."""
    if p.thock is None:
        return np.zeros(int(dur * 48000))
    return sine(hz(p.thock), dur) * env_ar(dur, 0.0015, p.ring * 1.4, curve=3.8) * p.thock_amp


def typing(p: Profile) -> np.ndarray:
    """One keystroke: a short transient, a damped ring, and a knock where the set has one."""
    dur = max(p.ring * 3.2, p.tap_decay * 6) + 0.02
    return voice(soft_clip(mix(tap(p, dur), ring(p, dur), thock(p, dur)), 1.2) * 0.5,
                 width=p.width, rt60=p.rt60, seed=p.seed + 41, tilt=p.tilt)


PACKS = {name: {'typing': partial(typing, profile)} for name, profile in PROFILES.items()}
