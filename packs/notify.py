"""A notification cue for each of the four Playhook sound sets, built from two real references.

**What the references taught, and where the first attempt went wrong.** Two shipped achievement
sounds were transcribed frame by frame: Steam Deck's `deck_ui_achievement_toast` and the Xbox
360's `Achievement`. The first version here was designed to blend into its sound set. Both
references do the opposite:

| | centroid | flatness | the UI set it ships with |
|---|---|---|---|
| Deck toast | 2179 Hz | 0.113 | Steam Big Picture, 580-1119 Hz |
| Xbox 360 achievement | 1447 Hz | 0.072 | Xbox, whose navigation sits at 1483 Hz |

The Deck's notification is **two to four times brighter than its own interface**, and both are
noisier than the clean tones around them. That is the point of the sound: it interrupts. The
first attempt matched each set's measured centroid exactly and produced four cues that are
polite and easy to miss — 304-1216 Hz, flatness 0.006-0.039, and up to three seconds long.

**Two shapes, because the references are two different ideas.**

- `toast` follows the Deck: a short noisy swell (~150 ms), then a chord that arrives stacked —
  the top note first, the octave below it 40 ms later, the fifth 100 ms after that — and rings
  for half a second. The Deck's is C6 / C5 / G5 over an F4 bed; ours is the same shape in E.
  Total 774 ms there, ~950 ms here.
- `chime` follows the Xbox: two notes an OCTAVE apart, 70 ms apart — not the fifth the first
  attempt used, and four times faster than its 260 ms. Nearly all the energy is spent inside
  250 ms; what remains is a quiet low hum under the decay. (The Xbox's C#5 to D6 measures 2.03x
  in frequency: it reads as an odd interval only because that sound is not tuned to A440.)

**What still comes from the set.** Timbre construction (`voice`), stereo width, noise level and
the pitch register are per-set, so Abyss's cue is still the dark wide one and Tactile's is still
the one led by a click. What is no longer per-set is the brightness ceiling: every cue is lifted
toward the references rather than toward its own set's average.

**Shipped:** `chime`, in all four sets, installed as `playhook/audio/ui/<set>/notify.wav`.
`toast` is kept as the alternative it was chosen against. Nothing in the launcher reads either
file yet — `SfxName` has no notification slot.

```bash
sfxsmith render notify -o out
cp out/<set>/notify-chime.wav ../playhook/audio/ui/<set>/notify.wav
```
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from sfxsmith.engine import (additive, at, bp, env_ar, env_perc, fm_voice, hz, inharmonic,
                             mix, noise, sine, soft_clip, step, voice)

PEAK_DB = {'notify-toast': -3.0, 'notify-chime': -3.0}

SHIMMER_RATIOS = [1.0, 1.51, 2.13, 2.79, 3.61]

TOAST_LEN = 0.95
CHIME_LEN = 0.75

# Where the chord's three notes land, in seconds, transcribed from the Deck toast: its top note
# enters at 200 ms, the octave below it at 240 ms, the fifth at 300 ms. Staggering them is what
# makes the chord read as arriving rather than as a block.
TOAST_ENTRY = (0.08, 0.11, 0.17)

# The Xbox pair: the second note is an octave above the first, 70 ms later.
CHIME_GAP = 0.07


@dataclass(frozen=True)
class Profile:
    """One set's cue, in the terms its own sounds were measured in.

    `voice` picks the construction and `top` the register — both per set. `tilt` is deliberately
    lower than the set's own sounds would suggest, because a notification is supposed to be
    brighter than the interface it interrupts.
    """

    voice: str
    top: str
    tilt: float
    noise: float
    width: float
    rt60: float
    seed: int
    sub: str | None = None
    sub_amp: float = 0.4


PROFILES: dict[str, Profile] = {
    # The dark set (centroid 324-696 Hz, up to 73% of its energy under 200 Hz), so its cue lands
    # an octave below the others and keeps a sub — but at tilt 0.42 rather than the set's 0.62,
    # or it would be the one notification nobody notices.
    'playhook-abyss': Profile('weight', 'E5', 0.30, 0.10, 0.46, 1.1, 305, sub='E2', sub_amp=0.5),
    # FM bells, the most tonal set (flatness 0.011-0.033). The cue keeps the bell construction
    # and gains the reference's noise floor, which is what a shipped notification actually has.
    'playhook-aurora': Profile('bell', 'E6', 0.26, 0.09, 0.34, 0.9, 307, sub='E3', sub_amp=0.3),
    # The hybrid, and the only one to keep the inharmonic metal-plate layer — that shimmer is
    # this set's whole point.
    'playhook-cartridge': Profile('glass', 'E6', 0.22, 0.10, 0.28, 0.75, 309, sub='E4',
                                  sub_amp=0.25),
    # Onsets of 1-5 ms against Aurora's 24-406: the cue stays the one led by its click.
    'playhook-tactile': Profile('tick', 'E6', 0.40, 0.11, 0.2, 0.45, 311, sub='E3', sub_amp=0.45),
}


def grit(dur: float, seed: int, amp: float, lo: float, hi: float, decay: float = 0.05) -> np.ndarray:
    """Noise in two parts: an audible core around the note, plus a quiet wideband floor.

    The floor is there to hit a measured flatness. Flatness is a geometric mean, so the
    near-zero bins between clean sine partials drag a purely additive sound to ~0.005 — an
    order of magnitude under both references. Filling them quietly restores the number.
    """
    core = bp(noise(dur, seed=seed), lo, hi) * env_perc(dur, 0.003, decay)
    floor = bp(noise(dur, seed=seed + 7), hi * 0.7, 12000) * env_perc(dur, 0.005, decay * 1.6) * 0.3
    return (core + floor) * amp


def bell(f: float, dur: float, attack: float, decay: float, p: Profile, seed: int) -> np.ndarray:
    """FM bell over a harmonic body — Aurora's construction, and the most tonal of the four."""
    body = fm_voice(f, 2.0, 1.6, dur, idx_decay=15) * 0.5
    body += additive(f, [(1, 1.0, 1.0), (2, 0.32, 0.5), (3, 0.14, 0.32),
                         (4.2, 0.06, 0.2)], dur, decay * 0.8) * 0.7
    body *= env_ar(dur, attack, decay, curve=3.0)
    return body + grit(dur, seed, p.noise, f * 0.6, f * 4.5, decay * 0.3)


def glass(f: float, dur: float, attack: float, decay: float, p: Profile, seed: int) -> np.ndarray:
    """Harmonic stack under a short inharmonic shimmer at metal-plate ratios — Cartridge."""
    body = additive(f, [(1, 1.0, 1.0), (2, 0.45, 0.55), (3, 0.24, 0.35),
                        (4, 0.12, 0.25), (6, 0.05, 0.16)], dur, decay * 0.7)
    body *= env_ar(dur, attack, decay, curve=3.4)
    shine = inharmonic(f * 2, dur, SHIMMER_RATIOS, amp=0.18, seed=seed, decay=decay * 0.3)
    return body + shine + grit(dur, seed + 3, p.noise, f * 0.9, f * 6, decay * 0.25)


def tick(f: float, dur: float, attack: float, decay: float, p: Profile, seed: int) -> np.ndarray:
    """Band-limited transient with a tone under it — Tactile, the set led by its click.

    The noise is band-passed around the note rather than high-passed: an open top end is what
    pushed the first Playhook build to 0.5 flatness on sounds meant to measure 0.03.
    """
    click = bp(noise(dur, seed=seed), f * 1.1, min(f * 6, 15000))
    click *= env_perc(dur, max(attack, 0.001), decay * 0.3)
    body = additive(f, [(1, 1.0, 1.0), (2, 0.34, 0.5), (3, 0.13, 0.3)], dur, decay * 0.55)
    body *= env_ar(dur, max(attack, 0.001), decay * 0.75, curve=4.0)
    return click * (0.4 + p.noise) + body * (0.66 - p.noise * 0.5)


def weight(f: float, dur: float, attack: float, decay: float, p: Profile, seed: int) -> np.ndarray:
    """Low body with its own sub-octave partial — Abyss, the dark one."""
    body = additive(f, [(1, 1.0, 1.0), (0.5, 0.7, 1.2), (2, 0.34, 0.5),
                        (3, 0.14, 0.3)], dur, decay * 0.9)
    body *= env_ar(dur, attack, decay, curve=2.8)
    return body + grit(dur, seed, p.noise, f * 0.5, f * 4, decay * 0.35)


VOICES = {'bell': bell, 'glass': glass, 'tick': tick, 'weight': weight}


def finish(p: Profile, *layers: np.ndarray) -> np.ndarray:
    """Shared output stage: saturate, spread to the set's measured width, tilt the top down."""
    return voice(soft_clip(mix(*layers), 1.4) * 0.6,
                 width=p.width, rt60=p.rt60, seed=p.seed + 41, tilt=p.tilt)


def toast(p: Profile) -> np.ndarray:
    """The Deck reading: a noisy swell, then a stacked chord that arrives and rings out.

    The chord is the top note, the octave below it and the fifth between them — the Deck's
    C6/C5/G5 shape — entering in that order, which is what makes it sound like it is landing
    rather than being switched on.
    """
    dur = TOAST_LEN
    render = VOICES[p.voice]
    top = hz(p.top)
    notes = (top, step(top, -12), step(top, -5))
    chord = [at(render(f, dur, 0.05 + i * 0.012, 0.40 - i * 0.03, p, p.seed + i * 13)
                * (0.85 ** i), TOAST_ENTRY[i] * 1000, dur)
             for i, f in enumerate(notes)]

    swell = bp(noise(dur, seed=p.seed + 71), top * 0.18, top * 1.5)
    swell *= env_ar(dur, 0.08, 0.09, curve=3.2) * (0.22 + p.noise)
    bed = sine(step(top, -24), dur) * env_ar(dur, 0.12, 0.30, curve=2.6) * 0.35

    layers = [*chord, swell, bed]
    if p.sub is not None:
        low = sine(hz(p.sub), dur) * env_ar(dur, 0.06, 0.34, curve=2.8)
        layers.append(low * p.sub_amp)
    return finish(p, *layers)


def chime(p: Profile) -> np.ndarray:
    """The Xbox reading: two notes an octave apart, 70 ms apart, over almost as fast.

    Nearly all of the energy is spent inside 250 ms — the reference's second note has decayed
    by 120 ms. What is left afterwards is a low hum two octaves down, quiet enough to be felt
    rather than heard, which is the whole of its tail.
    """
    dur = CHIME_LEN
    render = VOICES[p.voice]
    top = hz(p.top)
    first = render(step(top, -12), dur, 0.006, 0.11, p, p.seed) * 0.8
    second = at(render(top, dur, 0.004, 0.13, p, p.seed + 17), CHIME_GAP * 1000, dur)
    hum = sine(step(top, -24), dur) * env_ar(dur, 0.05, 0.32, curve=3.0) * 0.28

    layers = [first, second, hum]
    if p.sub is not None:
        low = sine(hz(p.sub), dur) * env_ar(dur, 0.02, 0.18, curve=3.4)
        layers.append(low * p.sub_amp * 0.7)
    return finish(p, *layers)


PACKS = {name: {'notify-toast': partial(toast, profile), 'notify-chime': partial(chime, profile)}
         for name, profile in PROFILES.items()}
