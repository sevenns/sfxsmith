"""Three UI packs in the Steam Big Picture lineage, rebuilt in E from measured structure.

The reference set's defining trait is what it leaves out: the 3-12 kHz band carries 0.0-0.1%
of its energy and 900 Hz-3 kHz carries under 1.2%. Everything lives below 900 Hz. Its `button`
transient is not a noise click but an instant onset of low-mid energy; its `play` is 96%
sub-bass on a single 43 Hz partial; its `back` walks downward (86 -> 129 -> 43 Hz); its `move`
peaks 114 ms in, a soft blip rather than a tick. Stereo is near-mono (correlation 0.73-0.97).

All packs here keep that spectral shape and narrow image, and differ in weight and pace:
Deck is the balanced reading, Vault the heavy one, Ember the soft one.

`playhook-abyss` is the shipped set: Deck with the space dial at 1.0, which stretches its
reverb to 1.5-3.0 s, lifts the reflection level to 40-64% and lengthens the envelopes to match.
Dry Deck is kept alongside it as the reference point the dial turns from.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from sfxsmith.engine import (NOTE as N, SR, additive, bp, env_ar, env_perc, glide, mix,
                             noise, pad, sine, soft_clip, voice)

PEAK_DB = {'move': -9.0, 'back': -4.5, 'button': -3.5, 'play': -1.0}

NARROW = 0.1
DARK = 0.62


def tilt_for(bright: float) -> float:
    """Maps a 0-1 brightness dial to the HF tilt amount: 0 keeps the reference's dark
    reading, 1 lets the upper partials through nearly untouched."""
    return DARK - 0.5 * bright


def rt_for(base: float, space: float) -> float:
    """Scales a dry reverb time by the 0-1 space dial."""
    return base * (1 + 4.0 * space)


def wet_for(base: float, space: float) -> float:
    """Scales the reverb send by the space dial, capped so the dry signal stays in front."""
    return min(0.7, base * (1 + 3.0 * space))


def hold(base: float, space: float) -> float:
    """Stretches an envelope decay so the sound itself lingers, not only its reflections."""
    return base * (1 + 0.7 * space)


def grit(dur: float, seed: int, amp: float, lo: float = 140, hi: float = 1900) -> np.ndarray:
    """Noise bed in two parts: an audible core in the low-mids, plus a very quiet wideband
    floor above it.

    The floor is what a recorded sound has and a pure synthesis does not. Spectral flatness is
    a geometric mean, so the near-zero bins between clean sine partials drag it to ~0.005 —
    an order of magnitude below the reference set. Filling those bins at -10 dB relative to the
    core restores the reference's 0.03-0.05 without adding audible hiss, and keeps the 3-12 kHz
    band under the 0.1% of total energy the reference holds there.
    """
    core = bp(noise(dur, seed=seed), lo, hi) * env_perc(dur, 0.004, 0.05)
    floor = bp(noise(dur, seed=seed + 7), hi * 0.6, 7000) * env_perc(dur, 0.006, 0.09) * 0.3
    return (core + floor) * amp


def deck_move(bright: float = 0.0, space: float = 0.0) -> np.ndarray:
    """Deck navigation: an octave pair with a soft onset, peaking around 40 ms."""
    d = 0.42 * (1 + space)
    up = 1 + 1.6 * bright
    top = additive(N['E4'], [(1, 1.0, 1.0), (2, 0.3 * up, 0.45), (3, 0.14 * up, 0.3),
                             (4, 0.05 * up, 0.22), (6, 0.02 * up, 0.16)],
                   d, hold(0.075, space))
    top *= env_ar(d, 0.035 - 0.02 * bright, hold(0.075, space), curve=3.4)
    low = sine(N['E3'], d) * env_ar(d, 0.03, hold(0.065, space), curve=3.6) * 0.85
    return voice(soft_clip(mix(top, low, grit(d, 71, 0.1 * (1 + bright))), 1.3) * 0.55,
                 width=wet_for(NARROW, space), rt60=rt_for(0.3, space), seed=101,
                 tilt=tilt_for(bright))


def deck_button(bright: float = 0.0, space: float = 0.0) -> np.ndarray:
    """Deck confirm: instant onset on E4 over its own octave, the reference's shape."""
    d = 0.6 * (1 + space)
    up = 1 + 1.6 * bright
    body = additive(N['E4'], [(1, 1.0, 1.0), (0.5, 0.9, 1.2), (2, 0.34 * up, 0.5),
                              (3, 0.16 * up, 0.35), (4, 0.07 * up, 0.25),
                              (6, 0.03 * up, 0.18)], d, hold(0.115, space))
    body *= env_ar(d, 0.002, hold(0.12, space), curve=3.0)
    sub = sine(N['E2'], d) * env_ar(d, 0.003, hold(0.07, space), curve=4.0) * 0.5
    return voice(soft_clip(mix(body, sub, grit(d, 73, 0.14 * (1 + bright))), 1.45) * 0.58,
                 width=wet_for(NARROW, space), rt60=rt_for(0.35, space), seed=103,
                 tilt=tilt_for(bright))


def deck_back(bright: float = 0.0, space: float = 0.0) -> np.ndarray:
    """Deck cancel: the downward walk E3 -> B2 -> E2, weighted to the sub band."""
    d = 0.65 * (1 + space)
    up = 1 + 1.6 * bright
    walk = pad(sine(glide(N['E3'], N['E2'], 0.26, shape=1.5), 0.26)
               * env_ar(0.26, 0.012, hold(0.1, space), curve=2.8), int(d * SR))
    mid = additive(N['B2'], [(1, 1.0, 1.0), (2, 0.4 * up, 0.5), (4, 0.16 * up, 0.3),
                             (6, 0.07 * up, 0.2), (9, 0.03 * up, 0.14)],
                   d, hold(0.09, space))
    mid *= env_ar(d, 0.02, hold(0.09, space), curve=3.2) * 0.8
    sub = sine(N['E1'], d) * env_ar(d, 0.015, hold(0.13, space), curve=3.0) * 0.9
    return voice(soft_clip(mix(walk, mid, sub, grit(d, 75, 0.11 * (1 + bright))), 1.5) * 0.58,
                 width=wet_for(NARROW, space), rt60=rt_for(0.4, space), seed=105,
                 tilt=tilt_for(bright))


def deck_play(bright: float = 0.0, space: float = 0.0) -> np.ndarray:
    """Deck launch: a sub-dominant E1 boom with a harmonic shell above it."""
    d = 1.15 * (1 + 0.6 * space)
    up = 1 + 1.6 * bright
    sub = sine(N['E1'], d) * env_ar(d, 0.03, hold(0.3, space), curve=2.4)
    octave = sine(N['E2'], d) * env_ar(d, 0.035, hold(0.24, space), curve=2.6) * 0.42
    shell = additive(N['E3'], [(1, 1.0, 1.0), (1.5, 0.5, 0.8), (2, 0.35 * up, 0.5),
                               (3, 0.15 * up, 0.3), (4.5, 0.07 * up, 0.22),
                               (6, 0.03 * up, 0.16)], d, hold(0.16, space))
    shell *= env_ar(d, 0.05, hold(0.2, space), curve=2.6) * (0.5 + 0.35 * bright)
    return voice(soft_clip(mix(sub, octave, shell, grit(d, 77, 0.085 * (1 + bright))), 1.6) * 0.6,
                 width=wet_for(0.16, space), rt60=rt_for(0.6, space), seed=107,
                 tilt=tilt_for(bright))


def vault_move() -> np.ndarray:
    """Vault navigation: a short, damped low knock with a narrow resonance."""
    d = 0.3
    knock = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.3, 0.4), (3.4, 0.14, 0.25)], d, 0.028)
    knock *= env_perc(d, 0.0015, 0.028)
    res = bp(noise(d, seed=81), 340, 1500) * env_perc(d, 0.002, 0.03) * 0.55
    sub = sine(N['E2'], d) * env_perc(d, 0.002, 0.035) * 0.7
    return voice(soft_clip(mix(knock, res, sub), 1.6) * 0.52,
                 width=0.08, rt60=0.25, seed=111, tilt=DARK)


def vault_button() -> np.ndarray:
    """Vault confirm: a bolt dropping home — B2 glides to E1 under a struck body."""
    d = 0.7
    drop = pad(sine(glide(N['B2'], N['E1'], 0.13, shape=1.25), 0.13)
               * env_perc(0.13, 0.0015, 0.055), int(d * SR))
    body = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.42, 0.45), (3, 0.2, 0.3),
                              (5, 0.08, 0.2)], d, 0.09)
    body *= env_ar(d, 0.0025, 0.085, curve=3.4) * 0.8
    sub = sine(N['E1'], d) * env_ar(d, 0.004, 0.13, curve=2.8) * 0.85
    return voice(soft_clip(mix(drop, body, sub, grit(d, 83, 0.16, 160, 1800)), 1.7) * 0.58,
                 width=0.08, rt60=0.4, seed=113, tilt=DARK)


def vault_back() -> np.ndarray:
    """Vault cancel: a long fall from E3 to E1 with the sub carrying most of the weight."""
    d = 0.85
    fall = pad(additive(N['E3'], [(1, 1.0, 1.0), (2, 0.35, 0.5)], 0.34, 0.14)
               * env_ar(0.34, 0.008, 0.14, curve=2.4), int(d * SR))
    glided = pad(sine(glide(N['E3'], N['E1'], 0.34, shape=1.8), 0.34)
                 * env_ar(0.34, 0.008, 0.14, curve=2.4), int(d * SR))
    sub = sine(N['E1'], d) * env_ar(d, 0.02, 0.19, curve=2.5)
    return voice(soft_clip(mix(fall * 0.45, glided, sub, grit(d, 85, 0.1, 150, 1600)), 1.6) * 0.58,
                 width=0.1, rt60=0.5, seed=115, tilt=DARK)


def vault_play() -> np.ndarray:
    """Vault launch: a slow-blooming sub with a fifth swelling underneath it."""
    d = 1.6
    sub = sine(N['E1'], d) * env_ar(d, 0.09, 0.44, curve=2.1)
    fifth = additive(N['B2'], [(1, 1.0, 1.0), (2, 0.35, 0.5), (3, 0.15, 0.3)], d, 0.3)
    fifth *= env_ar(d, 0.14, 0.3, curve=2.3) * 0.5
    swell = pad(sine(glide(N['E2'], N['E1'], 0.7, shape=1.3), 0.7)
                * env_ar(0.7, 0.12, 0.32, curve=2.2) * 0.55, int(d * SR))
    return voice(soft_clip(mix(sub, fifth, swell, grit(d, 87, 0.08, 140, 1500)), 1.7) * 0.6,
                 width=0.14, rt60=0.75, seed=117, tilt=DARK)


def ember_move() -> np.ndarray:
    """Ember navigation: the reference's slow blip — a fifth reaching its peak 110 ms in."""
    d = 0.55
    top = additive(N['B3'], [(1, 1.0, 1.0), (2, 0.26, 0.5), (3, 0.1, 0.3)], d, 0.085)
    top *= env_ar(d, 0.105, 0.085, curve=2.8)
    root = sine(N['E3'], d) * env_ar(d, 0.09, 0.08, curve=3.0) * 0.8
    return voice(soft_clip(mix(top, root, grit(d, 91, 0.085)), 1.2) * 0.55,
                 width=0.12, rt60=0.35, seed=121, tilt=DARK)


def ember_button() -> np.ndarray:
    """Ember confirm: a warm E-major triad, struck gently rather than hit."""
    d = 0.75
    triad = np.zeros(int(d * SR))
    for i, name in enumerate(['E3', 'Gs3', 'B3']):
        partials = [(1, 1.0, 1.0), (2, 0.28, 0.45), (3, 0.12, 0.3)]
        triad += (additive(N[name], partials, d, 0.13)
                  * env_ar(d, 0.022 + i * 0.006, 0.13, curve=2.8) * (0.82 ** i))
    sub = sine(N['E2'], d) * env_ar(d, 0.018, 0.1, curve=3.0) * 0.6
    return voice(soft_clip(mix(triad * 0.7, sub, grit(d, 93, 0.11)), 1.3) * 0.58,
                 width=0.14, rt60=0.45, seed=123, tilt=DARK)


def ember_back() -> np.ndarray:
    """Ember cancel: a gentle descent from B3 to E3, no impact at all."""
    d = 0.7
    down = pad(sine(glide(N['B3'], N['E3'], 0.3, shape=1.4), 0.3)
               * env_ar(0.3, 0.03, 0.12, curve=2.6), int(d * SR))
    warmth = additive(N['E2'], [(1, 1.0, 1.0), (2, 0.4, 0.5), (4, 0.18, 0.3),
                                (6, 0.08, 0.2)], d, 0.11)
    warmth *= env_ar(d, 0.035, 0.11, curve=2.9) * 0.65
    return voice(soft_clip(mix(down, warmth, grit(d, 95, 0.095)), 1.25) * 0.58,
                 width=0.14, rt60=0.45, seed=125, tilt=DARK)


def ember_play() -> np.ndarray:
    """Ember launch: an Emaj9 that blooms open instead of hitting, over a soft sub."""
    d = 1.5
    chord = np.zeros(int(d * SR))
    for i, name in enumerate(['E2', 'B2', 'E3', 'Gs3', 'B3']):
        chord += (additive(N[name], [(1, 1.0, 1.0), (2, 0.38, 0.5), (3, 0.18, 0.3)], d, 0.33 - i * 0.02)
                  * env_ar(d, 0.07 + i * 0.02, 0.33 - i * 0.02, curve=2.2) * (0.84 ** i))
    sub = sine(N['E1'], d) * env_ar(d, 0.06, 0.36, curve=2.3) * 0.9
    return voice(soft_clip(mix(chord * 0.72, sub, grit(d, 97, 0.08, 150, 1700)), 1.4) * 0.6,
                 width=0.18, rt60=0.7, seed=127, tilt=DARK)


def deck_slots(bright: float = 0.0, space: float = 0.0) -> dict[str, object]:
    """The four Deck slots at one setting of the brightness and space dials."""
    return {
        'move': partial(deck_move, bright, space),
        'button': partial(deck_button, bright, space),
        'back': partial(deck_back, bright, space),
        'play': partial(deck_play, bright, space),
    }


PACKS = {
    'playhook-abyss': deck_slots(space=1.0),
    'playhook-deck': deck_slots(),
    'playhook-vault': {
        'move': vault_move, 'button': vault_button, 'back': vault_back, 'play': vault_play,
    },
    'playhook-ember': {
        'move': ember_move, 'button': ember_button, 'back': ember_back, 'play': ember_play,
    },
}
