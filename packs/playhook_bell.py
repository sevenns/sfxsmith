"""The PS5-lineage pack, rebuilt from measured structure rather than impression.

What the reference set actually does, and what a first pass gets wrong:

- **Brightness is inverted from intuition.** Its `move` is dark and soft (hi-mid 3.2%, top
  0.0%) while its `play` is the bright one (hi-mid 24.1%). A pack built on the assumption
  that navigation ticks are sparkly and launches are warm gets both backwards.
- **Nothing is struck.** Band peaks land 26 ms (move), 109-200 ms (button), 136-243 ms (back)
  and 220-385 ms (play) after onset. These are swells, not hits.
- **The stereo image is a per-slot decision.** `button` is near-mono (correlation 0.94),
  `play` is wide (0.53). Spreading every slot equally erases that.
- **`play` sweeps upward over time**: its dominant partial climbs 234 -> 703 -> 1266 Hz.
- **Intervals are thirds, not octaves** — its `button` sits on 215/258 Hz, a minor third.
- Energy is centred in the mids, with almost nothing above 3 kHz.

Two dials sit on top of that structure. `space` carries over from `playhook_lowend` and
lengthens reverb, reflection level and envelopes together. `tail` is narrower: it moves only
the reverb time and send, from this pack's dry values toward the longer, wetter profile of the
first Aurora — the residual echo that version was liked for — leaving the spectrum untouched.

The shipped set is `tail=0.5`: half that echo, on the dry body. It supersedes the original
Aurora, which stays available in `packs/playhook.py` as `playhook-aurora-v1`.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from sfxsmith.engine import (NOTE as N, SR, additive, at, bp, env_ar, fm_voice, glide, mix,
                             noise, pad, sine, soft_clip, sweep_filter, t_axis, voice)

PEAK_DB = {'move': -9.0, 'back': -4.5, 'button': -3.5, 'play': -1.0}

TILT = 0.6


def rt_for(base: float, space: float) -> float:
    """Scales a dry reverb time by the 0-1 space dial."""
    return base * (1 + 4.0 * space)


def wet_for(base: float, space: float) -> float:
    """Scales the reverb send by the space dial, capped so the dry signal stays in front."""
    return min(0.7, base * (1 + 3.0 * space))


def hold(base: float, space: float) -> float:
    """Stretches an envelope decay so the sound itself lingers, not only its reflections."""
    return base * (1 + 0.7 * space)


ECHO_PROFILE = {
    'move': (0.5, 0.32, 0.9, 0.30),
    'button': (0.7, 0.12, 1.4, 0.36),
    'back': (0.8, 0.14, 1.3, 0.34),
    'play': (1.1, 0.42, 2.0, 0.42),
}


def echo_rt(slot: str, tail: float, space: float) -> float:
    """Reverb time for a slot, interpolating from this pack's dry value toward the longer
    tail of the first Aurora, then scaled by the space dial. `tail` may exceed 1."""
    dry, _, wet_rt, _ = ECHO_PROFILE[slot]
    return rt_for(dry + (wet_rt - dry) * tail, space)


def echo_wet(slot: str, tail: float, space: float) -> float:
    """Reverb send for a slot, on the same interpolation as `echo_rt`."""
    dry_rt, dry, _, wet = ECHO_PROFILE[slot]
    return wet_for(dry + (wet - dry) * tail, space)


def floor_noise(dur: float, seed: int, amp: float, lo: float = 200,
                hi: float = 4200) -> np.ndarray:
    """Quiet wideband bed that gives the synthesis the spectral flatness a recording has."""
    core = bp(noise(dur, seed=seed), lo, hi) * env_ar(dur, 0.02, 0.06, curve=4) * 0.7
    top = bp(noise(dur, seed=seed + 7), hi * 0.7, 9000) * env_ar(dur, 0.03, 0.09, curve=3) * 0.16
    return (core + top) * amp


def bell_move(space: float = 0.0, tail: float = 0.0) -> np.ndarray:
    """Navigation: a dark, soft-swelling bell on E4 that peaks around 30 ms, never clicks."""
    d = 0.44 * (1 + space)
    body = fm_voice(N['E4'], 1.0, 1.1, d, idx_decay=16)
    body *= env_ar(d, 0.028, hold(0.075, space), curve=3.2)
    third = sine(N['Gs4'], d) * env_ar(d, 0.034, hold(0.06, space), curve=3.4) * 0.3
    low = sine(N['E3'], d) * env_ar(d, 0.024, hold(0.055, space), curve=3.6) * 0.28
    return voice(soft_clip(mix(body, third, low, floor_noise(d, 141, 0.05, 260, 2600)), 1.25) * 0.55,
                 width=echo_wet('move', tail, space), rt60=echo_rt('move', tail, space), seed=201, tilt=TILT)


def bell_button(space: float = 0.0, tail: float = 0.0) -> np.ndarray:
    """Confirm: a third (E4 + G#4) whose mid blooms at ~110 ms over a low that arrives at ~200 ms.

    Near-mono, as the reference's 0.94 channel correlation demands.
    """
    d = 0.95 * (1 + space)
    body = additive(N['E4'], [(1, 1.0, 1.0), (1.26, 0.72, 0.9), (2, 0.3, 0.55),
                              (3, 0.13, 0.35), (5, 0.05, 0.22)], d, hold(0.135, space),
                    detune=0.0018)
    body *= env_ar(d, 0.105, hold(0.14, space), curve=2.5)
    low = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.28, 0.5)], d, hold(0.16, space))
    low *= env_ar(d, 0.2, hold(0.17, space), curve=2.2) * 0.95
    sub = sine(N['E2'], d) * env_ar(d, 0.16, hold(0.11, space), curve=2.8) * 0.09
    return voice(soft_clip(mix(body, low, sub, floor_noise(d, 143, 0.07, 300, 3400)), 1.35) * 0.58,
                 width=echo_wet('button', tail, space), rt60=echo_rt('button', tail, space), seed=203, tilt=TILT)


def bell_back(space: float = 0.0, tail: float = 0.0) -> np.ndarray:
    """Cancel: low-weighted, peaking late (~240 ms), with a high trail two octaves up."""
    d = 1.05 * (1 + space)
    low = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.34, 0.6), (3, 0.14, 0.4),
                             (4.5, 0.05, 0.25)], d, hold(0.19, space))
    low *= env_ar(d, 0.235, hold(0.2, space), curve=2.2)
    mid = pad(sine(glide(N['B4'], N['E4'], 0.3, shape=1.5), 0.3)
              * env_ar(0.3, 0.13, hold(0.12, space), curve=2.6) * 0.42, int(d * SR))
    trail = fm_voice(N['B4'], 2.0, 0.8, d, idx_decay=7) * env_ar(d, 0.14, hold(0.1, space),
                                                                 curve=2.8) * 0.2
    sub = sine(N['E2'], d) * env_ar(d, 0.2, hold(0.15, space), curve=2.6) * 0.13
    return voice(soft_clip(mix(low, mid, trail, sub, floor_noise(d, 145, 0.05, 260, 2800)), 1.3) * 0.58,
                 width=echo_wet('back', tail, space), rt60=echo_rt('back', tail, space), seed=205, tilt=TILT)


def bell_play(space: float = 0.0, tail: float = 0.0) -> np.ndarray:
    """Launch: the bright slot. An arpeggio climbing E3 -> E5 with a shimmer that arrives late,
    reproducing the reference's 234 -> 703 -> 1266 Hz ascent, in the widest image of the set.
    """
    d = 1.75 * (1 + 0.6 * space)
    arp = np.zeros(int(d * SR))
    for i, (name, off) in enumerate([('B3', 0), ('E4', 120), ('Gs4', 240),
                                     ('B4', 360), ('E5', 470), ('Gs5', 580)]):
        seg = d - off / 1000
        v = fm_voice(N[name], 1.0, 1.5 + 0.35 * i, seg, idx_decay=7)
        v *= env_ar(seg, 0.07 + i * 0.012, hold(0.34 - i * 0.03, space), curve=2.1) * (0.88 ** i)
        arp += at(v, off, d)
    climb = np.zeros(int(d * SR))
    for i, name in enumerate(['B5', 'E6', 'Gs6']):
        seg = d - 0.24
        v = sine(N[name], seg) * env_ar(seg, 0.16 + i * 0.06, hold(0.24, space), curve=2.0)
        climb += at(v * (0.62 ** i), 240, d)
    rise = at(sweep_filter(noise(0.9, seed=147), 500, 5200)
              * (t_axis(0.9) / 0.9) ** 2.1 * 0.14, 120, d)
    low = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.34, 0.6), (0.5, 0.3, 0.8)], d, hold(0.42, space))
    low *= env_ar(d, 0.38, hold(0.4, space), curve=2.0) * 0.26
    return voice(soft_clip(mix(arp * 0.62, climb * 0.34, rise, low,
                               floor_noise(d, 149, 0.05, 340, 5000)), 1.45) * 0.6,
                 width=echo_wet('play', tail, space), rt60=echo_rt('play', tail, space), seed=207, tilt=TILT)


def bell_slots(space: float = 0.0, tail: float = 0.0) -> dict[str, object]:
    """The four slots at one setting of the space and tail dials."""
    return {
        'move': partial(bell_move, space, tail),
        'button': partial(bell_button, space, tail),
        'back': partial(bell_back, space, tail),
        'play': partial(bell_play, space, tail),
    }


PACKS = {
    'playhook-aurora': bell_slots(tail=0.5),
}
