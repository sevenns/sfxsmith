"""UI sound packs for the Playhook launcher: three characters over one E-major-9 tonality.

Each pack fills the four slots Playhook expects (move, button, back, play). The slot
peaks are staggered so navigation never shouts over confirmation.

`playhook-aurora-v1` is the superseded first take on the PS5 lineage: it was measured only
coarsely, and reads too bright (its `move` carries 23.5% hi-mid against the reference's 3.2%).
`packs/playhook_bell.py` replaced it as `playhook-aurora`; this one is kept because its longer
reverb tail is what the replacement's echo dial was tuned against.

The reference study behind these: PS5 sounds are highly tonal (spectral flatness 0.016-0.054),
soft-attacked and long-tailed; Steam Big Picture sounds are bass-heavy and percussive with a
5 ms transient. Aurora follows the first, Tactile the second, Cartridge fuses them and adds
an inharmonic shimmer of its own.
"""

from __future__ import annotations

import numpy as np

from sfxsmith.engine import (NOTE as N, SR, additive, at, bp, env_ar, env_perc, fm_voice,
                             glide, inharmonic, mix, noise, pad, sine, soft_clip,
                             sweep_filter, t_axis, voice)

PEAK_DB = {'move': -9.0, 'back': -4.5, 'button': -3.5, 'play': -1.0}

SHIMMER_RATIOS = [1.0, 2.41, 3.83, 5.17, 7.03]


def aurora_move() -> np.ndarray:
    """Aurora navigation tick: a soft FM bell with a whisper of air."""
    d = 0.5
    bell = fm_voice(N['E5'], 2.0, 2.2, d, idx_decay=26) * env_ar(d, 0.006, 0.055, curve=5)
    shine = fm_voice(N['B5'], 3.0, 1.1, d, idx_decay=34) * env_ar(d, 0.004, 0.035, curve=6) * 0.35
    air = bp(noise(d, seed=3), 2500, 6000) * env_ar(d, 0.003, 0.02, curve=8) * 0.05
    return voice(soft_clip(mix(bell, shine, air), 1.2) * 0.55, width=0.3, rt60=0.9, seed=11)


def aurora_button() -> np.ndarray:
    """Aurora confirmation: detuned harmonic body under a bell, anchored by a low sine."""
    d = 1.1
    body = additive(N['E4'], [(1, 1.0, 1.0), (2, 0.45, 0.7), (3, 0.22, 0.5),
                              (4.02, 0.14, 0.35), (6, 0.07, 0.25)], d, 0.16, detune=0.002)
    body *= env_ar(d, 0.012, 0.18, curve=2.6)
    bell = fm_voice(N['B4'], 1.5, 3.0, d, idx_decay=12) * env_ar(d, 0.02, 0.13, curve=3) * 0.4
    low = sine(N['E2'], d) * env_ar(d, 0.008, 0.09, curve=5) * 0.3
    air = bp(noise(d, seed=5), 1500, 5500) * env_ar(d, 0.03, 0.06, curve=5) * 0.035
    return voice(soft_clip(mix(body, bell, low, air), 1.3) * 0.6, width=0.36, rt60=1.4, seed=13)


def aurora_back() -> np.ndarray:
    """Aurora cancel: a falling fifth settling onto E3."""
    d = 1.1
    down = pad(sine(glide(N['B4'], N['E3'], 0.34, shape=1.6), 0.34), int(d * SR))
    down *= env_ar(d, 0.014, 0.16, curve=3)
    body = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.3, 0.6), (3, 0.12, 0.4)], d, 0.2)
    body *= env_ar(d, 0.02, 0.19, curve=2.8) * 0.7
    low = sine(N['E2'], d) * env_ar(d, 0.01, 0.11, curve=4) * 0.35
    return voice(soft_clip(mix(down, body, low), 1.2) * 0.6, width=0.34, rt60=1.3, seed=17)


def aurora_play() -> np.ndarray:
    """Aurora launch: a staggered Emaj9 arpeggio over a swelling pad and sub."""
    d = 2.4
    out = np.zeros(int(d * SR))
    for i, (name, off) in enumerate([('E3', 0), ('B3', 90), ('E4', 180),
                                     ('Gs4', 270), ('B4', 360), ('E5', 450)]):
        seg = d - off / 1000
        v = fm_voice(N[name], 2.0, 1.8, seg, idx_decay=9)
        v *= env_ar(seg, 0.03 + i * 0.006, 0.55 - i * 0.05, curve=2.2) * (0.85 ** i)
        out += at(v, off, d)
    swell = bp(noise(d, seed=9), 900, 4800) * env_ar(d, 0.5, 0.4, curve=2.2) * 0.06
    low = sine(N['E2'], d) * env_ar(d, 0.05, 0.5, curve=2.5) * 0.4
    sub = sine(N['E1'], d) * env_ar(d, 0.09, 0.4, curve=3) * 0.25
    return voice(soft_clip(mix(out * 0.5, swell, low, sub), 1.4) * 0.62,
                 width=0.42, rt60=2.0, seed=19)


def tactile_move() -> np.ndarray:
    """Tactile navigation tick: band-limited noise transient over a short tone."""
    d = 0.28
    tick = bp(noise(d, seed=21), 700, 4200) * env_perc(d, 0.0008, 0.012) * 0.32
    tone = pad(sine(glide(N['E5'], N['B4'], 0.05), 0.05) * env_perc(0.05, 0.001, 0.02),
               int(d * SR))
    body = sine(N['E4'], d) * env_perc(d, 0.001, 0.022) * 0.5
    return voice(soft_clip(mix(tick, tone, body), 1.6) * 0.5, width=0.14, rt60=0.28, seed=23)


def tactile_button() -> np.ndarray:
    """Tactile confirmation: a hard click, a pitch-dropping punch and a sub tail."""
    d = 0.55
    click = bp(noise(d, seed=25), 600, 3800) * env_perc(d, 0.0005, 0.008) * 0.34
    punch = pad(sine(glide(N['E3'], N['E2'], 0.09, shape=1.2), 0.09) * env_perc(0.09, 0.001, 0.035),
                int(d * SR))
    body = additive(N['E4'], [(1, 1.0, 1.0), (2, 0.3, 0.6), (3, 0.1, 0.4)], d, 0.07)
    body *= env_perc(d, 0.002, 0.07) * 0.7
    sub = sine(N['E2'], d) * env_perc(d, 0.002, 0.05) * 0.55
    return voice(soft_clip(mix(click * 0.55, punch, body, sub), 1.7) * 0.55,
                 width=0.16, rt60=0.4, seed=27)


def tactile_back() -> np.ndarray:
    """Tactile cancel: a thud dropping to E1 with a descending tone above it."""
    d = 0.6
    thud = pad(sine(glide(N['B2'], N['E1'], 0.16, shape=1.4), 0.16) * env_perc(0.16, 0.0015, 0.06),
               int(d * SR))
    click = bp(noise(d, seed=29), 700, 4000) * env_perc(d, 0.0005, 0.006) * 0.16
    tone = pad(sine(glide(N['E4'], N['B3'], 0.13, shape=1.8), 0.13) * env_perc(0.13, 0.002, 0.05) * 0.55,
               int(d * SR))
    return voice(soft_clip(mix(thud, click, tone), 1.8) * 0.58, width=0.18, rt60=0.45, seed=31)


def tactile_play() -> np.ndarray:
    """Tactile launch: sub boom, noise hit, filter rise and a sustained Emaj chord."""
    d = 1.5
    boom = pad(sine(glide(N['E2'], N['E1'], 0.5, shape=1.5), 0.5) * env_ar(0.5, 0.004, 0.16, curve=3),
               int(d * SR))
    hit = bp(noise(d, seed=33), 400, 3600) * env_perc(d, 0.001, 0.025) * 0.2
    rise = sweep_filter(noise(0.42, seed=35), 300, 3200) * (t_axis(0.42) / 0.42) ** 2 * 0.22
    chord = np.zeros(int(d * SR))
    for i, name in enumerate(['E3', 'B3', 'E4', 'Gs4']):
        chord += sine(N[name], d) * env_ar(d, 0.006, 0.26, curve=2.6) * (0.8 ** i)
    return voice(soft_clip(mix(boom, hit, at(rise, 0, d), chord * 0.45), 1.6) * 0.6,
                 width=0.22, rt60=0.9, seed=37)


def cartridge_move() -> np.ndarray:
    """Cartridge navigation tick: transient bite, FM tone and a short metallic shimmer."""
    d = 0.4
    bite = bp(noise(d, seed=41), 1400, 5000) * env_perc(d, 0.0006, 0.007) * 0.13
    tone = fm_voice(N['E5'], 2.0, 1.4, d, idx_decay=30) * env_ar(d, 0.0025, 0.05, curve=4.5)
    shim = inharmonic(N['B5'], d, SHIMMER_RATIOS, amp=0.16, seed=43, decay=0.035)
    return voice(soft_clip(mix(bite, tone, shim), 1.4) * 0.5, width=0.24, rt60=0.55, seed=45)


def cartridge_button() -> np.ndarray:
    """Cartridge confirmation: Steam's bite and punch fused to a PS5-style harmonic body."""
    d = 0.9
    bite = bp(noise(d, seed=47), 900, 4500) * env_perc(d, 0.0005, 0.006) * 0.16
    punch = pad(sine(glide(N['E3'], N['E2'], 0.07, shape=1.3), 0.07) * env_perc(0.07, 0.001, 0.03) * 0.8,
                int(d * SR))
    body = additive(N['E4'], [(1, 1.0, 1.0), (2, 0.4, 0.7), (3, 0.18, 0.5), (5, 0.08, 0.3)],
                    d, 0.13, detune=0.0025)
    body *= env_ar(d, 0.004, 0.15, curve=2.8)
    shim = inharmonic(N['B4'], d, SHIMMER_RATIOS, amp=0.2, seed=49, decay=0.1)
    air = bp(noise(d, seed=51), 1600, 5200) * env_ar(d, 0.012, 0.05, curve=5) * 0.04
    return voice(soft_clip(mix(bite, punch, body, shim, air), 1.5) * 0.58,
                 width=0.3, rt60=0.9, seed=53)


def cartridge_back() -> np.ndarray:
    """Cartridge cancel: falling fifth, sub thud and a dimmed shimmer."""
    d = 0.9
    bite = bp(noise(d, seed=55), 1000, 4600) * env_perc(d, 0.0005, 0.005) * 0.12
    down = pad(sine(glide(N['B4'], N['E3'], 0.2, shape=1.7), 0.2) * env_ar(0.2, 0.003, 0.09, curve=3),
               int(d * SR))
    thud = pad(sine(glide(N['E2'], N['E1'], 0.13, shape=1.3), 0.13) * env_perc(0.13, 0.0015, 0.05) * 0.7,
               int(d * SR))
    body = additive(N['E3'], [(1, 1.0, 1.0), (2, 0.26, 0.6), (3, 0.1, 0.4)], d, 0.14)
    body *= env_ar(d, 0.006, 0.13, curve=3) * 0.6
    shim = inharmonic(N['E4'], d, SHIMMER_RATIOS, amp=0.12, seed=57, decay=0.07)
    return voice(soft_clip(mix(bite, down, thud, body, shim), 1.5) * 0.58,
                 width=0.28, rt60=0.85, seed=59)


def cartridge_play() -> np.ndarray:
    """Cartridge launch: the seating transient, a filter rise and a shimmering arpeggio."""
    d = 2.2
    seat = bp(noise(d, seed=61), 700, 4200) * env_perc(d, 0.0008, 0.016) * 0.17
    boom = pad(sine(glide(N['E2'], N['E1'], 0.45, shape=1.5), 0.45) * env_ar(0.45, 0.004, 0.18, curve=2.8),
               int(d * SR))
    rise = sweep_filter(noise(0.5, seed=63), 350, 3400) * (t_axis(0.5) / 0.5) ** 2.2 * 0.2
    arp = np.zeros(int(d * SR))
    for i, (name, off) in enumerate([('E3', 0), ('B3', 70), ('E4', 140), ('Gs4', 210), ('B4', 280)]):
        seg = d - off / 1000
        v = fm_voice(N[name], 2.0, 1.6, seg, idx_decay=11)
        v *= env_ar(seg, 0.008 + i * 0.004, 0.42 - i * 0.04, curve=2.3) * (0.86 ** i)
        arp += at(v, off, d)
    shim = at(inharmonic(N['E5'], d - 0.28, SHIMMER_RATIOS, amp=0.18, seed=65, decay=0.3), 280, d)
    return voice(soft_clip(mix(seat, boom, at(rise, 0, d), arp * 0.5, shim), 1.5) * 0.6,
                 width=0.4, rt60=1.8, seed=67)


PACKS = {
    'playhook-aurora-v1': {
        'move': aurora_move, 'button': aurora_button,
        'back': aurora_back, 'play': aurora_play,
    },
    'playhook-tactile': {
        'move': tactile_move, 'button': tactile_button,
        'back': tactile_back, 'play': tactile_play,
    },
    'playhook-cartridge': {
        'move': cartridge_move, 'button': cartridge_button,
        'back': cartridge_back, 'play': cartridge_play,
    },
}
