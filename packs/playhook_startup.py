"""Five-second startup stings for the Playhook launcher, built from the playhook-abyss material.

**What it inherits.** Two things already carry the Abyss identity and this has to sit between
them: the UI set (`playhook_lowend`, Deck at space 1.0) — everything below 900 Hz, near-mono,
long reflections — and the ambience bed (`playhook_ambience`) — C# minor, chords that fade in
instead of striking, chimes with a ~0.9 s rise. A startup sound is heard once, immediately
before both, so it must not be brighter or harder than either or it will read as a different
product announcing itself.

**Harmony.** The bed passes through C#m - A - E - B; the UI sounds are all in E. The sting
therefore starts in C# minor and arrives on E, which is the bed's relative major and the UI
set's home in one move: the launcher opens in the dark and settles where every click will
land. `beacon` states the same idea melodically, over the bed's own chime notes (G#3, C#4, E4).

**Shape.** Each variant is exactly `LENGTH` seconds including its tail: the reverberated result
is cut to length and eased to silence by `settle`, so nothing depends on where a decay happens
to fall under the trim threshold. Four readings of the same material, differing in how they
begin:

- `surface` rises out of the sub with no transient at all;
- `beacon` leads with three ascending bells and brings the weight in under the last;
- `breach` opens with the launch sound's own onset, then lets the pad bloom behind it;
- `tide` crescendos on noise before anything is pitched.

**Shipped:** `surface`, encoded to MP3 and installed as Playhook's `assets/playhook-startup.mp3`
(the build copies it to `dist/startup.mp3`). Unlike the UI sets it is app-wide, not per-set, so
it lives beside the wallpaper rather than in `audio/ui/` — hence the rename on the way in. The
other three are kept as the points this one was chosen from.

```bash
sfxsmith render playhook_startup -o out --mp3
cp out/playhook-startup/surface.mp3 ../playhook/assets/playhook-startup.mp3
```
"""

from __future__ import annotations

from functools import partial

import numpy as np

from sfxsmith.engine import (NOTE as N, SR, additive, at, bp, env_ahd, env_ar, glide, mix,
                             noise, sine, soft_clip, t_axis, voice)

LENGTH = 5.2
OPEN = 2.0
TAIL = 1.4
TILT = 0.6
RT60 = 2.2
WIDTH = 0.3

DARK_CHORD = ['Cs3', 'E3', 'Gs3']
HOME_CHORD = ['E3', 'Gs3', 'B3']


def bloom(names: list[str], dur: float, attack: float, decay: float, seed: int,
          cents: float = 3.5, hold: float = 0.0) -> np.ndarray:
    """A chord that fades in rather than striking, each note as three scattered-detune voices.

    The detune amounts are scattered per voice for the same reason as in the ambience bed: a
    uniform spread puts every pair of voices on a similar beat rate and they stack into one
    throb. Over five seconds that would be heard as a wobble on the launcher's first sound.
    """
    n = int(dur * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    for i, name in enumerate(names):
        f0 = N[name]
        stack = np.zeros(n)
        for k in range(3):
            offset = (k - 1) * cents * rng.uniform(0.5, 1.5) / 1200
            stack += sine(f0 * 2 ** offset, dur, phase=rng.uniform(0, 6.28)) * (0.5 if k == 1 else 0.28)
        stack += sine(f0 * 2, dur, phase=rng.uniform(0, 6.28)) * 0.09
        stack += sine(f0 * 3, dur, phase=rng.uniform(0, 6.28)) * 0.035
        out += stack * (0.8 ** i)
    return out / len(names) * env_ahd(dur, attack, hold, decay, curve=1.8)


def bell(note: str, dur: float, attack: float, decay: float, seed: int) -> np.ndarray:
    """One distant bell that fades in over `attack` seconds instead of being struck.

    The same construction the ambience uses for its chimes, with a shorter rise: nothing in
    this family has a hard onset, and a struck bell here would out-attack every UI sound.
    """
    t = t_axis(dur)
    env = (1 - np.exp(-t / attack)) * np.exp(-t / decay)
    rng = np.random.default_rng(seed)
    f = N[note]
    tone = (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
            + 0.26 * np.sin(2 * np.pi * f * 2 * t)
            + 0.09 * np.sin(2 * np.pi * f * 3.01 * t))
    return tone * env


def depth(note: str, dur: float, attack: float, decay: float, amp: float = 1.0,
          hold: float = 0.0) -> np.ndarray:
    """The sub anchor: a near-pure sine with a touch of its octave for definition on small
    speakers, where the fundamental itself may not reproduce at all."""
    body = sine(N[note], dur) + 0.3 * sine(N[note] * 2, dur)
    return body * env_ahd(dur, attack, hold, decay, curve=2.0) * amp


def air(dur: float, seed: int, attack: float, decay: float, amp: float,
        lo: float = 90, hi: float = 1400) -> np.ndarray:
    """A band-limited noise swell, kept under the pitched material rather than over it."""
    return bp(noise(dur, seed=seed), lo, hi) * env_ar(dur, attack, decay, curve=1.6) * amp


def settle(x: np.ndarray, dur: float = LENGTH, tail: float = TAIL) -> np.ndarray:
    """Cuts the reverberated result to `dur` and eases its last `tail` seconds to silence.

    Convolution reverb runs past the end of its input, so without this the length of the file
    would be decided by where the tail happens to cross the trim threshold — which moves
    whenever the mix changes. Fixing the length here makes it a stated property instead.
    """
    n = int(dur * SR)
    y = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
    ramp = np.ones(n)
    k = int(tail * SR)
    ramp[n - k:] = np.linspace(1, 0, k) ** 1.7
    return y * ramp[:, None]


def finish(*layers: np.ndarray, drive: float = 1.35, level: float = 0.62) -> np.ndarray:
    """Shared output stage: saturate, spread, tilt the top down, then cut to length."""
    return settle(voice(soft_clip(mix(*layers), drive) * level,
                        width=WIDTH, rt60=RT60, seed=201, tilt=TILT))


def surface(open_s: float = OPEN) -> np.ndarray:
    """Rises out of the sub with no transient at all: C# minor opens underwater for `open_s`
    seconds, then resolves onto E as the bell arrives. The reading with the least attack.

    The opening is held, not merely started early. Its first version had the C# minor layers on
    plain attack-decay envelopes: the chord peaked at 1.1 s and had lost two thirds of its level
    by 2.0 s, so although E did not enter until 1.9 s the underwater part was heard as barely a
    second. The hold below keeps those layers at full until `open_s`, and only then lets them
    fall away underneath the E that replaces them. The two overlap into C#m7 for the length of
    the crossfade, which is why the change reads as a resolution rather than a cut.
    """
    d = LENGTH
    dark_dur = open_s + 2.0
    sub = depth('E1', d, 0.85, 2.4, amp=0.95, hold=open_s - 0.85)
    low = at(bloom(['Cs2', 'Gs2'], dark_dur, 0.9, 1.5, seed=11, hold=open_s - 0.9), 100, d) * 0.42
    dark = at(bloom(DARK_CHORD, dark_dur, 1.0, 1.4, seed=21, hold=open_s - 1.0), 150, d) * 0.75
    home = at(bloom(HOME_CHORD, 3.4, 1.15, 1.7, seed=31), open_s * 1000, d) * 0.82
    ring = at(bell('B4', 3.0, 0.45, 1.9, seed=41), open_s * 1000 + 300, d) * 0.28
    breath = at(air(open_s + 0.9, 51, 0.75, 1.2, amp=0.07), 200, d)
    return finish(sub, low, dark, home, ring, breath)


def beacon() -> np.ndarray:
    """Three ascending bells on the bed's own chime notes, with the weight arriving under the
    last one. The melodic reading: G#3 - C#4 - E4 states C# minor, the pad answers in E."""
    d = LENGTH
    first = at(bell('Gs3', 4.4, 0.3, 1.9, seed=61), 0, d) * 0.42
    second = at(bell('Cs4', 3.9, 0.28, 1.8, seed=63), 620, d) * 0.38
    third = at(bell('E4', 3.4, 0.26, 2.1, seed=65), 1240, d) * 0.4
    sub = at(depth('E1', 4.0, 0.5, 2.0, amp=0.9), 1150, d)
    dark = at(bloom(['Cs3', 'Gs3'], 3.0, 0.7, 1.3, seed=67), 300, d) * 0.4
    home = at(bloom(HOME_CHORD, 3.4, 0.85, 1.8, seed=69), 1300, d) * 0.72
    breath = at(air(2.4, 71, 0.6, 1.0, amp=0.06), 900, d)
    return finish(first, second, third, sub, dark, home, breath)


def breach() -> np.ndarray:
    """Opens with the launch sound's own onset — an E1 with a 20 ms attack and a harmonic shell
    over it — and lets the pad bloom behind the impact. The one that starts, rather than
    arrives."""
    d = LENGTH
    hit = additive(N['E1'], [(1, 1.0, 1.0), (2, 0.42, 0.6), (3, 0.16, 0.35),
                             (4.5, 0.06, 0.22)], d, 1.5)
    hit *= env_ar(d, 0.02, 1.0, curve=1.9)
    shell = additive(N['E3'], [(1, 1.0, 1.0), (1.5, 0.5, 0.8), (2, 0.3, 0.5),
                               (3, 0.12, 0.3)], d, 0.55)
    shell *= env_ar(d, 0.05, 0.5, curve=2.2) * 0.42
    thud = air(0.5, 73, 0.004, 0.09, amp=0.16, lo=120, hi=1600)
    dark = at(bloom(DARK_CHORD, 3.6, 0.55, 1.5, seed=75), 220, d) * 0.6
    home = at(bloom(HOME_CHORD, 3.0, 0.9, 1.7, seed=77), 1800, d) * 0.78
    ring = at(bell('Gs4', 2.8, 0.4, 1.7, seed=79), 2400, d) * 0.22
    return finish(hit, shell, thud, dark, home, ring, drive=1.5)


def tide() -> np.ndarray:
    """Crescendos on noise before anything is pitched, with the sub sliding up an octave
    underneath: the launcher's environment powering up, then the chord landing on it."""
    d = LENGTH
    rise = air(2.0, 81, 1.05, 0.55, amp=0.17, lo=70, hi=1100)
    slide = np.pad(sine(glide(N['E1'] / 2, N['E1'], 1.3, shape=1.4), 1.3)
                   * env_ar(1.3, 0.7, 0.7, curve=1.4) * 0.7, (0, int(d * SR) - int(1.3 * SR)))
    sub = at(depth('E1', 3.8, 0.25, 2.2, amp=0.95), 1250, d)
    dark = at(bloom(['Cs3', 'Gs3'], 2.4, 0.6, 0.9, seed=83), 400, d) * 0.34
    home = at(bloom(HOME_CHORD, 3.6, 0.6, 1.9, seed=85), 1300, d) * 0.8
    ring = at(bell('E4', 3.2, 0.35, 2.0, seed=87), 1450, d) * 0.3
    return finish(rise, slide, sub, dark, home, ring)


PACKS = {
    'playhook-startup': {
        'surface': surface,
        'surface-early': partial(surface, 1.5),
        'surface-late': partial(surface, 2.6),
        'beacon': beacon,
        'breach': breach,
        'tide': tide,
    },
}
