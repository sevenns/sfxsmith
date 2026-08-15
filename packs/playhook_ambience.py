"""A seamless two-minute ambience bed for the Playhook launcher, in C# minor.

Measured from three reference beds (PS5, PS2, Steam Big Picture), which agree more than they
differ: nothing above 800 Hz matters (hi-mid 0.1-1.6%, top under 0.2%), the low end carries the
weight, the image is wide (channel correlation 0.49-0.64), and the material breathes.

**Intended use.** This bed is the companion to the `playhook-abyss` UI set and ships under the
same name, following the launcher's convention that an ambience track is named for the sound set
it belongs with. It sits under those sounds: dark, deep and unhurried,
weighted the way Steam Big Picture is rather than the brighter PS5 bed. Chimes fade in over
~0.9 s rather than striking, so nothing in it asks to be noticed.

**Key choice.** The references sit in dark flat minors (F minor, Bb minor). Playing one of those
under UI sounds pitched in E would put a semitone clash under every click. C# minor is the fix:
its scale contains C#-E-G#-B, the exact notes Aurora and Abyss are built from.

**Movement.** A static chord with LFOs on it reads as monotonous however slow the LFOs are —
the ear stops hearing modulation and starts hearing sameness. So the bed moves harmonically
instead: four chords (C#m - A - E - B) crossfade across the loop on circular windows, each with
its own bass note, register and chime (the A section answers its C#4 chime with a D4 a
a minor third above, landing as a fifth over that chord). A chime's gain compensates for where
it falls in the phrase envelope: the answering E4 sits on the decline and needs ~1.7x to read
as level with the C#4 that precedes it. The E chord is deliberate — it is the tonality the UI
sounds themselves are built in, so the bed passes through home once per loop. For the same
reason the noise floor is windowed into two gusts instead of running throughout: constant noise
is heard as hiss within seconds.

**Shape.** The loop is two phrases of two chords each, separated by silence: it fades up from
nothing, plays C#m and A, falls back to silence around the one-minute mark, then plays E and B
and fades out again. The silence in the middle is what makes the silence at the loop point
unremarkable — a bed that goes quiet twice reads as composed, where one that goes quiet only at
its boundary reads as a gap. The envelope is applied after the reverb, so the tails fall away
with it and the quiet is genuinely silent.

**Seamlessness** is structural, not patched in afterwards. Every partial and LFO is snapped to
a whole number of cycles per loop, noise is generated in the frequency domain so it is periodic
by construction, filters run circularly, and reverb is applied by circular convolution so its
tail wraps into the head.
"""

from __future__ import annotations

import numpy as np

from sfxsmith.engine import (NOTE as N, SR, cyclic, lfo, loop_noise, loop_reverb, lp,
                             sine, snap, soft_clip, warm, window)

NOTE_NAMES = list(N)

LOOP = 120.0

SECTIONS = [
    {'centre': 16.0, 'bass': ['Cs1', 'Cs2'], 'mid': ['Cs3', 'E3', 'Gs3'],
     'high': ['Gs3', 'Cs4', 'E4'], 'chimes': [('Gs3', 20.0, 1.0)]},
    {'centre': 42.0, 'bass': ['A1', 'A2'], 'mid': ['A2', 'Cs3', 'E3'],
     'high': ['E3', 'A3', 'Cs4'], 'chimes': [('Cs4', 44.0, 1.0), ('E4', 47.5, 1.7)]},
    {'centre': 78.0, 'bass': ['E1', 'E2'], 'mid': ['E3', 'Gs3', 'B3'],
     'high': ['B3', 'E4', 'Gs4'], 'chimes': [('E3', 82.0, 1.0)]},
    {'centre': 104.0, 'bass': ['B1', 'B2'], 'mid': ['B2', 'Ds3', 'Fs3'],
     'high': ['Fs3', 'B3', 'Ds4'], 'chimes': [('B3', 106.0, 1.0)]},
]

PHRASES = [(30.0, 55.5), (90.0, 55.5)]

SECTION_WIDTH = 36.0
REST_OFFSET = 10.0
BEAT_WIDTH = 10.0

NOTE_SEED = {name: i * 37 for i, name in enumerate(NOTE_NAMES)}


def drift(freq: float, dur: float, seed: int, cents: float = 4.0) -> np.ndarray:
    """A note as three slightly detuned voices, each snapped to the loop grid.

    Detune is what turns a dead sine into something that moves — and also the easiest way to
    ruin a bed. Two voices `cents` apart beat at their frequency difference, and a uniform
    detune across a chord lands every pair on a similar rate, stacking into one throb. An early
    version detuned the bass by 3 cents and produced a 0.12 Hz beat at full strength, against
    roughly 0.3 in the references. Here the detune is small and scattered per voice by seed, so
    the pairs beat at unrelated rates that never gang up.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(int(dur * SR))
    for i in range(3):
        spread = cents * rng.uniform(0.5, 1.5)
        offset = (i - 1) * spread / 1200
        f = snap(freq * (2 ** offset), dur)
        out += sine(f, dur, phase=rng.uniform(0, 6.28)) * (0.55 if i == 1 else 0.3)
    return out


def chord(notes: list[str], dur: float, seed: int, cents: float) -> np.ndarray:
    """A chord of drifting voices, quieter as it climbs."""
    out = np.zeros(int(dur * SR))
    for i, name in enumerate(notes):
        out += drift(N[name], dur, seed + NOTE_SEED[name], cents) * (0.82 ** i)
    return out / len(notes)


def unison(notes: list[str], dur: float, seed: int) -> np.ndarray:
    """The same chord with every voice in perfect unison — nothing to beat against.

    Matched in level to `chord` so the two can be crossfaded without a volume step.
    """
    out = np.zeros(int(dur * SR))
    for i, name in enumerate(notes):
        phase = np.random.default_rng(seed + NOTE_SEED[name]).uniform(0, 6.28)
        out += sine(snap(N[name], dur), dur, phase=phase) * 1.15 * (0.82 ** i)
    return out / len(notes)


def unrest(dur: float) -> np.ndarray:
    """How much beating is allowed at each moment: present around each chord's centre, gone
    across the changes.

    Detune beating is the bed's only real texture, and also the thing that wears the listener
    down when it never stops. Rather than weakening it everywhere, it is switched off entirely
    for a stretch before each chord change — the voices converge to unison, the pulsing stops,
    and it returns with the next chord.

    The rest has to be budgeted against the reverb, not just against the dry signal: with an
    7 s tail, beating that stops at t is still audible at t+6. So the beating window is both
    narrow (10 s of each 30 s section) and placed early in it, leaving 20 s of dry silence in
    the modulation — around twelve of which survive the reverb tail as true rest. The reverb
    was shortened from 8.5 s for the same reason: past a point, a longer tail simply smears
    the beating across the gap that was meant to be quiet.

    Note that detune is not the only thing that beats. Overlapping chords beat against each
    other too, and with a 62 s window on a 30 s spacing three chords sounded at once and
    produced more pulsing at the changes (0.16) than the detune ever did at the centres (0.02).
    A subtler version of the same trap: a note shared by two chords (C#m and A both contain
    C#3 and E3) used to get a DIFFERENT random detune in each section, so where the sections
    overlapped the two versions of one note beat against each other. Detune is now keyed to the
    note itself, so a shared note is literally the same signal in both chords and they simply
    reinforce.
    """
    env = np.zeros(int(dur * SR))
    for sec in SECTIONS:
        env += window(dur, sec['centre'] - REST_OFFSET, BEAT_WIDTH)
    return np.clip(env, 0.0, 1.0)


def morph(x: np.ndarray, dur: float, cycles: int, dark: float, bright: float,
          phase: float = 0.0) -> np.ndarray:
    """Slowly crossfades between a dark and a bright filtering of the same signal, giving the
    wandering centroid the references have. Filters run circularly to protect the loop point."""
    m = 0.5 + 0.5 * lfo(cycles, dur, phase)
    return (cyclic(lambda y: lp(y, dark), x) * (1 - m)
            + cyclic(lambda y: lp(y, bright), x) * m)


def chime(note: str, at_s: float, dur: float, seed: int) -> np.ndarray:
    """One distant bell that fades in over ~0.9 s instead of striking.

    Written into the buffer circularly, so a chime placed near the end continues into the
    start of the next pass rather than being cut off at the boundary.
    """
    n = int(dur * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    span = int(11 * SR)
    t = np.arange(span) / SR
    env = (1 - np.exp(-t / 0.9)) * np.exp(-t / 3.4)
    f = snap(N[note], dur)
    tone = (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
            + 0.28 * np.sin(2 * np.pi * f * 2 * t)
            + 0.1 * np.sin(2 * np.pi * f * 3.01 * t))
    np.add.at(out, (np.arange(span) + int(at_s * SR)) % n, tone * env)
    return out


def breath(dur: float, seed: int) -> np.ndarray:
    """Noise that arrives in two slow gusts rather than sitting there throughout."""
    bed = loop_noise(dur, seed=seed, lo=60, hi=1700)
    gusts = window(dur, 26.0, 34.0) * 0.9 + window(dur, 84.0, 40.0)
    return bed * gusts


def voices(dur: float, seed: int, phase_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Builds the bed as (mono foundation, wide upper half) across all four sections.

    The bass is shared by both channels on purpose: decorrelating low frequencies measured a
    channel correlation of 0.19 against the references' 0.49-0.64, and reads as a hollow mix.
    """
    n = int(dur * SR)
    base = np.zeros(n)
    wide = np.zeros(n)
    beat = unrest(dur)
    for i, sec in enumerate(SECTIONS):
        env = window(dur, sec['centre'], SECTION_WIDTH)
        low = (drift(N[sec['bass'][0]], dur, 7001 + NOTE_SEED[sec['bass'][0]], cents=0.5) * beat
               + sine(snap(N[sec['bass'][0]], dur), dur) * 1.15 * (1 - beat)) * 0.2
        low = low + chord(sec['bass'][1:], dur, 7040, cents=0.8) * 0.34 * beat \
            + unison(sec['bass'][1:], dur, 7040) * 0.34 * (1 - beat)
        base += low * env

        mid = (chord(sec['mid'], dur, seed + 80, cents=1.6) * beat
               + unison(sec['mid'], dur, seed + 80) * (1 - beat)) * 1.45
        mid = morph(mid, dur, 2, dark=1100, bright=3600, phase=phase_shift + i * 0.12)
        high = (chord(sec['high'], dur, seed + 120, cents=2.2) * beat
                + unison(sec['high'], dur, seed + 120) * (1 - beat)) * 0.72
        high = morph(high, dur, 1, dark=1400, bright=4400, phase=0.5 + phase_shift + i * 0.09)
        wide += (mid + high) * env
    return base, wide


def abyss() -> np.ndarray:
    """The shipped bed: two minutes of C# minor moving through four chords, looping seamlessly.

    The reverb is split deliberately. One pass is shared by both channels and carries the
    foundation and chimes; a second, per-channel pass carries only the upper half. Running a
    single decorrelated reverb over everything collapsed channel correlation to -0.02 on an
    early attempt: the wet signal was half the mix, and none of it agreed between the sides.
    """
    dur = LOOP
    base, wide_l = voices(dur, 1001, 0.0)
    _, wide_r = voices(dur, 2002, 0.37)

    bells = np.zeros(int(dur * SR))
    for i, sec in enumerate(SECTIONS):
        for j, (note, at_s, gain) in enumerate(sec['chimes']):
            bells += chime(note, at_s, dur, 3003 + i * 31 + j * 7) * (0.55 + 0.12 * (i % 2)) * gain

    air = breath(dur, 4004)
    common = loop_reverb(base + bells * 0.2, wet=0.5, rt60=7.0, seed=5005, hf_damp=3200)

    shape = np.zeros(int(dur * SR))
    for centre, width in PHRASES:
        shape += window(dur, centre, width)
    shape = np.clip(shape, 0.0, 1.0) ** 0.7

    out = []
    for i, side in enumerate((wide_l, wide_r)):
        wet = loop_reverb(side, wet=0.56, rt60=7.5, seed=6006 + i * 101, hf_damp=3200)
        mixed = (common + wet * 0.95 + air * 0.06) * shape
        mixed /= np.max(np.abs(mixed)) + 1e-9
        out.append(cyclic(lambda y: warm(y, cut=4200, amount=0.32), soft_clip(mixed * 0.85, 1.1)))

    stereo = np.stack(out, axis=1)
    return stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.82


TRACKS = {
    'playhook-abyss': abyss,
}
