"""DSP primitives for procedural sound-effect synthesis.

Every function is pure and deterministic: noise sources take an explicit seed, so a
given pack always renders byte-identical output.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000

NOTE: dict[str, float] = {
    'E1': 41.20, 'E2': 82.41, 'B2': 123.47, 'E3': 164.81, 'Gs3': 207.65,
    'A3': 220.00, 'B3': 246.94, 'Cs4': 277.18, 'E4': 329.63, 'Fs4': 369.99,
    'Gs4': 415.30, 'B4': 493.88, 'Cs5': 554.37, 'E5': 659.25, 'Fs5': 739.99,
    'Gs5': 830.61, 'B5': 987.77, 'E6': 1318.51, 'Gs6': 1661.22, 'B6': 1975.53,
}


def t_axis(dur: float, sr: int = SR) -> np.ndarray:
    """Time axis in seconds for a signal of `dur` seconds."""
    return np.arange(int(dur * sr)) / sr


def env_ar(dur: float, attack: float, decay: float, curve: float = 3.0, sr: int = SR) -> np.ndarray:
    """Attack-release envelope. Higher `curve` makes the exponential tail snappier."""
    t = t_axis(dur, sr)
    a = np.clip(t / max(attack, 1e-5), 0, 1) ** 0.7
    rel = np.exp(-curve * np.clip((t - attack) / max(decay, 1e-5), 0, None))
    return a * rel


def env_perc(dur: float, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    """Percussive envelope: `env_ar` with a fixed steep curve."""
    return env_ar(dur, attack, decay, curve=4.5, sr=sr)


def sine(freq: float | np.ndarray, dur: float, phase: float = 0.0, sr: int = SR) -> np.ndarray:
    """Sine oscillator. `freq` may be a scalar or a per-sample frequency array."""
    t = t_axis(dur, sr)
    if np.isscalar(freq):
        return np.sin(2 * np.pi * freq * t + phase)
    return np.sin(2 * np.pi * np.cumsum(freq) / sr + phase)


def glide(f0: float, f1: float, dur: float, shape: float = 2.0, sr: int = SR) -> np.ndarray:
    """Exponential pitch glide from f0 to f1 as a per-sample frequency array."""
    k = np.linspace(0, 1, int(dur * sr)) ** shape
    return f0 * (f1 / f0) ** k


def fm_voice(carrier: float, ratio: float, index: float, dur: float,
             idx_decay: float = 8.0, sr: int = SR) -> np.ndarray:
    """Two-operator FM voice. The falling modulation index gives a bell-like attack."""
    t = t_axis(dur, sr)
    idx = index * np.exp(-idx_decay * t)
    mod = np.sin(2 * np.pi * carrier * ratio * t) * idx
    return np.sin(2 * np.pi * carrier * t + mod)


def additive(f0: float, partials: list[tuple[float, float, float]], dur: float,
             decay: float, detune: float = 0.0, sr: int = SR) -> np.ndarray:
    """Sum of partials given as (frequency_ratio, amplitude, decay_multiplier) triples."""
    out = np.zeros(int(dur * sr))
    t = t_axis(dur, sr)
    for i, (ratio, amp, dm) in enumerate(partials):
        f = f0 * ratio * (1.0 + detune * (0.5 - (i * 0.37) % 1.0))
        out += amp * np.sin(2 * np.pi * f * t + i * 1.7) * np.exp(-t / max(decay * dm, 1e-4))
    return out


def inharmonic(f0: float, dur: float, ratios: list[float], amp: float = 1.0,
               seed: int = 1, decay: float = 0.09, sr: int = SR) -> np.ndarray:
    """Metallic shimmer: partials at non-integer ratios, each with its own decay."""
    out = np.zeros(int(dur * sr))
    t = t_axis(dur, sr)
    rng = np.random.default_rng(seed)
    for i, r in enumerate(ratios):
        f = f0 * r * (1 + rng.uniform(-0.004, 0.004))
        out += (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
                * (0.58 ** i) * np.exp(-t / (decay * (0.85 ** i))))
    return out * amp


def noise(dur: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Gaussian white noise from a seeded generator."""
    return np.random.default_rng(seed).standard_normal(int(dur * sr))


def bp(x: np.ndarray, lo: float, hi: float, order: int = 2, sr: int = SR) -> np.ndarray:
    """Band-pass filter."""
    nyq = sr / 2
    return sosfilt(butter(order, [max(lo, 20) / nyq, min(hi, nyq * 0.98) / nyq],
                          btype='band', output='sos'), x)


def lp(x: np.ndarray, cut: float, order: int = 2, sr: int = SR) -> np.ndarray:
    """Low-pass filter."""
    return sosfilt(butter(order, min(cut, sr / 2 * 0.98) / (sr / 2),
                          btype='low', output='sos'), x)


def hp(x: np.ndarray, cut: float, order: int = 2, sr: int = SR) -> np.ndarray:
    """High-pass filter."""
    return sosfilt(butter(order, max(cut, 20) / (sr / 2), btype='high', output='sos'), x)


def sweep_filter(x: np.ndarray, f0: float, f1: float, q_width: float = 1.6,
                 sr: int = SR) -> np.ndarray:
    """Band-pass whose centre frequency sweeps f0 to f1, applied in overlapping blocks."""
    n = len(x)
    blocks = 24
    out = np.zeros(n)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for i in range(blocks):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        fc = f0 * (f1 / f0) ** ((i + 0.5) / blocks)
        seg = bp(x[max(0, s - 256):e], fc / q_width, fc * q_width, sr=sr)
        out[s:e] = seg[-(e - s):]
    return out


def warm(x: np.ndarray, cut: float = 5500, amount: float = 0.55, sr: int = SR) -> np.ndarray:
    """Gentle high-frequency tilt that pulls the spectral centroid down."""
    return x * (1 - amount) + lp(x, cut, order=2, sr=sr) * amount


def make_ir(dur: float, rt60: float, sr: int = SR, seed: int = 1, hf_damp: float = 6000,
            predelay: float = 0.0) -> np.ndarray:
    """Synthetic impulse response: HF-damped decaying noise plus four early reflections."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / (rt60 * sr))
    tail = lp(tail, hf_damp, sr=sr)
    for delay, gain in ((0.011, 0.5), (0.019, 0.36), (0.029, 0.26), (0.041, 0.18)):
        d = int(delay * sr)
        if d < n:
            tail[d] += gain
    pd = int(predelay * sr)
    if pd:
        tail = np.concatenate([np.zeros(pd), tail])[:n]
    return tail / (np.max(np.abs(tail)) + 1e-9)


def reverb(x: np.ndarray, wet: float = 0.3, rt60: float = 1.2, seed: int = 1,
           hf_damp: float = 6000, predelay: float = 0.01, sr: int = SR) -> np.ndarray:
    """Convolution reverb against a synthetic impulse response."""
    ir = make_ir(min(rt60 * 1.4, 3.0), rt60, sr=sr, seed=seed, hf_damp=hf_damp, predelay=predelay)
    w = fftconvolve(x, ir)[:len(x) + int(rt60 * sr)]
    w /= (np.max(np.abs(w)) + 1e-9)
    dry = np.pad(x, (0, len(w) - len(x)))
    return dry * (1 - wet) + w * wet


def stereo(x: np.ndarray, width: float = 0.35, rt60: float = 1.0, seed: int = 7,
           sr: int = SR) -> np.ndarray:
    """Decorrelates a mono signal into L/R by reverberating it with two different tails."""
    left = reverb(x, wet=width, rt60=rt60, seed=seed, sr=sr)
    right = reverb(x, wet=width, rt60=rt60, seed=seed + 101, sr=sr)
    n = min(len(left), len(right))
    return np.stack([left[:n], right[:n]], axis=1)


def voice(x: np.ndarray, width: float = 0.35, rt60: float = 1.0, seed: int = 7,
          tilt: float = 0.55, sr: int = SR) -> np.ndarray:
    """Finishing chain applied to a finished mono layer stack: stereo spread, then HF tilt."""
    y = stereo(x, width=width, rt60=rt60, seed=seed, sr=sr)
    return np.stack([warm(y[:, 0], amount=tilt, sr=sr), warm(y[:, 1], amount=tilt, sr=sr)], axis=1)


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh saturation, normalised so unity input stays at unity."""
    return np.tanh(x * drive) / np.tanh(drive)


def pad(x: np.ndarray, n: int) -> np.ndarray:
    """Pads or truncates a signal to exactly n samples."""
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def mix(*layers: np.ndarray) -> np.ndarray:
    """Sums layers of differing lengths, zero-padding to the longest."""
    n = max(len(layer) for layer in layers)
    return sum(pad(layer, n) for layer in layers)


def at(x: np.ndarray, offset_ms: float, total_dur: float, sr: int = SR) -> np.ndarray:
    """Places a signal at an offset inside a buffer of `total_dur` seconds."""
    n = int(total_dur * sr)
    off = int(offset_ms / 1000 * sr)
    out = np.zeros(n)
    seg = x[:max(0, n - off)]
    out[off:off + len(seg)] = seg
    return out


def trim(x: np.ndarray, thr: float = 1e-4, sr: int = SR, tail_ms: float = 25) -> np.ndarray:
    """Trims silence below `thr`, keeping a short tail so the fade-out stays smooth."""
    mono = np.abs(x).max(axis=1) if x.ndim > 1 else np.abs(x)
    idx = np.where(mono > thr)[0]
    if len(idx) == 0:
        return x
    end = min(len(mono), idx[-1] + int(tail_ms / 1000 * sr))
    return x[idx[0]:end]


def fade_out(x: np.ndarray, ms: float = 18, sr: int = SR) -> np.ndarray:
    """Applies a short curved fade so a trimmed tail never clicks."""
    n = min(int(ms / 1000 * sr), len(x))
    ramp = np.linspace(1, 0, n) ** 1.6
    y = x.copy()
    if y.ndim > 1:
        y[-n:] *= ramp[:, None]
    else:
        y[-n:] *= ramp
    return y


def normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Scales the signal so its peak sits at `peak_db` dBFS."""
    peak = np.max(np.abs(x))
    if peak < 1e-9:
        return x
    return x / peak * (10 ** (peak_db / 20))


def write(path: str, x: np.ndarray, sr: int = SR, trim_thr: float = 1e-4) -> np.ndarray:
    """Trims, fades, clips and writes a 16-bit WAV; returns the samples actually written."""
    y = np.clip(fade_out(trim(x, thr=trim_thr, sr=sr), sr=sr), -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767).astype(np.int16))
    return y
