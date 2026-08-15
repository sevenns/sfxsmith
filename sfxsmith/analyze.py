"""Spectral analysis used to judge synthesised sounds against real reference sets.

The four numbers that matter in practice: audible duration, attack time, spectral
centroid (brightness) and spectral flatness (0 = pure tone, 1 = white noise). Matching a
reference set's ranges is what makes a synthesised pack sit next to it without sounding cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


@dataclass(frozen=True)
class Report:
    """Measured characteristics of one sound file."""

    path: str
    sample_rate: int
    audible_ms: float
    attack_ms: float
    centroid_hz: float
    rolloff85_hz: float
    flatness: float
    centroid_start_hz: float
    centroid_end_hz: float
    peaks: list[tuple[float, float]]
    peak_dbfs: float
    dc_offset: float
    clipped_samples: int

    def line(self) -> str:
        """One-line human-readable summary."""
        peaks = ', '.join(f'{f:.0f}Hz({a:.2f})' for f, a in self.peaks[:5])
        return (f'{self.path}\n'
                f'  {self.sample_rate}Hz  audible {self.audible_ms:.0f}ms  '
                f'attack {self.attack_ms:.1f}ms  peak {self.peak_dbfs:+.1f}dBFS\n'
                f'  centroid {self.centroid_hz:.0f}Hz  rolloff85 {self.rolloff85_hz:.0f}Hz  '
                f'flatness {self.flatness:.3f}  drift {self.centroid_start_hz:.0f}'
                f'->{self.centroid_end_hz:.0f}Hz\n'
                f'  dc {self.dc_offset:+.5f}  clipped {self.clipped_samples}\n'
                f'  peaks {peaks}')


def _mono(data: np.ndarray) -> np.ndarray:
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x / 32768.0


def analyze(path: str, silence_thr: float = 0.01) -> Report | None:
    """Measures one WAV file. Returns None for a file that is silent throughout."""
    sr, data = wavfile.read(path)
    x = _mono(data)
    peak = float(np.max(np.abs(x)))
    xn = x / peak if peak > 0 else x

    idx = np.where(np.abs(xn) > silence_thr)[0]
    if len(idx) == 0:
        return None
    seg = xn[idx[0]:idx[-1] + 1]

    attack = float(np.argmax(np.abs(seg))) / sr

    f, _, spectrogram = stft(seg, fs=sr, nperseg=2048, noverlap=1536)
    mag = np.abs(spectrogram)
    spec = mag.mean(axis=1)
    centroid = float(np.sum(f * spec) / np.sum(spec))

    tops: list[tuple[float, float]] = []
    for i in np.argsort(spec)[::-1]:
        if all(abs(f[i] - fp) > 60 for fp, _ in tops):
            tops.append((float(f[i]), float(spec[i])))
        if len(tops) >= 6:
            break
    loudest = max(a for _, a in tops)
    peaks = [(fp, a / loudest) for fp, a in tops]

    cumulative = np.cumsum(spec)
    rolloff = float(f[np.searchsorted(cumulative, 0.85 * cumulative[-1])])

    positive = spec + 1e-12
    flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive))

    per_frame = (f[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-12)
    fifth = max(1, len(per_frame) // 5)

    return Report(
        path=path,
        sample_rate=sr,
        audible_ms=len(seg) / sr * 1000,
        attack_ms=attack * 1000,
        centroid_hz=centroid,
        rolloff85_hz=rolloff,
        flatness=flatness,
        centroid_start_hz=float(per_frame[:fifth].mean()),
        centroid_end_hz=float(per_frame[-fifth:].mean()),
        peaks=peaks,
        peak_dbfs=20 * np.log10(peak + 1e-9),
        dc_offset=float(x.mean()),
        clipped_samples=int(np.sum(np.abs(data) >= 32767)),
    )


def envelope(path: str, points: int = 160) -> tuple[list[float], float]:
    """Peak envelope of a file reduced to `points` values, plus its duration in seconds."""
    sr, data = wavfile.read(path)
    x = np.abs(_mono(data))
    edges = np.linspace(0, len(x), points + 1).astype(int)
    env = [float(x[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    loudest = max(env) or 1.0
    return [round(v / loudest, 3) for v in env], len(x) / sr
