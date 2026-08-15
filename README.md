# sfxsmith

Procedural synthesis of UI sound effects — no samples, no neural audio model, just DSP that
runs anywhere Python does. Sounds are *code*: a pack is a Python module, and rendering it is
deterministic, so the same commit always produces byte-identical WAVs.

Built to give the [Playhook](../playhook) launcher its own sound identity instead of borrowing
a console's. It generalises to any project that needs a coherent set of interface sounds.

## Why not just download a sound pack

Because a pack you download is somebody else's identity, licensed under somebody else's terms,
and it cannot be tuned. Here, "make the confirm sound 20% darker and half as long" is a
two-number edit and a re-render. Diffs are readable, review is possible, and the whole set
stays in one tonality because the tonality is a constant in the source.

## Install

Needs Python 3.11+ (macOS system Python is too old; Homebrew's works).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e .
```

## Use

```bash
sfxsmith render playhook -o out
```

Renders every pack in `packs/playhook.py` to `out/<pack-name>/<slot>.wav`. Add `--only
playhook-aurora` for a single pack, or `--trim 3e-4` to cut reverb tails more aggressively
(smaller files; inaudible difference above roughly `3e-4`).

```bash
sfxsmith analyze 'out/playhook-*/move.wav'
```

Measures audible duration, attack, spectral centroid, 85% rolloff, spectral flatness, peak
level, DC offset and clipped samples. Point it at a reference set you admire and you get the
numbers to aim for.

```bash
sfxsmith track playhook_ambience -o out --mp3
```

Renders long-form looping tracks — ambience beds rather than one-shots — from a module's
`TRACKS` mapping, and reports the loop seam: the discontinuity across the wrap point, relative
to the signal's own sample-to-sample motion. Under 1.0 is inaudible. `--mp3` encodes with
ffmpeg alongside the WAV.

```bash
sfxsmith player out -o player.html
```

Builds one self-contained HTML page with every rendered sound embedded as a data URI: click
a tile to hear it, or play a whole pack as a scripted path through a UI (three moves, a
confirm, a move, a cancel, a launch). Nothing else is needed to audition or share a pack.

## Looping without a seam

An ambience bed has to meet itself at the wrap point, and crossfading there is a patch, not a
fix. `sfxsmith` makes the loop periodic by construction instead:

- `snap` rounds every partial to a whole number of cycles per loop. At a 120 s loop the grid is
  1/120 Hz — under a thousandth of a cent at musical pitches, so the pitch survives intact.
- `lfo` and `window` complete an integer number of cycles, and `window` measures its distance
  circularly, so a chord that starts near the end continues into the next pass.
- `loop_noise` builds noise from random phases in the frequency domain, which is periodic where
  ordinary white noise is not.
- `cyclic` runs a filter over two copies of the signal and keeps the second, so the filter's
  start-up transient never lands on the loop point.
- `loop_reverb` convolves circularly, wrapping the tail into the head.

`write_loop` then writes the result verbatim, skipping the trim and fade that `write` applies to
one-shots — both would cut into the wrap.

## How a pack is written

A pack module exports `PACKS` — a mapping of pack name to slot-name-to-function — and
optionally `PEAK_DB` to stagger slot loudness. Each function returns a stereo float array;
the CLI normalises, trims, fades and writes it.

```python
from sfxsmith.engine import NOTE as N, env_ar, fm_voice, mix, soft_clip, voice

def my_click():
    """A soft FM bell."""
    d = 0.4
    bell = fm_voice(N['E5'], 2.0, 2.2, d, idx_decay=26) * env_ar(d, 0.006, 0.055, curve=5)
    return voice(soft_clip(bell, 1.2) * 0.55, width=0.3, rt60=0.9, seed=11)

PACKS = {'my-pack': {'move': my_click}}
PEAK_DB = {'move': -9.0}
```

The engine gives you envelopes (`env_ar`, `env_perc`), oscillators (`sine`, `glide`,
`fm_voice`), spectra (`additive`, `inharmonic`), filtered noise (`noise`, `bp`, `lp`, `hp`,
`sweep_filter`), a convolution `reverb` on a synthetic impulse response, and `voice` — the
finishing chain that spreads a mono layer stack to stereo and tilts its top end down.

## Method

The three Playhook packs came out of measuring two reference sets rather than guessing:

| | PS5 | Steam Big Picture |
|---|---|---|
| spectral flatness | 0.016–0.054 | 0.028–0.053 |
| centroid | 453–1051 Hz | 497–842 Hz |
| attack | 55–240 ms | 5–155 ms |
| lowest peak | 151 Hz | 43 Hz |

Both are far more *tonal* than intuition suggests — a UI click that is mostly noise reads as
cheap. The first synthesis pass here landed at flatness 0.22–0.51 and sounded like static;
band-limiting the transients and tilting the top end brought it to 0.005–0.055, inside the
reference range. That loop — synthesise, measure, compare, adjust — is the whole method, and
it is why `analyze` ships as a first-class command.

The resulting packs, all in E (Emaj9):

- **playhook-aurora** — soft FM bells, slow attack, long airy tail. The PS5 lineage.
- **playhook-tactile** — hard transient, sub-bass, minimal tail. The Steam lineage.
- **playhook-cartridge** — the hybrid, plus a layer of inharmonic partials at metal-plate
  ratios (1 / 2.41 / 3.83 / 5.17 / 7.03) that reads as a cartridge seating in its slot. That
  layer is the part neither reference has.

`packs/playhook_lowend.py` reworks the Steam lineage from a deeper measurement of it — that set
puts 0.0-0.1% of its energy above 3 kHz, and its `button` transient is an instant onset of
low-mid energy rather than a noise click. `packs/playhook_bell.py` does the same for the PS5
lineage, where brightness runs opposite to intuition: `move` is the dark sound (3.2% hi-mid) and
`play` the bright one (24.1%).

`packs/playhook_ambience.py` is a two-minute bed in C# minor — a key chosen so its scale
contains the C#-E-G#-B the UI packs are built from, since a bed in the references' flat minors
would put a semitone under every click. It moves through four chords rather than sitting on
one, its detune beating switches off entirely for a stretch before each change, and it passes
through silence twice per loop.

## Licence

MIT. The reference sets used for measurement are not redistributed here — only the numbers
measured from them.
