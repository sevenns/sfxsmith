# sfxsmith — contributor & agent guide

Rules for changing this repo without breaking what already sounds right.

## The one thing that must not break: determinism

A pack renders byte-identical output from the same commit. Every noise source takes an
explicit `seed`; there is no unseeded `np.random` anywhere, and there must never be one.
This is what makes a sound reviewable — a diff in the WAV means a diff in the source, not
a different roll of the dice.

Concretely, before merging a change to `engine.py`, re-render an untouched pack and compare
hashes with the previous render. If they differ, you changed behaviour — decide whether that
was intended, and say so.

```bash
sfxsmith render playhook -o /tmp/check && shasum -a 256 /tmp/check/*/*.wav
```

The tail-trim threshold (`trim_thr`, default `1e-4`) is part of that contract: raising it
shortens every file. It is a CLI flag precisely so the default can stay stable.

## You cannot hear the output — measure it

This is the constraint that shapes everything. An agent working here has no ears, so
"sounds good" is not a claim it can make. What it can do:

- **Measure with `sfxsmith analyze`.** The four numbers that matter: audible duration, attack
  time, spectral centroid (brightness), spectral flatness (0 = pure tone, 1 = noise).
- **Compare against a real reference set.** Run `analyze` over sounds from a product whose UI
  audio is respected, and aim inside its ranges. Ranges measured so far: flatness 0.016–0.054,
  centroid 450–1050 Hz, attack 5–240 ms. **A UI sound that measures above ~0.1 flatness reads
  as noise and sounds cheap** — this was the single biggest correction during the first build.
- **Check hygiene:** zero clipped samples, DC offset under ~0.005, peak at the intended dBFS.
- **Hand the human a player.** `sfxsmith player` builds one HTML file with everything embedded.
  Judgement of timbre belongs to whoever has ears; state plainly that you are reporting
  measurements, not impressions.

## Layout

- `sfxsmith/engine.py` — pure DSP primitives. No pack-specific knowledge, ever.
- `sfxsmith/analyze.py` — measurement. Returns a frozen `Report`; printing is the caller's job.
- `sfxsmith/player.py` — HTML comparator, self-contained (data URIs, no external requests).
- `sfxsmith/cli.py` — argparse dispatch. Thin: logic belongs in the modules above.
- `packs/<name>.py` — the sounds themselves. Exports `PACKS`, optionally `PEAK_DB`.

A pack module may import from `sfxsmith.engine`. Nothing in `sfxsmith/` may import from
`packs/` — packs are data to the engine, loaded by path in `cli.load_pack_module`.

## Adding a pack

1. Create `packs/<name>.py` with `PACKS = {'<pack-name>': {'<slot>': fn}}`.
2. Slot names are the consumer's vocabulary, not ours. Playhook wants `move`, `button`,
   `back`, `play`; another project may want something else entirely.
3. Stagger loudness via `PEAK_DB` so navigation never shouts over confirmation. The Playhook
   ladder is a good default: move −9, back −4.5, button −3.5, play −1 dBFS.
4. Keep one pack in one tonality. Mixed keys across slots is what makes a set sound assembled
   from stock libraries.

## Layer recipe that works

Sounds here are built as stacked mono layers, mixed, saturated, then finished with `voice`
(stereo spread + HF tilt) as the last step. A typical confirm sound:

- a **transient** — band-limited noise, sub-10 ms decay, quiet (0.1–0.35 amplitude);
- a **body** — `additive` or `fm_voice` on the root, carrying the pitch;
- a **low anchor** — a sine an octave or two down, or a `glide` that drops;
- optional **air** or **shimmer** — very quiet, or it dominates the measurement.

Band-pass the noise layers; do not high-pass them. An open top end is what pushed the first
build to 0.5 flatness.

## Style

- Type-annotate signatures. Keep functions pure and small.
- **Docstrings on functions, no inline comments.** Anything worth explaining goes in the
  docstring, the README, or the message to the human — not in the body of a function.
- Do not delete existing docstrings or documentation while refactoring.
- Constants that shape a sound (ratios, peak ladders) are named module-level values, not
  literals buried in an expression.

## Dependencies

numpy and scipy only. Adding a third runtime dependency needs a real justification: the point
of this tool is that it runs anywhere Python does, with no audio stack, no native build, no
model download.
