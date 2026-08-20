"""Command-line entry point: render packs, analyse WAVs, build an HTML comparator."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from .analyze import analyze
from .engine import loop_seam, normalize, write, write_loop
from .player import build

PACKS_DIR = Path(__file__).resolve().parent.parent / 'packs'


def load_pack_module(name: str):
    """Imports a pack module by file stem from the packs/ directory."""
    path = PACKS_DIR / f'{name}.py'
    if not path.exists():
        available = ', '.join(sorted(p.stem for p in PACKS_DIR.glob('*.py')))
        raise SystemExit(f'no pack module {name!r} in {PACKS_DIR} (available: {available})')
    spec = importlib.util.spec_from_file_location(f'packs.{name}', path)
    if spec is None or spec.loader is None:
        raise SystemExit(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because that is what importlib expects of any loader: a module
    # defining a dataclass resolves its own annotations through sys.modules, and without this the
    # decorator fails with an AttributeError that names neither the pack nor the cause.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_render(args: argparse.Namespace) -> None:
    """Renders every pack in a module to `<out>/<pack-name>/<slot>.wav`."""
    module = load_pack_module(args.module)
    peaks = getattr(module, 'PEAK_DB', {})
    out_root = Path(args.out)
    for pack_name, slots in module.PACKS.items():
        if args.only and pack_name not in args.only:
            continue
        pack_dir = out_root / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)
        for slot, render in slots.items():
            # A pack may override one slot's level with a 'pack-name/slot' key. Needed whenever
            # packs in one module differ in register: peak dBFS ignores the ear's sensitivity
            # curve, so a sound built on a 41 Hz fundamental measures level with its siblings
            # and still sounds several decibels quieter than them.
            target = peaks.get(f'{pack_name}/{slot}', peaks.get(slot, args.peak_db))
            samples = normalize(render(), target)
            path = pack_dir / f'{slot}.wav'
            written = write(str(path), samples, trim_thr=args.trim)
            print(f'{path}  {len(written) / 48000 * 1000:.0f}ms  '
                  f'{path.stat().st_size / 1024:.0f}KB')
            if args.mp3:
                mp3 = to_mp3(path, args.bitrate)
                print(f'{mp3}  {mp3.stat().st_size / 1024:.0f}KB  ({args.bitrate})')


def expand(pattern: str) -> list[Path]:
    """Expands a glob pattern, handling absolute patterns as well as relative ones."""
    p = Path(pattern)
    if p.is_absolute():
        anchor = Path(p.anchor)
        matches = sorted(anchor.glob(str(p.relative_to(anchor))))
    else:
        matches = sorted(Path().glob(pattern))
    return matches or [p]


def cmd_analyze(args: argparse.Namespace) -> None:
    """Prints spectral measurements for each given WAV file."""
    for pattern in args.files:
        for path in expand(pattern):
            report = analyze(str(path))
            print(report.line() if report is not None else f'{path}: silent')
            print()


def to_mp3(wav_path: Path, bitrate: str) -> Path:
    """Encodes a rendered WAV to MP3 with ffmpeg, leaving the WAV in place."""
    mp3_path = wav_path.with_suffix('.mp3')
    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', str(wav_path), '-codec:a', 'libmp3lame',
         '-b:a', bitrate, str(mp3_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f'ffmpeg failed: {result.stderr.strip()}')
    return mp3_path


def cmd_track(args: argparse.Namespace) -> None:
    """Renders long-form looping tracks from a module's TRACKS mapping."""
    module = load_pack_module(args.module)
    tracks = getattr(module, 'TRACKS', None)
    if tracks is None:
        raise SystemExit(f'module {args.module!r} defines no TRACKS')
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for name, render in tracks.items():
        if args.only and name not in args.only:
            continue
        samples = render()
        path = out_root / f'{name}.wav'
        write_loop(str(path), samples)
        seam = loop_seam(samples)
        size = path.stat().st_size / 1024 / 1024
        print(f'{path}  {len(samples) / 48000:.1f}s  {size:.1f} MB  seam {seam:.2f}'
              f'{"" if seam < 1.0 else "  ← AUDIBLE, loop is not periodic"}')
        if args.mp3:
            mp3 = to_mp3(path, args.bitrate)
            print(f'{mp3}  {mp3.stat().st_size / 1024 / 1024:.1f} MB  ({args.bitrate})')


def cmd_player(args: argparse.Namespace) -> None:
    """Builds the HTML comparator for a directory of pack folders."""
    out = build(Path(args.root), Path(args.out), title=args.title)
    print(f'{out}  {out.stat().st_size / 1024 / 1024:.2f} MB')


def main(argv: list[str] | None = None) -> None:
    """Parses arguments and dispatches to a subcommand."""
    parser = argparse.ArgumentParser(prog='sfxsmith', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    render = sub.add_parser('render', help='render a pack module to WAV files')
    render.add_argument('module', help='pack module name (file stem under packs/)')
    render.add_argument('-o', '--out', default='out', help='output directory (default: out)')
    render.add_argument('--only', nargs='*', help='render only these pack names')
    render.add_argument('--peak-db', type=float, default=-1.0,
                        help='fallback peak for slots the module does not list')
    render.add_argument('--trim', type=float, default=1e-4,
                        help='silence threshold for tail trimming (raise to shrink files)')
    render.add_argument('--mp3', action='store_true', help='also encode to MP3 (needs ffmpeg)')
    render.add_argument('--bitrate', default='192k', help='MP3 bitrate (default: 192k)')
    render.set_defaults(func=cmd_render)

    an = sub.add_parser('analyze', help='measure WAV files (duration, centroid, flatness)')
    an.add_argument('files', nargs='+', help='paths or glob patterns')
    an.set_defaults(func=cmd_analyze)

    track = sub.add_parser('track', help='render long-form looping tracks (ambience beds)')
    track.add_argument('module', help='pack module name (file stem under packs/)')
    track.add_argument('-o', '--out', default='out', help='output directory (default: out)')
    track.add_argument('--only', nargs='*', help='render only these track names')
    track.add_argument('--mp3', action='store_true', help='also encode to MP3 (needs ffmpeg)')
    track.add_argument('--bitrate', default='192k', help='MP3 bitrate (default: 192k)')
    track.set_defaults(func=cmd_track)

    player = sub.add_parser('player', help='build a self-contained HTML comparator')
    player.add_argument('root', help='directory containing pack folders')
    player.add_argument('-o', '--out', default='player.html', help='output HTML file')
    player.add_argument('--title', default='sfxsmith', help='page title')
    player.set_defaults(func=cmd_player)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
