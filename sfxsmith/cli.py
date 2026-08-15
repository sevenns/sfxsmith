"""Command-line entry point: render packs, analyse WAVs, build an HTML comparator."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .analyze import analyze
from .engine import normalize, write
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
            samples = normalize(render(), peaks.get(slot, args.peak_db))
            path = pack_dir / f'{slot}.wav'
            written = write(str(path), samples, trim_thr=args.trim)
            print(f'{path}  {len(written) / 48000 * 1000:.0f}ms  '
                  f'{path.stat().st_size / 1024:.0f}KB')


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
    render.set_defaults(func=cmd_render)

    an = sub.add_parser('analyze', help='measure WAV files (duration, centroid, flatness)')
    an.add_argument('files', nargs='+', help='paths or glob patterns')
    an.set_defaults(func=cmd_analyze)

    player = sub.add_parser('player', help='build a self-contained HTML comparator')
    player.add_argument('root', help='directory containing pack folders')
    player.add_argument('-o', '--out', default='player.html', help='output HTML file')
    player.add_argument('--title', default='sfxsmith', help='page title')
    player.set_defaults(func=cmd_player)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
