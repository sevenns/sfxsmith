"""Builds a self-contained HTML comparator so a pack can be auditioned in a browser.

Audio is embedded as data URIs, so the produced file works offline and can be sent to
someone else as a single attachment.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from .analyze import envelope

SLOT_ROLE = {
    'move': 'navigation',
    'button': 'confirm / enter',
    'back': 'cancel / return',
    'play': 'launch',
}

_TEMPLATE = """<title>{title}</title>
<style>
  :root {{
    --bg: #f6f5f2; --panel: #fffefb; --line: #e2ded4; --ink: #1c1b18;
    --muted: #6e6a60; --accent: #b4552b; --accent-soft: #f0dcd0; --wave: #c9c3b6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #14130f; --panel: #1d1c17; --line: #302e26; --ink: #f0ece1;
      --muted: #918b7c; --accent: #e8834f; --accent-soft: #3a2418; --wave: #45423a;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #14130f; --panel: #1d1c17; --line: #302e26; --ink: #f0ece1;
    --muted: #918b7c; --accent: #e8834f; --accent-soft: #3a2418; --wave: #45423a;
  }}
  body {{
    background: var(--bg); color: var(--ink); margin: 0; padding: 40px 24px 80px;
    font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 28px; letter-spacing: -0.02em; margin: 0 0 6px; }}
  .lede {{ color: var(--muted); margin: 0 0 32px; max-width: 62ch; }}
  .card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px 24px; margin-bottom: 18px;
  }}
  .card h2 {{ font-size: 19px; margin: 0 0 14px; letter-spacing: -0.01em; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .snd {{
    flex: 1 1 180px; min-width: 170px; background: transparent; color: inherit;
    border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
    cursor: pointer; text-align: left; font: inherit;
    transition: border-color .12s, background .12s;
  }}
  .snd:hover {{ border-color: var(--accent); }}
  .snd.playing {{ background: var(--accent-soft); border-color: var(--accent); }}
  .snd .name {{
    font-weight: 600; font-size: 14px; display: flex;
    justify-content: space-between; align-items: baseline;
  }}
  .snd .name em {{ font-style: normal; color: var(--muted); font-weight: 400; font-size: 12px; }}
  .snd .role {{ color: var(--muted); font-size: 12px; margin-top: 1px; }}
  .snd svg {{ display: block; width: 100%; height: 34px; margin-top: 8px; }}
  .seq {{
    margin-top: 16px; background: transparent; border: 1px dashed var(--line);
    color: var(--muted); border-radius: 8px; padding: 8px 14px; font: inherit;
    font-size: 13px; cursor: pointer;
  }}
  .seq:hover {{ color: var(--accent); border-color: var(--accent); }}
</style>
<div class="wrap">
  <h1>{title}</h1>
  <p class="lede">Click a tile to play it. &ldquo;Run navigation scenario&rdquo; plays the pack
  as a real path through the interface: three moves, a confirm, a move, a cancel, a launch.</p>
  <div id="cards"></div>
</div>
<script>
  const DATA = {data};
  const cards = document.getElementById('cards');
  const els = {{}};

  function wave(env) {{
    const w = 100, h = 30, mid = h / 2;
    const pts = env.map((v, i) => [(i / (env.length - 1)) * w, Math.max(0.6, v * mid)]);
    const top = pts.map(([x, a]) => `${{x.toFixed(2)}},${{(mid - a).toFixed(2)}}`).join(' ');
    const bot = pts.slice().reverse()
      .map(([x, a]) => `${{x.toFixed(2)}},${{(mid + a).toFixed(2)}}`).join(' ');
    return `<svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="none" aria-hidden="true">
      <polygon points="${{top}} ${{bot}}" fill="var(--wave)"/></svg>`;
  }}

  function play(pack, slot) {{
    const a = els[pack + '/' + slot];
    a.currentTime = 0;
    a.play().catch(() => {{}});
    const btn = document.querySelector(`[data-id="${{pack}}/${{slot}}"]`);
    btn.classList.add('playing');
    setTimeout(() => btn.classList.remove('playing'), 220);
  }}

  function sequence(pack) {{
    const steps = [['move', 0], ['move', 190], ['move', 380], ['button', 700],
                   ['move', 1250], ['back', 1600], ['play', 2200]];
    steps.forEach(([s, t]) => setTimeout(() => play(pack, s), t));
  }}

  Object.entries(DATA).forEach(([pack, slots]) => {{
    const card = document.createElement('div');
    card.className = 'card';
    const tiles = Object.entries(slots).map(([slot, d]) => `
      <button class="snd" data-id="${{pack}}/${{slot}}">
        <span class="name">${{slot}}<em>${{d.dur}} ms &middot; ${{d.kb}} KB</em></span>
        <span class="role">${{d.role}}</span>
        ${{wave(d.env)}}
      </button>`).join('');
    card.innerHTML = `<h2>${{pack}}</h2><div class="row">${{tiles}}</div>
      <button class="seq" data-seq="${{pack}}">&#9654; Run navigation scenario</button>`;
    cards.appendChild(card);
    Object.entries(slots).forEach(([slot, d]) => {{
      const a = new Audio(d.src);
      a.preload = 'auto';
      els[pack + '/' + slot] = a;
    }});
  }});

  cards.addEventListener('click', (e) => {{
    const snd = e.target.closest('[data-id]');
    if (snd) {{ const [p, s] = snd.dataset.id.split('/'); play(p, s); return; }}
    const seq = e.target.closest('[data-seq]');
    if (seq) sequence(seq.dataset.seq);
  }});
</script>
"""


def build(root: Path, out_path: Path, title: str = 'sfxsmith') -> Path:
    """Writes an HTML comparator covering every pack folder directly under `root`."""
    data: dict[str, dict[str, object]] = {}
    for pack_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        slots: dict[str, object] = {}
        for wav in sorted(pack_dir.glob('*.wav')):
            env, dur = envelope(str(wav))
            slots[wav.stem] = {
                'src': 'data:audio/wav;base64,' + base64.b64encode(wav.read_bytes()).decode(),
                'env': env,
                'dur': round(dur * 1000),
                'kb': round(wav.stat().st_size / 1024),
                'role': SLOT_ROLE.get(wav.stem, ''),
            }
        if slots:
            data[pack_dir.name] = slots

    out_path.write_text(_TEMPLATE.format(title=title, data=json.dumps(data)))
    return out_path
