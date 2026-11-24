"""Renders the viewer's (index.html) panel of our network over the first STEPS steps into a GIF, at the viewer's default
5k steps per second: index.html runs in headless Chromium with its animation frames driven at a fixed rate."""
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
STEPS = 60_000
SPEED = 5000                  # steps per second
FPS = 20
SIZE = dict(width=800, height=600)

with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport=SIZE)
    page.add_init_script('window.requestAnimationFrame = f => { window.nextFrame = f; }')   # frames run only when called
    page.goto((HERE / 'index.html').as_uri())
    page.uncheck('#show-adam')
    page.uncheck('#show-target')
    page.add_style_tag(content='#legend, #loss, #bar > :not(#step) { display: none }'   # the network 50px taller, upward
                               '#view { top: -50px; height: calc(100% + 6px) }')
    page.evaluate('''() => {
        document.getElementById('title-ours').textContent = 'Feature Search Learner';
        const [d, r] = [dot, ring];   // units 25% larger
        window.dot = (p, radius, ...rest) => d(p, 1.25 * radius, ...rest);
        window.ring = (p, radius, ...rest) => r(p, 1.25 * radius, ...rest);
    }''')
    start = page.evaluate('last')
    frames = []
    for k in range(STEPS * FPS // SPEED + 1):   # each frame's stats cut to the RMSE
        page.evaluate('ms => { nextFrame(ms); statsOurs.textContent = statsOurs.textContent.split("\\n")[0]; }',
                      start + k * 1000 / FPS)
        frames.append(page.screenshot())

out = HERE / f'unit_search_{STEPS // 1000}k.gif'
# one palette for every frame, so that unchanged pixels are left out of each frame, then lossy compression: under 15 MB
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'image2pipe', '-framerate', str(FPS), '-i', '-', '-vf',
                'split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none:diff_mode=rectangle', out],
               input=b''.join(frames), check=True)
subprocess.run(['gifsicle', '-b', '-O3', '--lossy=60', out], check=True)
print(f'wrote {out.name} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)')
