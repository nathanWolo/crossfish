// Render frames with headless Chromium and encode them with ffmpeg.
//   node tools/render.mjs                 full video -> build/crossfish_explainer.mp4
//   node tools/render.mjs --still 12.5    one PNG    -> build/still_12.5.png
//   node tools/render.mjs --scene rules   one scene  -> build/scene_rules.mp4 (no audio)
//   node tools/render.mjs --sheet         contact sheet of every scene -> build/sheet_*.png
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');   // resolves a global install too

const args = process.argv.slice(2);
const flag = n => { const i = args.indexOf(n); if (i < 0) return null; const v = args[i + 1]; return v === undefined || v.startsWith('--') ? true : v; };
const FPS = 30, WORKERS = Number(flag('--workers') ?? 4), ROOT = process.cwd();
const durations = fs.existsSync('build/timings.json') ? JSON.parse(fs.readFileSync('build/timings.json')) : {};
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.wav': 'audio/wav' };

const server = http.createServer((req, res) => {
  const f = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(f)] ?? 'application/octet-stream' }); fs.createReadStream(f).pipe(res);
}).listen(0);
const port = await new Promise(r => server.on('listening', () => r(server.address().port)));
const browser = await chromium.launch();
async function openPage() {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => { console.error('PAGE ERROR', e.message); process.exitCode = 1; });
  await page.goto(`http://localhost:${port}/index.html?render`);
  const info = await page.evaluate(d => window.cf.init(d), durations);
  return { page, info };
}
const grab = (page, t, type = 'jpeg') => page.evaluate(([t, type]) => {
  window.cf.frame(t); return document.getElementById('c').toDataURL('image/' + type, 0.93).split(',')[1];
}, [t, type]).then(b => Buffer.from(b, 'base64'));

async function renderRange(t0, t1, out) {
  const n = Math.round((t1 - t0) * FPS), per = Math.ceil(n / WORKERS), parts = [];
  await Promise.all([...Array(WORKERS)].map(async (_, w) => {
    const a = w * per, b = Math.min(n, a + per); if (a >= b) return;
    const part = `${out}.part${w}.mp4`; parts[w] = part;
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', part], { stdio: ['pipe', 'inherit', 'inherit'] });
    const { page } = await openPage();
    for (let i = a; i < b; i++) {
      const buf = await grab(page, t0 + i / FPS);
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (w === 0 && (i - a) % 300 === 0) process.stderr.write(`\r${Math.round(100 * (i - a) / (b - a))}% `);
    }
    ff.stdin.end(); await new Promise(r => ff.on('close', r)); await page.close();
  }));
  const list = `${out}.txt`;
  fs.writeFileSync(list, parts.filter(Boolean).map(f => `file '${path.resolve(f)}'`).join('\n'));
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', out]);
  parts.filter(Boolean).forEach(f => fs.unlinkSync(f)); fs.unlinkSync(list);
}

const { page, info } = await openPage();
fs.writeFileSync('build/timeline.json', JSON.stringify(info, null, 1));
console.error(`timeline: ${info.scenes.length} scenes, ${(info.total / 60).toFixed(2)} min`);
if (flag('--still')) {
  for (const t of String(flag('--still')).split(',')) fs.writeFileSync(`build/still_${t}.png`, await grab(page, Number(t), 'png'));
} else if (flag('--sheet')) {
  // Several evenly spaced frames per scene, tiled, for a quick visual review.
  const only = flag('--sheet') === true ? null : String(flag('--sheet')).split(',');
  const per = Number(flag('--n') ?? 8);
  for (const s of info.scenes) {
    if (only && !only.includes(s.id)) continue;
    const dir = `build/sheet_tmp`; fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir);
    for (let i = 0; i < per; i++) fs.writeFileSync(`${dir}/${String(i).padStart(3, '0')}.jpg`, await grab(page, s.start + s.dur * (i + 0.7) / per));
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', `${dir}/%03d.jpg`, '-vf', `scale=960:-1,tile=${flag('--cols') ?? 2}x${Math.ceil(per / Number(flag('--cols') ?? 2))}`, '-frames:v', '1', `build/sheet_${s.id}.png`]);
  }
} else if (flag('--scene')) {
  const s = info.scenes.find(s => s.id === flag('--scene'));
  await renderRange(s.start, s.start + s.dur, `build/scene_${s.id}.mp4`);
} else {
  await renderRange(0, info.total, 'build/video_only.mp4');
}
await browser.close(); server.close();
