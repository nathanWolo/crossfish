// Timeline assembly. Loaded by index.html (preview) and by render.mjs (frames).
import { W, H, C, bg } from './lib.js';
import { SCENES } from './scenes/index.js';

export const FPS = 30;
const GAP = 0.35;        // default silence after each narration segment
const FADE = 0.35;       // scene cross-fade through black

/**
 * Lay the scenes out on a timeline. `durations` maps "scene.seg" to the length
 * in seconds of that segment's recorded narration; without it, a reading speed
 * of 2.6 words per second is assumed so the preview still works.
 */
export function buildTimeline(durations = {}) {
  let t = 0; const scenes = [];
  for (const sc of SCENES) {
    let local = 0; const segs = {}; const list = [];
    for (const sg of sc.segs) {
      const key = `${sc.id}.${sg.id}`;
      const d = sg.silence ?? durations[key] ?? (sg.text.split(/\s+/).length / 2.6);
      const pre = sg.pre ?? 0;
      const slot = { id: sg.id, key, text: sg.text, start: local + pre, dur: d, end: local + pre + d };
      local = slot.end + (sg.pause ?? (sg.silence ? 0 : GAP));
      segs[sg.id] = slot; list.push(slot);
    }
    scenes.push({ id: sc.id, title: sc.title, start: t, dur: local, segs, list, draw: sc.draw, data: sc.data });
    t += local;
  }
  return { scenes, total: t };
}

export function drawFrame(ctx, tl, time) {
  let sc = tl.scenes[tl.scenes.length - 1];
  for (const s of tl.scenes) if (time < s.start + s.dur) { sc = s; break; }
  const t = time - sc.start;
  const T = {
    t, dur: sc.dur,
    s: id => { if (!sc.segs[id]) throw new Error(`no segment ${sc.id}.${id}`); return sc.segs[id].start; },
    e: id => sc.segs[id].end,
  };
  ctx.save(); ctx.clearRect(0, 0, W, H); bg(ctx);
  sc.draw(ctx, T);
  ctx.restore();
  const k = Math.min(1, t / FADE, (sc.dur - t) / FADE);
  if (k < 1) { ctx.fillStyle = C.bg; ctx.globalAlpha = 1 - Math.max(0, k); ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1; }
}

let _tl, _ctx;
export async function init(durations) {
  await Promise.all(['500 40px Inter', '700 40px Inter', '400 40px "JetBrains Mono"'].map(f => document.fonts.load(f)));
  const cv = document.getElementById('c'); cv.width = W; cv.height = H;
  _ctx = cv.getContext('2d'); _tl = buildTimeline(durations);
  return { total: _tl.total, scenes: _tl.scenes.map(s => ({ id: s.id, title: s.title, start: s.start, dur: s.dur, list: s.list })) };
}
export function frame(time) { drawFrame(_ctx, _tl, time); }
window.cf = { init, frame, buildTimeline, FPS };
