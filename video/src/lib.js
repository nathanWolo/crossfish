// Drawing helpers for the Crossfish explainer.
//
// Every scene is a pure function of time: draw(ctx, T) paints one 1920x1080
// frame for scene-local time T.t. Nothing is stateful, so frames can be
// rendered in any order and in parallel.

export const W = 1920, H = 1080;

export const C = {
  bg: '#0d0f15', panel: '#161922', panel2: '#1d212c', grid: '#39404f', gridHi: '#6b7385',
  fg: '#ece9e4', dim: '#8b92a1', faint: '#535a69',
  x: '#ff8a4c', o: '#4cc2ff', accent: '#ffd166', good: '#6bd490', bad: '#ff5d6c', violet: '#b79cff',
};
export const SANS = 'Inter, "Helvetica Neue", Arial, sans-serif';
export const MONO = '"JetBrains Mono", "DejaVu Sans Mono", Menlo, monospace';

// ---------- time ----------
export const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
export const lerp = (a, b, u) => a + (b - a) * u;
export const ease = {
  lin: u => u,
  io: u => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2),
  out: u => 1 - Math.pow(1 - u, 3),
  in: u => u * u * u,
  back: u => { const c = 1.70158; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); },
};
/** Progress 0..1 of an animation that starts at t0 and lasts d seconds. */
export const p = (t, t0, d = 0.6, e = ease.io) => e(clamp((t - t0) / d));
/** Alpha that fades in at tin and out at tout (tout = Infinity to stay). */
export const fade = (t, tin, tout = Infinity, d = 0.4) =>
  clamp((t - tin) / d) * (tout === Infinity ? 1 : 1 - clamp((t - tout) / d));
/** Deterministic pseudo-random in [0,1) from an integer seed. */
export const rnd = n => { let x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

// ---------- primitives ----------
export function txt(ctx, s, x, y, o = {}) {
  const a = o.a ?? 1; if (a <= 0.003) return;
  const size = o.size ?? 44;
  ctx.save();
  ctx.globalAlpha *= a;
  ctx.font = `${o.italic ? 'italic ' : ''}${o.weight ?? 500} ${size}px ${o.mono ? MONO : SANS}`;
  ctx.fillStyle = o.color ?? C.fg;
  ctx.textAlign = o.align ?? 'center';
  ctx.textBaseline = o.base ?? 'middle';
  const lines = String(s).split('\n'), lh = (o.lh ?? 1.3) * size;
  const y0 = y - ((lines.length - 1) * lh) / 2 * (o.base ? 0 : 1);
  // o.reveal in 0..1 types the text out left to right.
  let budget = o.reveal === undefined ? Infinity : Math.round(o.reveal * String(s).length);
  lines.forEach((ln, i) => {
    const shown = ln.slice(0, Math.max(0, budget)); budget -= ln.length + 1;
    if (!shown) return;
    let dx = 0;
    if (o.reveal !== undefined && (o.align ?? 'center') === 'center') {
      dx = -ctx.measureText(ln).width / 2; ctx.textAlign = 'left';
    }
    ctx.fillText(shown, x + dx, y0 + i * lh);
  });
  ctx.restore();
}
export function measure(ctx, s, o = {}) {
  ctx.save();
  ctx.font = `${o.weight ?? 500} ${o.size ?? 44}px ${o.mono ? MONO : SANS}`;
  const w = ctx.measureText(s).width; ctx.restore(); return w;
}
/** Text made of differently coloured runs: [[text, color], ...], centred on x. */
export function runs(ctx, parts, x, y, o = {}) {
  const ws = parts.map(([s]) => measure(ctx, s, o)), total = ws.reduce((a, b) => a + b, 0);
  let cx = o.align === 'left' ? x : x - total / 2;
  parts.forEach(([s, color], i) => { txt(ctx, s, cx, y, { ...o, color, align: 'left' }); cx += ws[i]; });
}
export function line(ctx, x1, y1, x2, y2, o = {}) {
  const pr = o.prog ?? 1, a = o.a ?? 1; if (pr <= 0 || a <= 0.003) return;
  ctx.save(); ctx.globalAlpha *= a;
  ctx.strokeStyle = o.color ?? C.fg; ctx.lineWidth = o.w ?? 4; ctx.lineCap = o.cap ?? 'round';
  if (o.dash) ctx.setLineDash(o.dash);
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(lerp(x1, x2, pr), lerp(y1, y2, pr)); ctx.stroke();
  ctx.restore();
}
export function arrow(ctx, x1, y1, x2, y2, o = {}) {
  const pr = o.prog ?? 1; if (pr <= 0) return;
  const ex = lerp(x1, x2, pr), ey = lerp(y1, y2, pr), ang = Math.atan2(y2 - y1, x2 - x1), h = o.head ?? 18;
  line(ctx, x1, y1, ex, ey, { ...o, prog: 1 });
  ctx.save(); ctx.globalAlpha *= o.a ?? 1; ctx.fillStyle = o.color ?? C.fg;
  ctx.beginPath(); ctx.moveTo(ex + Math.cos(ang) * h * 0.6, ey + Math.sin(ang) * h * 0.6);
  ctx.lineTo(ex + Math.cos(ang + 2.5) * h, ey + Math.sin(ang + 2.5) * h);
  ctx.lineTo(ex + Math.cos(ang - 2.5) * h, ey + Math.sin(ang - 2.5) * h);
  ctx.closePath(); ctx.fill(); ctx.restore();
}
export function rrect(ctx, x, y, w, h, o = {}) {
  const a = o.a ?? 1; if (a <= 0.003) return;
  ctx.save(); ctx.globalAlpha *= a;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, o.r ?? 14);
  if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
  if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.w ?? 3; if (o.dash) ctx.setLineDash(o.dash); ctx.stroke(); }
  ctx.restore();
}
export function circle(ctx, x, y, r, o = {}) {
  const a = o.a ?? 1; if (a <= 0.003 || r <= 0) return;
  ctx.save(); ctx.globalAlpha *= a; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
  if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.w ?? 3; ctx.stroke(); }
  ctx.restore();
}
/** A labelled box; returns nothing. Text is centred. */
export function box(ctx, label, x, y, w, h, o = {}) {
  rrect(ctx, x - w / 2, y - h / 2, w, h, { fill: o.fill ?? C.panel, stroke: o.stroke ?? C.grid, a: o.a, r: o.r, w: o.w });
  txt(ctx, label, x, y, { size: o.size ?? 32, color: o.color ?? C.fg, a: o.a, weight: o.weight ?? 500, mono: o.mono });
}

// ---------- marks ----------
export function drawX(ctx, cx, cy, r, o = {}) {
  const pr = o.prog ?? 1, col = o.color ?? C.x, w = o.w ?? Math.max(2.5, r * 0.26);
  line(ctx, cx - r, cy - r, cx + r, cy + r, { prog: clamp(pr * 2), color: col, w, a: o.a });
  line(ctx, cx + r, cy - r, cx - r, cy + r, { prog: clamp(pr * 2 - 1), color: col, w, a: o.a });
}
export function drawO(ctx, cx, cy, r, o = {}) {
  const pr = o.prog ?? 1, a = o.a ?? 1; if (pr <= 0 || a <= 0.003) return;
  ctx.save(); ctx.globalAlpha *= a; ctx.strokeStyle = o.color ?? C.o; ctx.lineWidth = o.w ?? Math.max(2.5, r * 0.26);
  ctx.lineCap = 'round'; ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * pr); ctx.stroke(); ctx.restore();
}
export const mark = (ctx, who, cx, cy, r, o) => (who === 1 ? drawX : drawO)(ctx, cx, cy, r, o);

// ---------- tic-tac-toe ----------
export const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
export function winner(cells) { // 0 none, 1 X, 2 O, 3 full
  for (const [a, b, c] of LINES) if (cells[a] && cells[a] !== 3 && cells[a] === cells[b] && cells[b] === cells[c]) return cells[a];
  return cells.every(v => v) ? 3 : 0;
}
/** A single 3x3 board centred at (cx, cy). cells: 9 values 0/1/2; prog: optional 9 draw-in progresses. */
export function mini(ctx, cx, cy, size, cells, o = {}) {
  const a = o.a ?? 1; if (a <= 0.003) return;
  const s = size / 3, gp = o.gridProg ?? 1, gc = o.grid ?? C.grid, gw = o.gw ?? Math.max(2, size * 0.018);
  ctx.save(); ctx.globalAlpha *= a;
  for (let i = 1; i < 3; i++) {
    line(ctx, cx - size / 2 + i * s, cy - size / 2, cx - size / 2 + i * s, cy + size / 2, { color: gc, w: gw, prog: gp });
    line(ctx, cx - size / 2, cy - size / 2 + i * s, cx + size / 2, cy - size / 2 + i * s, { color: gc, w: gw, prog: gp });
  }
  for (let i = 0; i < 9; i++) {
    if (!cells[i]) continue;
    const x = cx - size / 2 + (i % 3 + 0.5) * s, y = cy - size / 2 + (Math.floor(i / 3) + 0.5) * s;
    mark(ctx, cells[i], x, y, s * 0.27, { prog: o.prog ? o.prog[i] : 1, a: o.markA ?? 1, color: o.colors ? o.colors[i] : undefined });
  }
  ctx.restore();
}
export const cellXY = (cx, cy, size, b, s) => {
  const big = size / 3, sm = (big * 0.8) / 3;
  const bx = cx - size / 2 + (b % 3 + 0.5) * big, by = cy - size / 2 + (Math.floor(b / 3) + 0.5) * big;
  return [bx - big * 0.4 + (s % 3 + 0.5) * sm, by - big * 0.4 + (Math.floor(s / 3) + 0.5) * sm];
};
export const boardXY = (cx, cy, size, b) => [cx - size / 2 + (b % 3 + 0.5) * (size / 3), cy - size / 2 + (Math.floor(b / 3) + 0.5) * (size / 3)];

/**
 * The full 9x9 board. st = { cells[81] (board*9+sq), prog[81], macro[9], macroProg[9],
 * active: [{b, a}] boards to light up, last: [b, s] }.
 */
export function uttt(ctx, cx, cy, size, st, o = {}) {
  const a = o.a ?? 1; if (a <= 0.003) return;
  const big = size / 3, gp = o.gridProg ?? 1;
  ctx.save(); ctx.globalAlpha *= a;
  for (const { b, a: ha, color } of st.active ?? []) {
    const [bx, by] = boardXY(cx, cy, size, b);
    rrect(ctx, bx - big * 0.46, by - big * 0.46, big * 0.92, big * 0.92, { fill: color ?? C.accent, a: 0.13 * ha, r: big * 0.06 });
    rrect(ctx, bx - big * 0.46, by - big * 0.46, big * 0.92, big * 0.92, { stroke: color ?? C.accent, a: 0.7 * ha, r: big * 0.06, w: 3 });
  }
  for (let b = 0; b < 9; b++) {
    const [bx, by] = boardXY(cx, cy, size, b), m = st.macro ? st.macro[b] : 0, mp = st.macroProg ? st.macroProg[b] : 1;
    const dimmed = m ? lerp(1, 0.22, clamp(mp * 1.5)) : 1;
    mini(ctx, bx, by, big * 0.8, st.cells.slice(b * 9, b * 9 + 9), {
      prog: st.prog ? st.prog.slice(b * 9, b * 9 + 9) : undefined, a: dimmed, gridProg: gp, gw: Math.max(1.5, size * 0.0035),
    });
    if (m === 1 || m === 2) mark(ctx, m, bx, by, big * 0.3, { prog: mp, w: big * 0.075 });
  }
  if (st.last && (st.lastA ?? 1) > 0) {
    const [x, y] = cellXY(cx, cy, size, st.last[0], st.last[1]);
    circle(ctx, x, y, big * 0.8 / 3 * 0.47, { stroke: C.fg, w: 2, a: 0.55 * (st.lastA ?? 1) });
  }
  for (let i = 1; i < 3; i++) {
    line(ctx, cx - size / 2 + i * big, cy - size / 2, cx - size / 2 + i * big, cy + size / 2, { color: o.grid ?? C.gridHi, w: Math.max(3, size * 0.008), prog: gp });
    line(ctx, cx - size / 2, cy - size / 2 + i * big, cx + size / 2, cy - size / 2 + i * big, { color: o.grid ?? C.gridHi, w: Math.max(3, size * 0.008), prog: gp });
  }
  ctx.restore();
}

/**
 * Replay a move list up to time t. moves: [{side, board, sq}], when(i) -> time move i lands.
 * Returns a state for uttt(), plus `n` (moves played) and `next` (board the next player is sent to, or -1).
 */
export function replay(moves, t, when, o = {}) {
  const cells = new Array(81).fill(0), prog = new Array(81).fill(0), macro = new Array(9).fill(0), macroProg = new Array(9).fill(0);
  let n = 0, last = null;
  for (let i = 0; i < moves.length; i++) {
    const ti = when(i); if (t < ti) break;
    const m = moves[i], k = m.board * 9 + m.sq;
    cells[k] = m.side + 1; prog[k] = p(t, ti, o.markDur ?? 0.25, ease.out);
    if (!macro[m.board]) {
      const w = winner(cells.slice(m.board * 9, m.board * 9 + 9));
      if (w) { macro[m.board] = w; macroProg[m.board] = p(t, ti + 0.15, o.macroDur ?? 0.45, ease.out); }
    }
    n = i + 1; last = [m.board, m.sq];
  }
  const next = last && !macro[last[1]] ? last[1] : -1;
  return { cells, prog, macro, macroProg, last, n, next };
}

// ---------- chrome ----------
export function bg(ctx) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  const g = ctx.createRadialGradient(W / 2, H / 2, 200, W / 2, H / 2, 1200);
  g.addColorStop(0, 'rgba(60,70,100,0.10)'); g.addColorStop(1, 'rgba(0,0,0,0.25)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
}
/** Small heading in the top-left corner that names the current chapter. */
export function heading(ctx, s, a = 1) {
  txt(ctx, s.toUpperCase(), 80, 70, { size: 26, color: C.dim, align: 'left', weight: 600, a });
}
/** Monospace code panel. lines: [string | [string, color]]. */
export function code(ctx, lines, x, y, o = {}) {
  const size = o.size ?? 30, lh = size * 1.5, a = o.a ?? 1, w = o.w ?? 800, h = lines.length * lh + 44;
  rrect(ctx, x, y, w, h, { fill: C.panel, stroke: C.grid, a, w: 2 });
  lines.forEach((ln, i) => {
    const [s, color] = Array.isArray(ln) ? ln : [ln, C.fg];
    const la = o.lineA ? o.lineA(i) : 1;
    txt(ctx, s, x + 28, y + 22 + lh * (i + 0.5), { size, mono: true, align: 'left', color, a: a * la, weight: 400 });
  });
  return h;
}
/** Chapter title card used by the short interstitial scenes. */
export function titleCard(ctx, t, dur, num, title, sub) {
  bg(ctx);
  const a = fade(t, 0.15, dur - 0.5, 0.4);
  txt(ctx, num, W / 2, 440, { size: 30, color: C.accent, weight: 600, a });
  txt(ctx, title, W / 2, 530, { size: 92, weight: 700, a });
  line(ctx, W / 2 - 220, 610, W / 2 + 220, 610, { color: C.accent, w: 3, prog: p(t, 0.3, 0.8), a });
  if (sub) txt(ctx, sub, W / 2, 670, { size: 36, color: C.dim, a: a * p(t, 0.6, 0.5) });
}

// ---------- shared figures ----------
/** A row of n small cells coloured by pseudo-random "activations"; seed picks the pattern. */
export function vec(ctx, x, y, n, cw, ch, seed, o = {}) {
  const a = o.a ?? 1; if (a <= 0.003) return;
  ctx.save(); ctx.globalAlpha *= a;
  for (let i = 0; i < n; i++) {
    const v = o.values ? o.values[i] : rnd(seed * 97 + i) * 2 - 1;
    const col = v >= 0 ? `rgba(255,138,76,${0.15 + 0.85 * Math.abs(v)})` : `rgba(76,194,255,${0.15 + 0.85 * Math.abs(v)})`;
    const [px, py] = o.vertical ? [x, y + i * ch] : [x + i * cw, y];
    ctx.fillStyle = col; ctx.fillRect(px, py, cw - (o.gap ?? 1), ch - (o.gap ?? 1));
  }
  ctx.restore();
}
/** Big number with a caption under it. */
export function stat(ctx, big, small, x, y, o = {}) {
  txt(ctx, big, x, y, { size: o.size ?? 150, weight: 800, color: o.color ?? C.good, a: o.a });
  if (small) txt(ctx, small, x, y + (o.size ?? 150) * 0.78, { size: o.small ?? 36, color: C.dim, a: o.a });
}
/** Horizontal bar with a label on the left and a value on the right. */
export function hbar(ctx, label, value, x, y, w, frac, o = {}) {
  const a = o.a ?? 1, h = o.h ?? 34;
  txt(ctx, label, x - 20, y, { size: o.size ?? 30, align: 'right', a, color: o.labelColor ?? C.fg });
  rrect(ctx, x, y - h / 2, Math.max(2, w * frac * (o.prog ?? 1)), h, { fill: o.color ?? C.accent, a, r: 6 });
  txt(ctx, value, x + w * frac * (o.prog ?? 1) + 16, y, { size: o.size ?? 30, align: 'left', mono: true, a, color: o.color ?? C.accent });
}
