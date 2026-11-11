import { W, H, C, p, fade, ease, clamp, lerp, txt, line, circle, rrect, heading, titleCard, measure, arrow, stat, rnd, box } from '../lib.js';

export const title = { id: 't_sprt', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 4', 'How do you know?', 'measuring strength') };

// Deterministic random walks standing in for a log-likelihood ratio.
function walk(seed, drift, n) { const v = [0]; for (let i = 1; i < n; i++) v.push(v[i - 1] + drift + (rnd(seed * 1000 + i) - 0.5) * 0.9); return v; }
const PASS = walk(3, 0.034, 400), FAIL = walk(8, -0.03, 400);
const cross = (v, b) => { const i = v.findIndex(x => (b > 0 ? x >= b : x <= b)); return i < 0 ? v.length : i; };

const REJECTED = [['Null-move pruning', ''], ['Razoring', ''], ['16× bigger hash table', 'searched 3.9% more nodes'], ['Hand-reweighted evaluation', '−110'], ['Send-to-threat move ordering', '−45'], ['Multi-ProbCut', '−9.7']];
const ACCEPTED = [['Small-board lookup tables', '+55'], ['Selective search bundle', '+80'], ['Precomputed eval tables', '+99.8'], ['Correction history + log reductions', '+30.6'], ['Hash keys that merge finished boards', '+18.8'], ['Hot-path rewrite, same search tree', '+18.8']];

export default {
  id: 'sprt', title: 'Chapter 4: How do you know?',
  segs: [
    { id: 'm1', text: 'That raised an uncomfortable question. If a bug that large could hide for two years, how would I know whether any change I make is an improvement?' },
    { id: 'm2', text: 'You cannot tell by watching games. A genuine five Elo improvement moves the expected score from fifty percent to fifty point seven. No one can see that.', say: 'You cannot tell by watching games. A genuine five E-low improvement moves the expected score from fifty percent to fifty point seven. No one can see that.' },
    { id: 'm3', text: 'So every change goes through the same loop. Freeze the current engine. Change one thing in a copy. Then let them play thousands of games from balanced openings, each opening twice with the colours swapped. A statistical test called S P R T watches the results, and stops as soon as it is confident either way.' },
    { id: 'm4', text: 'It is humbling. One time, a copy of the engine with no changes at all was tested against itself. After three thousand games it looked seven Elo stronger. That was pure noise.', say: 'It is humbling. One time, a copy of the engine with no changes at all was tested against itself. After three thousand games it looked seven E-low stronger. That was pure noise.' },
    { id: 'm5', text: 'And most ideas fail. Null-move pruning, razoring, a bigger hash table: all standard in chess engines, and all of them made this one worse. A line from the project log: chess folklore is not a patch list.' },
    { id: 'm6', text: 'The ones that passed were kept. Through September the engine got faster and smarter in small, measured steps.', pause: 1.2 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 4 · How do you know?', fade(t, 0.3));
    const a1 = fade(t, 0.3, s('m2') - 0.4);
    txt(ctx, 'How would I know if a change', W / 2, 460, { size: 70, weight: 600, a: a1 });
    txt(ctx, 'is an improvement?', W / 2, 560, { size: 70, weight: 600, a: a1, color: C.accent });

    // m2: what 5 Elo looks like
    const m2 = s('m2'), a2 = fade(t, m2 + 0.2, s('m3') - 0.4);
    if (a2 > 0) {
      const x0 = 360, w = 1200, sh = p(t, m2 + 5.5, 1.2) * 0.007;
      txt(ctx, 'expected score', W / 2, 300, { size: 34, color: C.dim, a: a2 });
      rrect(ctx, x0, 420, w * (0.5 + sh), 110, { fill: C.x, a: a2, r: 10 });
      rrect(ctx, x0 + w * (0.5 + sh), 420, w * (0.5 - sh), 110, { fill: C.o, a: a2, r: 10 });
      line(ctx, x0 + w / 2, 390, x0 + w / 2, 560, { color: C.fg, w: 2, dash: [6, 8], a: a2 });
      txt(ctx, (50 + sh * 100).toFixed(1) + '%', x0 + w * 0.25, 475, { size: 56, weight: 700, mono: true, color: C.bg, a: a2 });
      txt(ctx, (50 - sh * 100).toFixed(1) + '%', x0 + w * 0.75, 475, { size: 56, weight: 700, mono: true, color: C.bg, a: a2 });
      txt(ctx, 'new version', x0 + w * 0.25, 610, { size: 30, color: C.x, a: a2 }); txt(ctx, 'old version', x0 + w * 0.75, 610, { size: 30, color: C.o, a: a2 });
      txt(ctx, '+5 Elo', W / 2, 780, { size: 80, weight: 700, a: a2 * p(t, m2 + 4.0, 0.5) });
    }

    // m3: the loop and the sequential test
    const m3 = s('m3'), a3 = fade(t, m3 + 0.2, s('m4') - 0.4);
    if (a3 > 0) {
      box(ctx, 'Prev', 330, 330, 300, 110, { size: 44, stroke: C.o, a: a3 * p(t, m3 + 2.4, 0.4), weight: 700 });
      txt(ctx, 'frozen', 330, 420, { size: 28, color: C.dim, a: a3 * p(t, m3 + 2.4, 0.4) });
      box(ctx, 'Dev', 330, 560, 300, 110, { size: 44, stroke: C.x, a: a3 * p(t, m3 + 4.2, 0.4), weight: 700 });
      txt(ctx, 'one change', 330, 650, { size: 28, color: C.dim, a: a3 * p(t, m3 + 4.2, 0.4) });
      txt(ctx, 'vs', 330, 448 + 42, { size: 30, color: C.faint, a: a3 * p(t, m3 + 4.2, 0.4) });
      txt(ctx, '50,000 balanced openings · each played twice, colours swapped · 90 ms per move', 330, 800, { size: 24, color: C.dim, a: a3 * p(t, m3 + 8.5, 0.5), align: 'left' });
      // chart
      const gx = 640, gy = 250, gw = 1120, gh = 460, ya = v => gy + gh / 2 - v * (gh / 2 / 3.6), ca = a3 * p(t, m3 + 6.5, 0.5);
      rrect(ctx, gx, gy, gw, gh, { fill: C.panel, stroke: C.grid, a: ca, w: 2 });
      line(ctx, gx, ya(3), gx + gw, ya(3), { color: C.good, w: 3, dash: [10, 10], a: ca }); txt(ctx, 'accept  +3', gx + gw - 14, ya(3) - 24, { size: 24, color: C.good, align: 'right', a: ca });
      line(ctx, gx, ya(-3), gx + gw, ya(-3), { color: C.bad, w: 3, dash: [10, 10], a: ca }); txt(ctx, 'reject  −3', gx + gw - 14, ya(-3) + 26, { size: 24, color: C.bad, align: 'right', a: ca });
      line(ctx, gx, ya(0), gx + gw, ya(0), { color: C.grid, w: 2, a: ca });
      txt(ctx, 'evidence (log-likelihood ratio)', gx + 16, gy + 26, { size: 24, color: C.dim, align: 'left', a: ca });
      txt(ctx, 'games played →', gx + gw - 14, gy + gh + 30, { size: 24, color: C.dim, align: 'right', a: ca });
      const draw = (v, col, t0, lab) => {
        const end = cross(v, col === C.good ? 3 : -3), n = Math.floor(clamp((t - t0) / 5.5) * end);
        if (n < 2) return;
        ctx.save(); ctx.globalAlpha *= ca; ctx.strokeStyle = col; ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let i = 0; i <= n; i++) { const x = gx + (i / 400) * gw, y = ya(clamp(v[i], -3, 3)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke(); ctx.restore();
        if (n >= end) { circle(ctx, gx + (end / 400) * gw, ya(col === C.good ? 3 : -3), 10, { fill: col, a: ca }); txt(ctx, lab, gx + (end / 400) * gw + 20, ya(col === C.good ? 3 : -3) + (col === C.good ? 34 : -34), { size: 28, weight: 700, color: col, align: 'left', a: ca }); }
      };
      draw(PASS, C.good, m3 + 9.0, 'PASS: freeze it'); draw(FAIL, C.bad, m3 + 11.5, 'FAIL: revert');
      txt(ctx, 'illustrative runs', gx + 16, gy + gh - 22, { size: 20, color: C.faint, align: 'left', a: ca });
    }

    const m4 = s('m4'), a4 = fade(t, m4 + 0.2, s('m5') - 0.4);
    txt(ctx, 'identical engines · 3,000 games', W / 2, 330, { size: 40, color: C.dim, a: a4 });
    stat(ctx, '+7.07 ± 10.66 Elo', 'after 8,016 games: −1.43 ± 6.53', W / 2, 520, { a: a4 * p(t, m4 + 7.0, 0.5), size: 120, color: C.accent, small: 34 });

    // m5: the graveyard
    const m5 = s('m5'), a5 = fade(t, m5 + 0.2, s('m6') - 0.4);
    if (a5 > 0) {
      txt(ctx, 'Rejected', 420, 230, { size: 44, weight: 700, color: C.bad, a: a5, align: 'left' });
      REJECTED.forEach(([lab, v], i) => {
        const q = a5 * p(t, m5 + 1.6 + i * (i < 3 ? 1.1 : 0.35) + (i >= 3 ? 2.4 : 0), 0.4), y = 320 + i * 78;
        txt(ctx, lab, 420, y, { size: 38, align: 'left', a: q });
        const wdt = measure(ctx, lab, { size: 38 });
        line(ctx, 412, y + 2, 428 + wdt, y + 2, { color: C.bad, w: 3, prog: p(t, m5 + 2.2 + i * 0.5 + 4, 0.4), a: q * 0.8 });
        txt(ctx, v, 1500, y, { size: 32, mono: true, align: 'right', color: C.bad, a: q });
      });
      txt(ctx, '“Chess folklore is not a patch list.”', W / 2, 880, { size: 50, weight: 600, italic: true, a: a5 * p(t, m5 + 12.5, 0.6) });
    }

    const m6 = s('m6'), a6 = fade(t, m6 + 0.2);
    if (a6 > 0) {
      txt(ctx, 'Accepted', 420, 230, { size: 44, weight: 700, color: C.good, a: a6, align: 'left' });
      ACCEPTED.forEach(([lab, v], i) => {
        const q = a6 * p(t, m6 + 0.6 + i * 0.6, 0.4), y = 320 + i * 78;
        txt(ctx, lab, 420, y, { size: 38, align: 'left', a: q });
        txt(ctx, v, 1500, y, { size: 36, mono: true, align: 'right', color: C.good, a: q, weight: 700 });
      });
      txt(ctx, 'Elo against the version just before each change, August to September 2026. These do not simply add up.', W / 2, 880, { size: 26, color: C.dim, a: a6 });
    }
  },
};
