import { W, H, C, p, fade, ease, clamp, lerp, txt, line, rrect, heading, titleCard, stat, uttt, replay } from '../lib.js';
import { GAMES } from '../games.js';

export const title = { id: 't_outro', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 9', 'Diminishing returns', 'October 2026') };

const STEPS = [['first network', 267.4, 11.9], ['trained on its own games', 52.5, 7.0], ['80 million positions', 16.0, 10.9], ['filtered labels, new loss', 9.1, 5.9], ['ProbCut pruning', 16.0, 9.2], ['wider encoder', 13.8, 5.9]];
const G = GAMES.new_vs_legend.moves;

export default {
  id: 'outro', title: 'Chapter 9: Diminishing returns',
  segs: [
    { id: 'o1', text: 'After the big jump, the returns shrank. Retraining the network on its own games: plus fifty. More data: plus sixteen. Cleaner labels: plus nine. A wider encoder: plus fourteen. Each of those needed thousands of games just to detect.' },
    { id: 'o2', text: 'The ladder moved as well. Other authors kept improving their bots, and the latest version of Crossfish placed between sixth and ninth, not first.' },
    { id: 'o3', text: 'Looking back, the biggest gains did not come from the most sophisticated ideas. A hash table whose keys were never filled in. A compiler flag that was never set. And a testing loop strict enough to tell a real improvement from noise.' },
    { id: 'o4', text: 'The code is linked below, along with the log of every experiment, including the ones that failed. Thanks for watching.', pause: 3.0 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 9 · Diminishing returns', fade(t, 0.3, s('o4') - 0.4));
    const o1 = s('o1'), a1 = fade(t, 0.3, s('o2') - 0.4);
    if (a1 > 0) {
      const x0 = 330, bw = 150, gap = 70, y0 = 800, sc = 2.0, times = [0.6, 3.6, 6.4, 8.4, 9.6, 10.6];
      line(ctx, x0 - 40, y0, x0 + 6 * (bw + gap), y0, { color: C.gridHi, w: 3, a: a1 });
      txt(ctx, 'Elo gained over the previous version (with error bars)', x0 + 6 * (bw + gap), 260, { size: 32, color: C.dim, align: 'right', a: a1 });
      STEPS.forEach(([lab, v, err], i) => {
        const q = p(t, o1 + times[i], 0.8, ease.out), x = x0 + i * (bw + gap), h = v * sc * q;
        rrect(ctx, x, y0 - h, bw, h, { fill: i === 0 ? C.accent : C.good, a: a1, r: 6 });
        line(ctx, x + bw / 2, y0 - (v + err) * sc * q, x + bw / 2, y0 - (v - err) * sc * q, { color: C.fg, w: 3, a: a1 * clamp(q * 2 - 1) });
        txt(ctx, '+' + v.toFixed(v > 100 ? 0 : 1), x + bw / 2, y0 - (v + err) * sc * q - 34, { size: 38, weight: 700, mono: true, a: a1 * clamp(q * 2), color: i === 0 ? C.accent : C.good });
        txt(ctx, lab.replace(' ', '\n').replace(', ', ',\n'), x + bw / 2, y0 + 56, { size: 24, color: C.dim, a: a1 * clamp(q * 2), lh: 1.25 });
      });
    }
    const o2 = s('o2'), a2 = fade(t, o2 + 0.2, s('o3') - 0.4);
    txt(ctx, 'CodinGame ladder placements', W / 2, 300, { size: 38, color: C.dim, a: a2 });
    [['27 Sep', '#1', C.accent], ['4 Oct', '#3 – #4', C.fg], ['9 Oct', '#6', C.fg], ['10 Oct', '#8 – #9', C.fg]].forEach(([d, r, col], i) => {
      const q = a2 * p(t, o2 + 0.8 + i * 1.3, 0.5), x = 420 + i * 360;
      txt(ctx, r, x, 500, { size: 84, weight: 800, color: col, a: q }); txt(ctx, d, x, 590, { size: 30, color: C.dim, a: q });
    });
    txt(ctx, 'the field keeps improving, and placements on different days are not directly comparable', W / 2, 760, { size: 28, color: C.dim, a: a2 * p(t, o2 + 4.5, 0.5) });

    const o3 = s('o3'), a3 = fade(t, o3 + 0.2, s('o4') - 0.4);
    [['a hash table whose keys were never filled in', '+121', 5.6], ['a compiler flag that was never set', '+163', 9.0], ['a testing loop strict enough to trust', '', 11.6]].forEach(([lab, v, dt], i) => {
      const q = a3 * p(t, o3 + dt, 0.5), y = 380 + i * 140;
      txt(ctx, lab, 360, y, { size: 52, weight: 600, align: 'left', a: q });
      txt(ctx, v, 1620, y, { size: 56, weight: 800, mono: true, align: 'right', color: C.good, a: q });
    });

    // end card: replay the win over the 2024 bot behind the credits
    const o4 = s('o4'), a4 = fade(t, o4 + 0.2);
    if (a4 > 0) {
      const st = replay(G, t, i => o4 + 0.5 + i * ((T.dur - o4 - 2.0) / G.length));
      uttt(ctx, 520, 560, 700, st, { a: a4 });
      txt(ctx, 'Crossfish 2026 (X) vs Crossfish 2024 (O)', 520, 970, { size: 26, color: C.dim, a: a4 });
      txt(ctx, 'CROSSFISH', 1380, 400, { size: 110, weight: 800, a: a4 });
      txt(ctx, 'github.com/nathanWolo/crossfish', 1380, 520, { size: 40, mono: true, color: C.accent, a: a4 });
      txt(ctx, 'code · tests · the full improvement log', 1380, 600, { size: 32, color: C.dim, a: a4 });
      txt(ctx, 'Thanks for watching', 1380, 760, { size: 48, weight: 600, a: a4 * p(t, o4 + 6.5, 0.6) });
    }
  },
};
