import { W, H, C, p, fade, ease, clamp, lerp, txt, line, circle, rrect, heading, titleCard, mini, arrow, code } from '../lib.js';

export const title = { id: 't_fast', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 2', 'Speed', 'February 2024') };

const CELLS = [1, 0, 0, 0, 1, 0, 0, 0, 2];        // X on cells 0 and 4, O on cell 8

export default {
  id: 'fast', title: 'Chapter 2: Speed',
  segs: [
    { id: 'f1', text: 'The next stop was CodinGame, which runs a permanent online ladder for this game. The limit there is one hundred milliseconds per move, so I rewrote the engine in C++.', say: 'The next stop was Coding Game, which runs a permanent online ladder for this game. The limit there is one hundred milliseconds per move, so I rewrote the engine in C plus plus.' },
    { id: 'f2', text: 'Speed starts with how you store the board. Each small board is just two nine-bit numbers: one bit per cell for X, and one bit per cell for O.' },
    { id: 'f3', text: 'And a small board only has three to the power of nine possible states. That is nineteen thousand, six hundred and eighty-three. Few enough to work out everything about every one of them in advance: whether it is won, which cells would complete a line, what it is worth. While searching, the engine computes almost nothing about a small board. It looks it up.' },
    { id: 'f4', text: 'In March 2024 that engine reached Legend, the top league on the site. And then I left it alone for more than two years.', pause: 1.0 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 2 · Speed', fade(t, 0.3));
    const a1 = fade(t, 0.3, s('f2') - 0.4);
    txt(ctx, 'CodinGame ladder', W / 2, 360, { size: 40, color: C.dim, a: a1 });
    txt(ctx, '100 ms per move', W / 2, 460, { size: 96, weight: 700, a: a1 * p(t, s('f1') + 4.0, 0.5) });
    const q = p(t, s('f1') + 7.6, 0.6);
    txt(ctx, 'Python', W / 2 - 220, 640, { size: 60, weight: 600, color: C.dim, a: a1 * q });
    arrow(ctx, W / 2 - 90, 640, W / 2 + 70, 640, { color: C.accent, w: 6, prog: p(t, s('f1') + 8.0, 0.5), a: a1 });
    txt(ctx, 'C++', W / 2 + 180, 640, { size: 60, weight: 700, color: C.accent, a: a1 * p(t, s('f1') + 8.4, 0.4) });

    // f2-f3: one small board as bits, then as a table index
    const a2 = fade(t, s('f2') + 0.2, s('f4') - 0.4);
    if (a2 > 0) {
      const bx = 380, by = 470, f3 = s('f3');
      mini(ctx, bx, by, 420, CELLS, { a: a2, gridProg: p(t, s('f2') + 0.2, 0.8), gw: 5, grid: C.gridHi });
      for (let i = 0; i < 9; i++) txt(ctx, String(i), bx - 210 + (i % 3) * 140 + 22, by - 210 + Math.floor(i / 3) * 140 + 26, { size: 22, color: C.faint, a: a2 });
      // bit rows
      const rx = 860, cw = 84, bitsA = fade(t, s('f2') + 3.6, f3 + 6.5);
      [[1, 'X', C.x, 380], [2, 'O', C.o, 520]].forEach(([who, lab, col, y], r) => {
        const ra = a2 * bitsA * p(t, s('f2') + 4.6 + r * 2.4, 0.5);
        txt(ctx, lab, rx - 60, y, { size: 56, weight: 700, color: col, a: ra });
        for (let i = 0; i < 9; i++) {
          const on = CELLS[i] === who;
          rrect(ctx, rx + i * cw, y - 36, cw - 10, 72, { fill: on ? col : C.panel, stroke: C.grid, a: ra, r: 8, w: 2 });
          txt(ctx, on ? '1' : '0', rx + i * cw + (cw - 10) / 2, y + 2, { size: 40, mono: true, weight: 700, color: on ? C.bg : C.faint, a: ra });
          if (r === 0) txt(ctx, String(i), rx + i * cw + (cw - 10) / 2, y - 70, { size: 22, color: C.faint, a: ra });
        }
      });
      txt(ctx, 'two 9-bit masks per small board', rx + 370, 650, { size: 32, color: C.dim, a: a2 * bitsA * p(t, s('f2') + 7, 0.5) });

      // f3: 3^9 and the lookup table
      const ta = a2 * fade(t, f3 + 0.4);
      const up = p(t, f3 + 6.5, 0.8), yy = lerp(760, 250, up);
      txt(ctx, '3⁹ = 19,683 states', lerp(1230, 1260, up), yy, { size: lerp(64, 50, up), weight: 700, color: C.accent, a: ta * p(t, f3 + 3.0, 0.5) });
      const la = ta * fade(t, f3 + 7.2);
      txt(ctx, 'index = 1·3⁰ + 1·3⁴ + 2·3⁸ = 13,204', 1260, 330, { size: 32, mono: true, color: C.dim, a: la });
      const cols = [['state', 860], ['won?', 1060], ['cells that complete a line', 1310], ['worth', 1600]];
      const rows = [['13,202', '–', 'X: none · O: none', '…'], ['13,203', '–', 'X: none · O: none', '…'], ['13,204', 'no', 'X: none · O: none', '+66'], ['13,205', '–', '…', '…'], ['13,206', '–', '…', '…']];
      cols.forEach(([h, x]) => txt(ctx, h, x, 420, { size: 26, color: C.dim, a: la, weight: 600 }));
      rows.forEach((r, i) => {
        const y = 480 + i * 62, hot = i === 2, ra = la * (hot ? 1 : 0.35) * p(t, f3 + 7.6 + i * 0.1, 0.4);
        if (hot) rrect(ctx, 780, y - 28, 900, 56, { fill: C.panel2, stroke: C.accent, a: la * p(t, f3 + 9.0, 0.5), w: 2, r: 8 });
        r.forEach((v, j) => txt(ctx, hot || j === 0 ? v : '…', cols[j][1], y, { size: 30, mono: true, a: ra, color: hot ? C.fg : C.dim }));
      });
      arrow(ctx, 600, 470, 770, 600, { color: C.accent, w: 5, prog: p(t, f3 + 8.6, 0.6), a: la });
      txt(ctx, 'no computation at search time: just a lookup', 1230, 860, { size: 36, color: C.good, a: la * p(t, f3 + 18.5, 0.5) });
    }

    // f4: the long pause
    const a4 = fade(t, s('f4') + 0.2), f4 = s('f4');
    if (a4 > 0) {
      const y = 560, x0 = 260, x1 = 1660;
      line(ctx, x0, y, lerp(x0, 700, p(t, f4 + 0.3, 0.8)), y, { color: C.fg, w: 5, a: a4 });
      line(ctx, 700, y, x1, y, { color: C.faint, w: 5, dash: [4, 22], prog: p(t, f4 + 5.0, 2.2, ease.lin), a: a4 });
      [['Feb 2024', 'C++ port', x0, 0.3, C.fg], ['Mar 2024', 'Legend league', 700, 1.2, C.accent], ['Aug 2026', '', x1, 7.0, C.fg]].forEach(([d, lab, x, dt, col]) => {
        const q = p(t, f4 + dt, 0.4);
        circle(ctx, x, y, 14, { fill: col, a: a4 * q });
        txt(ctx, d, x, y + 60, { size: 30, color: C.dim, a: a4 * q });
        txt(ctx, lab, x, y - 60, { size: 44, weight: 600, color: col, a: a4 * q });
      });
      txt(ctx, 'nothing happens for 29 months', (700 + x1) / 2, y - 60, { size: 32, color: C.faint, italic: true, a: a4 * p(t, f4 + 6.0, 0.6) });
    }
  },
};
