import { W, H, C, p, fade, ease, clamp, lerp, txt, line, circle, rrect, heading, titleCard, arrow, stat, mini, uttt, replay, vec, rnd, box, boardXY } from '../lib.js';
import { GAMES } from '../games.js';

export const title = { id: 't_nnue', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 7', 'The network', '27 September 2026') };

const pat = seed => Array.from({ length: 9 }, (_, k) => { const r = rnd(seed * 17 + k); return r < 0.5 ? 0 : r < 0.75 ? 1 : 2; });
const G = GAMES.selfplay.moves, PLY = 24, MOVE = G[PLY];
const row = (seed, i) => rnd(seed * 97 + i) * 2 - 1;
const EX = [1, 0, 0, 0, 1, 0, 0, 0, 2];

export default {
  id: 'nnue', title: 'Chapter 7: The network',
  segs: [
    { id: 'n1', text: 'By late September, the handwritten evaluation and its helper networks had been pushed about as far as they would go. The replacement was built around one fact from earlier: a small board only has nineteen thousand patterns.' },
    { id: 'n2', text: 'Here is the design. Take one small board, sitting in one of the nine places. A small network, the encoder, reads its nine cells and outputs sixty-four numbers that describe that pattern in that place.' },
    { id: 'n3', text: 'Now the trick. There are only nineteen thousand patterns and nine places. So when the program starts, it runs the encoder on every pattern in every place, and stores the answers in a table. That takes a fraction of a second, once. After that the encoder never runs again.' },
    { id: 'n4', text: 'During search, the engine keeps a running total: the sum of the nine table rows for the nine small boards. When a move is made, only one small board changes. So subtract its old row, add its new row, and the most expensive layer of the network is already up to date.' },
    { id: 'n5', text: 'What is left is a tiny network on top, which turns that total into a score. Chess engines call this idea N N U E. This version is shaped around this particular game: it sees each small board as a whole pattern, never as nine separate cells.' },
    { id: 'n6', text: 'The current network has fifty-one thousand parameters. It was trained on about eighty million positions from the engine\'s own games, each one labelled with the verdict of a deeper search.' },
    { id: 'n7', text: 'The first version made the engine slower. It lost about half its speed, and looked one move less deep. It was also two hundred and seventy Elo stronger. That is the largest single step in the project.', say: 'The first version made the engine slower. It lost about half its speed, and looked one move less deep. It was also two hundred and seventy E-low stronger. That is the largest single step in the project.' },
    { id: 'n8', text: 'After that submission, Crossfish finished its placement games at number one on the ladder.', pause: 1.5 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 7 · The network', fade(t, 0.3));

    // n1: a wall of patterns
    const n1 = s('n1'), a1 = fade(t, 0.3, s('n2') - 0.4);
    if (a1 > 0) {
      for (let r = 0; r < 5; r++) for (let c = 0; c < 14; c++) {
        const i = r * 14 + c, q = p(t, n1 + 2.0 + rnd(i) * 7.0, 0.5);
        mini(ctx, 220 + c * 114, 260 + r * 118, 84, pat(i + 1), { a: a1 * q });
      }
      txt(ctx, '3⁹ = 19,683 patterns', W / 2, 900, { size: 60, weight: 700, color: C.accent, a: a1 * p(t, n1 + 11.0, 0.5) });
    }

    // n2-n3: encoder, then the table
    const n2 = s('n2'), n3 = s('n3'), a2 = fade(t, n2 + 0.2, s('n4') - 0.4);
    if (a2 > 0) {
      const up = p(t, n3 + 4.5, 0.9), y = lerp(480, 250, up);
      mini(ctx, 250, y, lerp(240, 150, up), EX, { a: a2 * p(t, n2 + 1.6, 0.5), gw: 4, grid: C.gridHi });
      // which of the nine places
      const ia = a2 * p(t, n2 + 3.4, 0.5), ix = 250, iy = y + lerp(200, 125, up);
      for (let k = 0; k < 9; k++) rrect(ctx, ix - 33 + (k % 3) * 22, iy - 33 + Math.floor(k / 3) * 22, 19, 19, { fill: k === 0 ? C.accent : C.grid, a: ia, r: 3 });
      txt(ctx, 'place', ix + 80, iy, { size: 24, color: C.dim, a: ia });
      // encoder
      const ea = a2 * p(t, n2 + 5.2, 0.5), dead = p(t, n3 + 14.5, 0.8), ex = 470, ew = 400, eh = lerp(300, 170, up);
      arrow(ctx, 400, y, ex - 14, y, { color: C.gridHi, w: 4, a: ea });
      rrect(ctx, ex, y - eh / 2, ew, eh, { fill: C.panel, stroke: lerp(1, 0, dead) > 0.5 ? C.violet : C.grid, a: ea * lerp(1, 0.35, dead), w: 3 });
      txt(ctx, 'encoder', ex + ew / 2, y - eh / 2 + 34, { size: 34, weight: 700, color: C.violet, a: ea * lerp(1, 0.35, dead) });
      [27, 128, 128, 32].forEach((n, i) => {
        const lx = ex + 62 + i * 92, lh = [70, 150, 150, 90][i] * lerp(1, 0.5, up);
        rrect(ctx, lx - 16, y + 20 - lh / 2, 32, lh, { fill: C.violet, a: ea * 0.7 * lerp(1, 0.35, dead), r: 6 });
        txt(ctx, String(n), lx, y + 20 + lh / 2 + 22, { size: 22, color: C.dim, mono: true, a: ea * (1 - up) });
        if (i) line(ctx, lx - 76 + 16, y + 20, lx - 16, y + 20, { color: C.grid, w: 2, a: ea * lerp(1, 0.35, dead) });
      });
      txt(ctx, 'runs once, at start-up', ex + ew / 2, y + eh / 2 + 34, { size: 28, color: C.bad, a: a2 * dead });
      // its output
      const va = a2 * p(t, n2 + 8.5, 0.5), vx = 960;
      arrow(ctx, ex + ew + 10, y, vx - 16, y, { color: C.gridHi, w: 4, a: va });
      vec(ctx, vx, y - 20, 64, 13, 40, 7, { a: va });
      txt(ctx, '64 numbers', vx + 416, y + 50, { size: 28, color: C.dim, a: va * (1 - up) });

      // n3: table rows
      const ta = a2 * fade(t, n3 + 5.0);
      if (ta > 0) {
        txt(ctx, 'the table: 9 places × 19,683 patterns', 1100, 400, { size: 32, color: C.dim, a: ta, weight: 600 });
        const shown = Math.floor(clamp((t - n3 - 5.4) / 4.5) * 10);
        for (let i = 0; i < shown; i++) {
          const yy = 460 + i * 50;
          mini(ctx, 740, yy, 38, pat(i + 60), { a: ta, gw: 1.5 });
          for (let k = 0; k < 9; k++) rrect(ctx, 790 + (k % 3) * 10, yy - 15 + Math.floor(k / 3) * 10, 8, 8, { fill: k === i % 9 ? C.accent : C.grid, a: ta, r: 1 });
          vec(ctx, 850, yy - 16, 64, 13, 32, 200 + i, { a: ta });
        }
        txt(ctx, '⋮', 1260, 975, { size: 40, color: C.dim, a: ta * (shown >= 10 ? 1 : 0) });
        const cnt = Math.floor(clamp((t - n3 - 5.4) / 6.0) * 177147);
        txt(ctx, cnt.toLocaleString('en-US') + ' rows', 300, 620, { size: 54, mono: true, weight: 700, color: C.accent, a: ta });
        txt(ctx, 'filled in about a quarter of a second', 300, 690, { size: 26, color: C.dim, a: ta * p(t, n3 + 11.5, 0.5) });
      }
    }

    // n4-n5: the accumulator
    const n4 = s('n4'), n5 = s('n5'), a4 = fade(t, n4 + 0.2, s('n6') - 0.4);
    if (a4 > 0) {
      const tm = n4 + 9.8;                                     // the move lands
      const st = replay(G.slice(0, PLY + 1), t, i => (i < PLY ? -10 : tm));
      st.active = [{ b: MOVE.board, a: fade(t, tm - 1.2, n5 - 0.5) }];
      const shrink = p(t, n5 - 0.2, 0.8);
      uttt(ctx, 330, 500, 500, st, { a: a4 });
      const rx = 720, cw = 12, ry0 = 250, rh = 46, chg = p(t, tm + 2.6, 1.0);
      txt(ctx, 'one table row per small board', rx, 200, { size: 28, color: C.dim, align: 'left', a: a4 * p(t, n4 + 2.0, 0.5) });
      const vals = b => Array.from({ length: 64 }, (_, i) => (b === MOVE.board ? lerp(row(300 + b, i), row(400 + b, i), chg) : row(300 + b, i)));
      for (let b = 0; b < 9; b++) {
        const q = a4 * p(t, n4 + 2.4 + b * 0.25, 0.4), y = ry0 + b * rh, hot = b === MOVE.board && t > tm;
        for (let k = 0; k < 9; k++) rrect(ctx, rx - 50 + (k % 3) * 11, y + 2 + Math.floor(k / 3) * 11, 9, 9, { fill: k === b ? C.accent : C.grid, a: q, r: 2 });
        vec(ctx, rx, y, 64, cw, rh - 10, 0, { a: q, values: vals(b) });
        if (hot) rrect(ctx, rx - 6, y - 5, 64 * cw + 10, rh, { stroke: C.accent, a: a4 * fade(t, tm, n5 - 0.3), w: 3, r: 6 });
      }
      const ha = a4 * fade(t, tm + 1.0, n5 - 0.3), hy = ry0 + MOVE.board * rh + 18;
      txt(ctx, '− old row', rx + 64 * cw + 30, hy - 16, { size: 26, color: C.bad, align: 'left', a: ha * p(t, tm + 1.6, 0.4), weight: 700 });
      txt(ctx, '+ new row', rx + 64 * cw + 30, hy + 16, { size: 26, color: C.good, align: 'left', a: ha * p(t, tm + 3.0, 0.4), weight: 700 });
      // the running total
      const sy = ry0 + 9 * rh + 40, sa = a4 * p(t, n4 + 5.0, 0.5);
      line(ctx, rx, sy - 22, rx + 64 * cw, sy - 22, { color: C.gridHi, w: 3, a: sa });
      txt(ctx, 'Σ', rx - 34, sy + 22, { size: 44, color: C.accent, a: sa });
      const acc = Array.from({ length: 64 }, (_, i) => { let v = 0; for (let b = 0; b < 9; b++) v += vals(b)[i]; return clamp(v / 2.2, -1, 1); });
      vec(ctx, rx, sy, 64, cw, 50, 0, { a: sa, values: acc });
      txt(ctx, 'running total (the “accumulator”)', rx, sy + 80, { size: 28, color: C.accent, align: 'left', a: sa });
      // n5: the head
      const ka = a4 * fade(t, n5 + 0.3), hx = rx + 64 * cw + 70, hy2 = sy + 25;
      if (ka > 0) {
        arrow(ctx, hx - 50, hy2, hx - 6, hy2, { color: C.gridHi, w: 4, a: ka });
        [[16, 70], [32, 110]].forEach(([n, h], i) => {
          rrect(ctx, hx + i * 70, hy2 - h / 2, 30, h, { fill: C.violet, a: ka * p(t, n5 + 0.6 + i * 0.4, 0.4), r: 6 });
          txt(ctx, String(n), hx + i * 70 + 15, hy2 + h / 2 + 22, { size: 22, mono: true, color: C.dim, a: ka });
        });
        arrow(ctx, hx + 112, hy2, hx + 150, hy2, { color: C.gridHi, w: 4, a: ka * p(t, n5 + 1.6, 0.4) });
        txt(ctx, 'score', hx + 250, hy2 - 30, { size: 24, color: C.dim, a: ka * p(t, n5 + 2.0, 0.4) });
        txt(ctx, '+1,866', hx + 250, hy2 + 10, { size: 40, mono: true, weight: 700, color: C.x, a: ka * p(t, n5 + 2.0, 0.4) });
        txt(ctx, 'NNUE: efficiently updatable neural network', W / 2, 960, { size: 34, color: C.dim, a: ka * fade(t, n5 + 5.5) });
      }
    }

    const n6 = s('n6'), a6 = fade(t, n6 + 0.2, s('n7') - 0.4);
    stat(ctx, '51,435', 'parameters', 520, 430, { a: a6, color: C.accent, size: 130 });
    stat(ctx, '≈ 80 million', 'self-play positions, each labelled by a depth-13 search', 1280, 430, { a: a6 * p(t, n6 + 3.2, 0.5), color: C.fg, size: 110, small: 30 });
    txt(ctx, '27-128-128-32 encoder → 64-wide accumulators → 16 → 32 → 1', W / 2, 760, { size: 32, mono: true, color: C.dim, a: a6 * p(t, n6 + 1.0, 0.5) });

    // n7: slower, shallower, far stronger
    const n7 = s('n7'), a7 = fade(t, n7 + 0.2, s('n8') - 0.4);
    if (a7 > 0) {
      txt(ctx, 'First network vs the handcrafted hybrid', W / 2, 220, { size: 40, weight: 600, a: a7 });
      [['positions per second', '31.2 M', '16.0 M', C.bad, 2.2], ['average depth reached', '17.4', '16.4', C.bad, 4.6], ['strength', '', '+267 Elo', C.good, 8.2]].forEach(([lab, a, b, col, dt], i) => {
        const q = a7 * p(t, n7 + dt, 0.5), y = 380 + i * 170;
        txt(ctx, lab, 700, y, { size: 40, align: 'right', color: C.dim, a: q });
        txt(ctx, a, 900, y, { size: 56, mono: true, a: q, color: C.dim });
        if (a) arrow(ctx, 1020, y, 1120, y, { color: C.gridHi, w: 4, a: q });
        txt(ctx, b, a ? 1280 : 1100, y, { size: i === 2 ? 110 : 56, mono: i !== 2, weight: i === 2 ? 800 : 500, a: q, color: col });
      });
      txt(ctx, '3,000 games at 90 ms per move · ± 12', W / 2, 900, { size: 28, color: C.dim, a: a7 * p(t, n7 + 9.0, 0.5) });
    }
    const a8 = fade(t, s('n8') + 0.2);
    stat(ctx, '#1', 'CodinGame Ultimate Tic-Tac-Toe ladder · submitted 27 September 2026', W / 2, 470, { a: a8, color: C.accent, size: 260, small: 34 });
  },
};
