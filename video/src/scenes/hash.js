import { W, H, C, p, fade, ease, clamp, lerp, txt, line, circle, rrect, heading, titleCard, mini, arrow, code, stat, rnd, box } from '../lib.js';

export const title = { id: 't_hash', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 3', 'The autopsy', 'August 2026') };

const hex = n => Array.from({ length: 16 }, (_, i) => '0123456789abcdef'[Math.floor(rnd(n * 31 + i) * 16)]).join('');
// The constructor of the March 2024 Legend bot (cpp_impl/cg_legend_hce.cpp), trimmed.
const BUG = [
  ['struct GlobalBoard {', C.dim],
  ['  uint64_t zobrist_hash = 0;', C.fg],
  ['  array<...> move_hashes;        // the keys make_move() uses', C.fg],
  ['', C.fg],
  ['  GlobalBoard() {', C.dim],
  ['    mt19937_64 rng(69420);', C.dim],
  ['    array<...> move_hashes;      // a new local with the same name', C.bad],
  ['    for (p, m, s)', C.dim],
  ['      move_hashes[p][m][s] = dist(rng);', C.bad],
  ['  }                              // ...and the local is thrown away', C.bad],
  ['};', C.dim],
];

export default {
  id: 'hash', title: 'Chapter 3: The autopsy',
  segs: [
    { id: 'h1', text: 'August 2026. I came back to the project, this time working with AI coding agents, and the first job was an autopsy of the old code.', say: 'August 2026. I came back to the project, this time working with A I coding agents, and the first job was an autopsy of the old code.' },
    { id: 'h2', text: 'To explain what it found, I need one more piece of search. Different move orders often reach exactly the same position. So engines keep a transposition table: a big hash table that remembers positions they have already searched, and what the answer was.' },
    { id: 'h3', text: 'The hash is built by giving every possible mark on the board its own random sixty-four bit key, and combining the keys of the marks that are present. Make a move, mix in one more key. It costs almost nothing.' },
    { id: 'h4', text: 'In my engine, those keys were filled in by the constructor. Into local variables, which happened to have the same names as the real ones. The real keys were never set. So every position in the game had the same hash: zero.' },
    { id: 'h5', text: 'Every position read and wrote the same single slot. Back in 2024 I had even measured that switching the table\'s cutoffs off made the engine stronger, and concluded they were a bad idea. They were bad because the table was full of garbage. The engine had reached the top league with its memory completely broken.' },
    { id: 'h6', text: 'Fixing it was worth about a hundred and twenty Elo. One of the biggest gains in the whole project was a bug.', say: 'Fixing it was worth about a hundred and twenty E-low. One of the biggest gains in the whole project was a bug.', pause: 1.2 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 3 · The autopsy', fade(t, 0.3));
    const a1 = fade(t, 0.3, s('h2') - 0.4);
    txt(ctx, 'August 2026', W / 2, 440, { size: 90, weight: 700, a: a1 });
    txt(ctx, 'reading the 2024 engine, line by line', W / 2, 560, { size: 40, color: C.dim, a: a1 * p(t, s('h1') + 5.5, 0.5) });

    // h2: two move orders, one position, one table entry
    const h2 = s('h2'), a2 = fade(t, h2 + 0.3, s('h3') - 0.4);
    if (a2 > 0) {
      const A = [0, 0, 0, 0, 1, 0, 0, 0, 0], B = [0, 0, 1, 0, 0, 0, 0, 0, 0], AB = [0, 0, 1, 0, 1, 0, 0, 0, 0], sz = 150;
      const P = { root: [300, 500], up: [620, 330], dn: [620, 670], end: [940, 500] };
      const m = (pos, cells, t0) => { const q = p(t, t0, 0.4); rrect(ctx, pos[0] - sz / 2 - 14, pos[1] - sz / 2 - 14, sz + 28, sz + 28, { fill: C.panel, stroke: C.grid, a: a2 * q }); mini(ctx, pos[0], pos[1], sz, cells, { a: a2 * q }); };
      const e = (a, b, t0) => arrow(ctx, a[0] + 95, a[1] + (b[1] - a[1]) * 0.25, b[0] - 100, b[1] - (b[1] - a[1]) * 0.25, { color: C.gridHi, w: 4, prog: p(t, t0, 0.4), a: a2 });
      m(P.root, Array(9).fill(0), h2 + 0.5);
      e(P.root, P.up, h2 + 4.0); m(P.up, A, h2 + 4.3); e(P.root, P.dn, h2 + 4.6); m(P.dn, B, h2 + 4.9);
      e(P.up, P.end, h2 + 5.6); e(P.dn, P.end, h2 + 5.9); m(P.end, AB, h2 + 6.2);
      txt(ctx, 'same position', P.end[0], P.end[1] + 125, { size: 28, color: C.accent, a: a2 * p(t, h2 + 6.6, 0.4) });
      // the table
      const tx = 1200, ty = 300, ta = a2 * p(t, h2 + 9.0, 0.5);
      txt(ctx, 'transposition table', tx + 280, ty - 50, { size: 34, weight: 600, a: ta });
      ['hash', 'depth', 'score', 'best move'].forEach((h, j) => txt(ctx, h, tx + [90, 250, 370, 500][j], ty + 10, { size: 24, color: C.dim, a: ta }));
      for (let i = 0; i < 7; i++) {
        const y = ty + 60 + i * 56, hot = i === 3, q = hot ? p(t, h2 + 12.5, 0.5) : 1;
        rrect(ctx, tx, y - 24, 580, 48, { fill: hot ? C.panel2 : C.panel, stroke: hot ? C.accent : C.grid, a: ta * (hot ? 1 : 0.5), w: 2, r: 6 });
        if (!hot || q > 0) [hex(i + 3).slice(0, 8) + '…', hot ? '12' : String(4 + Math.floor(rnd(i) * 9)), hot ? '+310' : (rnd(i + 9) > 0.5 ? '+' : '−') + Math.floor(rnd(i + 5) * 900), hot ? 'centre' : '…']
          .forEach((v, j) => txt(ctx, v, tx + [90, 250, 370, 500][j], y, { size: 24, mono: true, a: ta * (hot ? q : 0.45) }));
      }
      arrow(ctx, P.end[0] + 105, P.end[1], tx - 16, ty + 60 + 3 * 56, { color: C.accent, w: 5, prog: p(t, h2 + 11.8, 0.6), a: a2 });
      txt(ctx, 'searched once, remembered for every other path', W / 2, 900, { size: 36, color: C.good, a: a2 * p(t, h2 + 14.0, 0.5) });
    }

    // h3: Zobrist keys
    const h3 = s('h3'), a3 = fade(t, h3 + 0.3, s('h4') - 0.4);
    if (a3 > 0) {
      txt(ctx, 'Zobrist hashing', W / 2, 220, { size: 48, weight: 600, a: a3 });
      const rows = [['X on board 4, cell 4', C.x, 3.0], ['O on board 4, cell 2', C.o, 4.0], ['X on board 2, cell 6', C.x, 5.0]];
      rows.forEach(([lab, col, dt], i) => {
        const q = a3 * p(t, h3 + dt, 0.4), y = 360 + i * 90;
        txt(ctx, lab, 760, y, { size: 34, align: 'right', color: col, a: q });
        txt(ctx, (i ? '⊕ ' : '  ') + hex(i + 40), 820, y, { size: 38, mono: true, align: 'left', a: q });
      });
      line(ctx, 820, 590, 1330, 590, { color: C.grid, w: 3, a: a3 * p(t, h3 + 7.5, 0.4) });
      txt(ctx, 'position hash', 760, 650, { size: 34, align: 'right', color: C.accent, a: a3 * p(t, h3 + 7.8, 0.4) });
      txt(ctx, '= ' + hex(77), 820, 650, { size: 38, mono: true, align: 'left', color: C.accent, a: a3 * p(t, h3 + 7.8, 0.4) });
      txt(ctx, 'one XOR per move', W / 2, 820, { size: 36, color: C.dim, a: a3 * p(t, h3 + 10.5, 0.5) });
    }

    // h4: the bug, from the real fossil
    const h4 = s('h4'), a4 = fade(t, h4 + 0.2, s('h5') - 0.3);
    if (a4 > 0) {
      const hi = p(t, h4 + 4.6, 0.5);
      code(ctx, BUG, 260, 210, { w: 1400, size: 32, a: a4, lineA: i => (BUG[i][1] === C.bad ? lerp(0.9, 1, hi) : lerp(0.9, 0.45, hi)) });
      txt(ctx, 'cg_legend_hce.cpp · the bot that reached Legend in March 2024', 960, 800, { size: 26, color: C.faint, a: a4 });
      txt(ctx, 'hash(every position) = 0', W / 2, 920, { size: 60, weight: 700, mono: true, color: C.bad, a: a4 * p(t, s('h4') + 10.5, 0.5) });
    }

    // h5: everything lands on slot 0
    const h5 = s('h5'), a5 = fade(t, h5 + 0.2, s('h6') - 0.3);
    if (a5 > 0) {
      const tx = 1180, ty = 250;
      for (let i = 0; i < 9; i++) rrect(ctx, tx, ty + i * 66, 460, 52, { fill: i ? C.panel : C.panel2, stroke: i ? C.grid : C.bad, a: a5 * (i ? 0.4 : 1), w: i ? 2 : 4, r: 6 });
      txt(ctx, 'slot 0', tx + 230, ty + 26, { size: 28, mono: true, color: C.bad, a: a5 });
      txt(ctx, 'empty, forever', tx + 230, ty + 26 + 66 * 4, { size: 28, color: C.faint, italic: true, a: a5 });
      for (let i = 0; i < 6; i++) {
        const y = 300 + i * 100, q = p(t, h5 + 0.4 + i * 0.25, 0.5);
        mini(ctx, 420, y, 76, Array.from({ length: 9 }, (_, k) => Math.floor(rnd(i * 13 + k) * 3)), { a: a5 * q });
        arrow(ctx, 480, y, tx - 14, ty + 26, { color: C.bad, w: 3, prog: p(t, h5 + 1.0 + i * 0.25, 0.6), a: a5 * 0.7, head: 14 });
      }
      const qa = a5 * fade(t, h5 + 6.0);
      rrect(ctx, 250, 880, 1420, 110, { fill: C.panel, stroke: C.grid, a: qa });
      txt(ctx, 'February 2024 commit:  “Remove TT cutoffs”   +37 Elo', W / 2, 935, { size: 38, mono: true, a: qa });
    }

    const a6 = fade(t, s('h6') + 0.2);
    stat(ctx, '+121 Elo', 'filling in the hash keys · 12 August 2026 · 569 games', W / 2, 470, { a: a6 });
  },
};
