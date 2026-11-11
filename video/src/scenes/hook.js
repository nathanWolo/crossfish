import { W, H, C, p, fade, ease, clamp, lerp, txt, uttt, replay, rrect, line } from '../lib.js';
import { GAMES } from '../games.js';

const G = GAMES.selfplay.moves.slice(0, 36);   // the opening and middlegame, before either side sees a forced win
const fmt = n => n.toLocaleString('en-US');

export default {
  id: 'hook', title: 'Cold open',
  segs: [
    { id: 'a', text: 'This is a game of tic-tac-toe, played on a board made of nine smaller tic-tac-toe boards.', pre: 1.2 },
    { id: 'b', text: 'Both sides here are Crossfish, a program I wrote to play it. It gets a tenth of a second per move. In that time it examines around seven hundred thousand positions, and reads the game fifteen to twenty moves ahead.' },
    { id: 'c', text: 'It started as a two-week Python project for a university tournament. Two years later, it reached number one on a worldwide ladder.', say: 'It started as a two-week Python project for a university tournament. Two years later, it reached number one on a worldwide ladder.' },
    { id: 'd', text: 'This is how it works. And it is also the story of a hash table that never worked, a missing compiler flag, and a neural network small enough to fit in a text box.', pause: 2.2 },
  ],
  draw(ctx, T) {
    const t = T.t, step = (T.s('c') - 0.6) / G.length, when = i => 0.6 + i * step;
    const st = replay(G, t, when);
        const cx = lerp(960, 640, p(t, T.s('b') - 0.2, 0.9)), size = 800;
    if (st.next >= 0 && st.n < G.length) st.active = [{ b: st.next, a: 0.8 }];
    uttt(ctx, cx, 560, size, st, { a: fade(t, 0.2) * (1 - p(t, T.s('c') - 0.5, 0.6)) });

    // Live read-out of what the engine printed for each move.
    const ha = fade(t, T.s('b') + 0.5, T.s('c') - 0.5);
    if (ha > 0) {
      const last = k => { for (let i = st.n - 1; i >= 0; i--) if (G[i].side === k && G[i].D) return G[i]; return null; };
      const x0 = 1180;
      txt(ctx, 'CROSSFISH vs CROSSFISH', x0, 250, { size: 26, color: C.dim, align: 'left', weight: 600, a: ha });
      [0, 1].forEach(k => {
        const m = last(k), y = 360 + k * 250, col = k ? C.o : C.x;
        txt(ctx, k ? 'O' : 'X', x0, y, { size: 64, weight: 700, color: col, align: 'left', a: ha });
        if (!m) return;
        const mate = Math.abs(m.E) > 90000;
        txt(ctx, 'depth', x0 + 90, y - 22, { size: 24, color: C.dim, align: 'left', a: ha });
        txt(ctx, mate ? (m.E > 0 ? 'forced win found' : 'forced loss seen') : String(m.D), x0 + 90, y + 22, { size: mate ? 34 : 46, mono: true, align: 'left', a: ha, color: mate ? (m.E > 0 ? C.accent : C.dim) : C.fg });
        if (!mate) {
          txt(ctx, 'positions searched', x0 + 260, y - 22, { size: 24, color: C.dim, align: 'left', a: ha });
          txt(ctx, fmt(m.N), x0 + 260, y + 22, { size: 46, mono: true, align: 'left', a: ha });
        }
      });
      txt(ctx, 'move ' + st.n + ' · 90 ms per move', x0, 800, { size: 28, color: C.dim, align: 'left', a: ha });
    }

    // Two-line origin story.
    const ca = fade(t, T.s('c') + 0.2, T.s('d') - 0.2);
    txt(ctx, 'January 2024', W / 2, 400, { size: 34, color: C.dim, a: ca });
    txt(ctx, 'a two-week Python project', W / 2, 470, { size: 64, weight: 600, a: ca });
    const c2 = ca * p(t, T.s('c') + 4.2, 0.5);
    txt(ctx, 'September 2026', W / 2, 620, { size: 34, color: C.dim, a: c2 });
    txt(ctx, '#1 on the CodinGame ladder', W / 2, 690, { size: 64, weight: 600, color: C.accent, a: c2 });

    // Teasers, then the title.
    const items = [['a hash table that never worked', 2.6], ['a missing compiler flag', 5.2], ['a neural network that fits in a text box', 7.0]];
    items.forEach(([s, dt], i) => {
      const a = fade(t, T.s('d') + dt, T.e('d') - 1.0, 0.5);
      txt(ctx, s, W / 2, 420 + i * 110, { size: 54, weight: 500, a, color: [C.bad, C.accent, C.good][i] });
    });
    const ta = fade(t, T.e('d') - 0.3);
    txt(ctx, 'CROSSFISH', W / 2, 500, { size: 150, weight: 800, a: ta });
    line(ctx, W / 2 - 300, 600, W / 2 + 300, 600, { color: C.accent, w: 4, prog: p(t, T.e('d'), 0.9) });
    txt(ctx, 'the story of an Ultimate Tic-Tac-Toe engine', W / 2, 660, { size: 40, color: C.dim, a: ta * p(t, T.e('d') + 0.4, 0.6) });
  },
};
