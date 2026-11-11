import { W, H, C, p, fade, ease, clamp, lerp, txt, line, rrect, heading, titleCard, stat, hbar, code, measure } from '../lib.js';

export const title = { id: 't_flag', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 6', 'The missing flag', '24 September 2026') };

export default {
  id: 'flag', title: 'Chapter 6: The missing flag',
  segs: [
    { id: 'g1', text: 'All of my testing ran on my own machines, compiled with full optimization. CodinGame compiles your code itself. One day, we checked how.', say: 'All of my testing ran on my own machines, compiled with full optimization. Coding Game compiles your code itself. One day, we checked how.' },
    { id: 'g2', text: 'No optimization flag at all. The bot I had been carefully tuning at seven hundred thousand positions per move was running on the ladder at about a hundred and fifty thousand.' },
    { id: 'g3', text: 'Every result I had ever seen there came from an engine running at a fifth of its speed. The fix was a few lines that force optimization from inside the source file. It was worth a hundred and sixty Elo. And no test on my machine could have found it, because on my machine nothing was wrong.', say: 'Every result I had ever seen there came from an engine running at a fifth of its speed. The fix was a few lines that force optimization from inside the source file. It was worth a hundred and sixty E-low. And no test on my machine could have found it, because on my machine nothing was wrong.', pause: 1.2 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 6 · The missing flag', fade(t, 0.3));
    const g1 = s('g1'), g2 = s('g2'), g3 = s('g3'), a1 = fade(t, 0.3, g3 + 5.0);
    const term = (y, title, cmd, t0, hot) => {
      const a = a1 * p(t, t0, 0.5);
      rrect(ctx, 210, y, 1500, 150, { fill: C.panel, stroke: C.grid, a, w: 2 });
      txt(ctx, title, 240, y + 36, { size: 26, color: C.dim, align: 'left', weight: 600, a });
      txt(ctx, '$ ' + cmd, 240, y + 98, { size: 32, mono: true, align: 'left', a, weight: 400 });
      if (hot) hot(a, y + 98);
    };
    term(210, 'MY MACHINE', 'g++ -O3 -std=c++17 -mavx2 -mbmi2 bot.cpp', g1 + 1.5, (a, y) => {
      const x = 240 + measure(ctx, '$ g++ ', { size: 32, mono: true, weight: 400 }), w = measure(ctx, '-O3', { size: 32, mono: true, weight: 400 });
      rrect(ctx, x - 6, y - 26, w + 12, 52, { stroke: C.good, a: a * p(t, g2 + 0.2, 0.4), w: 3, r: 8 });
    });
    term(400, 'CODINGAME', 'g++ -std=gnu++17 -Werror=return-type -g -pthread bot.cpp -lm …', g1 + 6.0, (a, y) => {
      const x = 240 + measure(ctx, '$ g++ ', { size: 32, mono: true, weight: 400 });
      rrect(ctx, x - 8, y - 26, 16, 52, { fill: C.bad, a: a * p(t, g2 + 0.2, 0.4) * (0.6 + 0.4 * Math.sin(t * 6)), r: 4 });
      txt(ctx, 'no -O', x, y + 62, { size: 26, color: C.bad, a: a * p(t, g2 + 0.4, 0.4), weight: 700 });
    });
    const a2 = a1 * fade(t, g2 + 2.5);
    if (a2 > 0) {
      txt(ctx, 'positions searched per move', W / 2, 660, { size: 30, color: C.dim, a: a2 });
      hbar(ctx, 'local build', '≈ 700,000', 620, 740, 900, 1, { a: a2, prog: p(t, g2 + 4.0, 0.8, ease.out), color: C.good, h: 50, size: 36 });
      hbar(ctx, 'on the ladder', '≈ 147,000', 620, 830, 900, 0.21, { a: a2, prog: p(t, g2 + 8.0, 0.8, ease.out), color: C.bad, h: 50, size: 36 });
    }
    const a3 = fade(t, g3 + 5.6);
    if (a3 > 0) {
      code(ctx, [['#pragma GCC optimize("O3")', C.good], ['#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")', C.good], ['', C.fg], ['// without -O, GCC inlines only functions marked:', C.dim], ['__attribute__((always_inline))', C.good]],
        310, 200, { w: 1300, size: 34, a: a3 });
      stat(ctx, '+163 Elo', 'new file vs old file, both compiled the way CodinGame does · ± 32', W / 2, 660, { a: a3 * p(t, g3 + 10.5, 0.5) });
    }
  },
};
