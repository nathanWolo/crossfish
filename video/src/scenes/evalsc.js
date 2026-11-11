import { W, H, C, p, fade, ease, clamp, lerp, txt, line, circle, rrect, heading, titleCard, arrow, stat, hbar, uttt, replay } from '../lib.js';
import { GAMES } from '../games.js';

export const title = { id: 't_eval', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 5', 'Evaluation', 'what is a position worth?') };

// The handcrafted evaluation's real weights (crossfish_dev.hpp), in eval units.
const WEIGHTS = [['small boards won', 2410], ['two won boards in a line', 1316], ['centre board', 836], ['two in a row inside a board', 534], ['corner boards', 464], ['side to move', 112], ['centre cell', 33], ['corner cell', 10]];
const POS = replay(GAMES.new_vs_legend.moves.slice(0, 30), 1e9, () => 0);

export default {
  id: 'evalsc', title: 'Chapter 5: Evaluation',
  segs: [
    { id: 'e1', text: 'So far I have skipped the hardest part. At the bottom of the tree, the search needs a number: how good is this position? The function that answers is called the evaluation.' },
    { id: 'e2', text: 'The first version was written by hand. Count the small boards each side has won. Add a bonus for the centre board, for two in a row, and a few more. About ten weights, tuned on games the engine played against itself.' },
    { id: 'e3', text: 'The obvious upgrade is a neural network, and the first attempts taught me the central trade-off of this whole field. One larger network was clearly a better judge of positions: forty-eight Elo stronger when both engines searched to the same depth. On the clock, in real games, it lost by nearly three hundred. It was twelve times slower.', say: 'The obvious upgrade is a neural network, and the first attempts taught me the central trade-off of this whole field. One larger network was clearly a better judge of positions: forty-eight E-low stronger when both engines searched to the same depth. On the clock, in real games, it lost by nearly three hundred. It was twelve times slower.' },
    { id: 'e4', text: 'Every microsecond spent judging one position is a microsecond not spent searching others. A better evaluation that you cannot afford is a worse engine.' },
    { id: 'e5', text: 'So the first network that shipped was tiny. It only nudged the handwritten score, and it gained seven Elo. Progress. But a much bigger gain was hiding somewhere else entirely.', say: 'So the first network that shipped was tiny. It only nudged the handwritten score, and it gained seven E-low. Progress. But a much bigger gain was hiding somewhere else entirely.', pause: 1.0 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 5 · Evaluation', fade(t, 0.3));
    // e1-e2: a position and the number we want for it
    const a1 = fade(t, 0.3, s('e3') - 0.4), sh = p(t, s('e2') - 0.2, 0.8);
    uttt(ctx, lerp(620, 400, sh), 560, lerp(640, 560, sh), POS, { a: a1 });
    const qa = a1 * fade(t, s('e1') + 4.5, s('e2') - 0.3);
    arrow(ctx, 980, 560, 1200, 560, { color: C.accent, w: 6, prog: p(t, s('e1') + 4.5, 0.5), a: qa });
    txt(ctx, '?', 1380, 550, { size: 220, weight: 800, color: C.accent, a: qa });
    txt(ctx, 'evaluation: position → number', 1300, 760, { size: 36, color: C.dim, a: qa * p(t, s('e1') + 9.5, 0.5) });
    const a2 = a1 * fade(t, s('e2') + 0.3);
    if (a2 > 0) {
      txt(ctx, 'Handcrafted evaluation', 1300, 200, { size: 42, weight: 600, a: a2 });
      WEIGHTS.forEach(([lab, v], i) => {
        const q = p(t, s('e2') + 2.4 + i * 0.75, 0.6, ease.out);
        hbar(ctx, lab, String(v), 1240, 300 + i * 78, 480, v / 2410, { a: a2 * clamp(q * 3), prog: q, h: 34, size: 28, color: i < 5 ? C.accent : C.dim });
      });
      txt(ctx, 'weights from the current source, fitted on self-play games', 1300, 950, { size: 24, color: C.faint, a: a2 });
    }

    // e3: equal depth vs on the clock
    const e3 = s('e3'), a3 = fade(t, e3 + 0.2, s('e4') - 0.4);
    if (a3 > 0) {
      const ia = a3 * fade(t, e3 + 0.3, e3 + 9.0);
      txt(ctx, 'The obvious upgrade:', W / 2, 420, { size: 44, color: C.dim, a: ia });
      txt(ctx, 'a neural network', W / 2, 520, { size: 96, weight: 700, color: C.violet, a: ia * p(t, e3 + 1.2, 0.5) });
      txt(ctx, 'quality of judgement  vs  speed of judgement', W / 2, 680, { size: 40, a: ia * p(t, e3 + 5.0, 0.5) });
      ctx.save(); ctx.globalAlpha *= fade(t, e3 + 9.4);
      txt(ctx, 'A bigger network (32-wide embeddings, 128 hidden units)', W / 2, 220, { size: 38, color: C.dim, a: a3 });
      const y0 = 620, sc = 0.9, col = (x, lab, v, c, t0) => {
        const q = p(t, t0, 0.8, ease.out), h = v * sc * q;
        rrect(ctx, x - 130, h > 0 ? y0 - h : y0, 260, Math.abs(h), { fill: c, a: a3, r: 8 });
        txt(ctx, (v > 0 ? '+' : '−') + Math.abs(v), x, h > 0 ? y0 - h - 50 : y0 - h + 55, { size: 72, weight: 800, color: c, a: a3 * clamp(q * 2) });
        txt(ctx, lab, x, v > 0 ? y0 + 50 : y0 - 50, { size: 34, a: a3 * clamp(q * 2) });
      };
      line(ctx, 420, y0, 1500, y0, { color: C.gridHi, w: 3, a: a3 });
      col(680, 'same search depth', 48, C.good, e3 + 11.0); col(1240, 'same time per move', -288, C.bad, e3 + 15.5);
      txt(ctx, '12× slower', 1240, 340, { size: 54, weight: 700, color: C.accent, a: a3 * p(t, e3 + 19.0, 0.5) });
      txt(ctx, 'Elo', 400, y0, { size: 28, color: C.dim, align: 'right', a: a3 });
      ctx.restore();
    }
    const a4 = fade(t, s('e4') + 0.2, s('e5') - 0.4);
    txt(ctx, 'A better evaluation', W / 2, 440, { size: 76, weight: 600, a: a4 * p(t, s('e4') + 4.4, 0.5) });
    txt(ctx, 'that you cannot afford', W / 2, 550, { size: 76, weight: 600, a: a4 * p(t, s('e4') + 5.4, 0.5) });
    txt(ctx, 'is a worse engine.', W / 2, 660, { size: 76, weight: 700, color: C.bad, a: a4 * p(t, s('e4') + 6.6, 0.5) });
    txt(ctx, 'time judging ↔ time searching', W / 2, 300, { size: 36, color: C.dim, a: a4 });

    const e5 = s('e5'), a5 = fade(t, e5 + 0.2);
    txt(ctx, 'MiniNet · 15 August 2026', W / 2, 300, { size: 38, color: C.dim, a: a5 });
    txt(ctx, 'score = handcrafted + small correction from a tiny network', W / 2, 380, { size: 36, a: a5 * p(t, e5 + 2.0, 0.5) });
    stat(ctx, '+7 Elo', 'at 95 ms per move · ladder rank 82 → 68', W / 2, 560, { a: a5 * p(t, e5 + 5.0, 0.5) });
  },
};
