import { W, H, C, p, fade, ease, clamp, lerp, txt, line, circle, rrect, heading, titleCard, stat, hbar, arrow } from '../lib.js';

// A three-level game tree. Leaf scores are from X's point of view.
const LEAVES = [[3, 6, 5], [1, 8, 4], [7, 2, 9]];
const MINS = LEAVES.map(l => Math.min(...l));           // 3, 1, 2
const ROOT = { x: 960, y: 250 }, KX = [400, 960, 1520], KY = 520, LY = 800, LDX = 150;
const leafX = (k, j) => KX[k] + (j - 1) * LDX;

function node(ctx, x, y, r, label, o = {}) {
  circle(ctx, x, y, r, { fill: C.panel, stroke: o.color ?? C.grid, w: 4, a: o.a });
  if (label !== '' && label !== undefined) txt(ctx, label, x, y + 2, { size: r * 0.9, weight: 600, a: (o.a ?? 1) * (o.la ?? 1), color: o.tc ?? C.fg, mono: true });
}

export const title = { id: 't_search', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 1', 'Search', 'January 2024') };

export default {
  id: 'search', title: 'Chapter 1: Search',
  segs: [
    { id: 's1', text: 'January 2024. The UVic AI club is running an Ultimate Tic-Tac-Toe bot tournament, and I have about two weeks to write an entry.', say: 'January 2024. The U Vic A I club is running an Ultimate Tic-Tac-Toe bot tournament, and I have about two weeks to write an entry.' },
    { id: 's2', text: 'The core idea is older than computers. From the current position, list every move you could make. For each of those, list every reply. Keep going, and you get a tree.' },
    { id: 's3', text: 'At the bottom, give each position a score: high if it is good for X, low if it is good for O. Then work back up. On O\'s turn, take the lowest score below. On X\'s turn, take the highest. This is minimax. It finds the best move, assuming both sides play as well as they can.' },
    { id: 's4', text: 'The problem is size. A typical position has about nine legal moves, so every extra move of lookahead multiplies the work by nine.' },
    { id: 's5', text: 'The fix is called alpha-beta pruning. Say X has already found a move worth three. While checking the next move, O finds a reply worth only one. We can stop right there. X would never choose this branch, so whatever else is under it does not matter.' },
    { id: 's6', text: 'How much you can skip depends on the order you look at moves. Try the best move first, and almost everything else gets cut. So a large part of engine design is guessing well. Try the move that was best the last time you saw this position. Try moves that win a small board. Avoid moves that hand over a free move.' },
    { id: 's7', text: 'In that first week, one change, trying the remembered best move first, was worth more than four hundred Elo by itself.', say: 'In that first week, one change, trying the remembered best move first, was worth more than four hundred E-low by itself.' },
    { id: 's8', text: 'And since the clock is fixed, the engine searches one move deep, then two, then three, each pass using the previous one to order its moves, until time runs out.' },
    { id: 's9', text: 'Two weeks in, that Python bot won the tournament.', pause: 1.2 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 1 · Search', fade(t, 0.3));

    // s1: where it starts
    const a1 = fade(t, 0.3, s('s2') - 0.4);
    txt(ctx, 'January 2024', W / 2, 400, { size: 36, color: C.dim, a: a1 });
    txt(ctx, 'UVicAI Ultimate Tic-Tac-Toe tournament', W / 2, 480, { size: 66, weight: 600, a: a1 });
    txt(ctx, 'two weeks · Python · numpy', W / 2, 590, { size: 40, color: C.accent, a: a1 * p(t, s('s1') + 5.2, 0.5) });

    // The tree, shown twice: plain minimax (s2-s3) and alpha-beta (s5-s6).
    const mm = fade(t, s('s2') + 0.2, s('s4') - 0.3), ab = fade(t, s('s5') - 0.2, s('s6') + 5.0);
    const ta = Math.max(mm, ab), isAB = t > s('s4');
    if (ta > 0) {
      ctx.save(); ctx.globalAlpha *= ta;
      const g0 = isAB ? -10 : s('s2') + 3.0;           // edges grow in
      const e1 = k => p(t, g0 + k * 0.35, 0.6), e2 = (k, j) => p(t, (isAB ? -10 : s('s2') + 6.2) + (k * 3 + j) * 0.14, 0.5);
      // Alpha-beta reveal times for each leaf (Infinity = never visited).
      const s5 = s('s5');
      const tv = isAB
        ? [[s5 + 2.4, s5 + 2.9, s5 + 3.4], [s5 + 6.6, Infinity, Infinity], [s('s6') + 0.3, s('s6') + 1.0, Infinity]]
        : LEAVES.map((l, k) => l.map((_, j) => s('s3') + 1.2 + (k * 3 + j) * 0.2));
      const tk = isAB ? [s5 + 4.2, s5 + 7.6, s('s6') + 1.6] : [0, 1, 2].map(k => s('s3') + 7.4 + k * 0.6);
      const tr = isAB ? s('s6') + 2.4 : s('s3') + 11.0;
      const cutT = isAB ? [Infinity, s5 + 9.0, s('s6') + 1.6] : [Infinity, Infinity, Infinity];
      for (let k = 0; k < 3; k++) {
        line(ctx, ROOT.x, ROOT.y, KX[k], KY, { color: C.grid, w: 4, prog: e1(k) });
        for (let j = 0; j < 3; j++) {
          const pruned = tv[k][j] === Infinity && t > cutT[k];
          const pa = pruned ? lerp(1, 0.25, p(t, cutT[k], 0.5)) : 1;
          line(ctx, KX[k], KY, leafX(k, j), LY, { color: C.grid, w: 4, prog: e2(k, j), a: pa });
          if (e2(k, j) > 0.9) {
            const seen = t > tv[k][j];
            node(ctx, leafX(k, j), LY, 44, seen ? LEAVES[k][j] : (isAB ? '?' : ''), { a: pa, la: seen ? p(t, tv[k][j], 0.3) : 0.5, tc: seen ? C.fg : C.faint });
          }
          if (pruned) { // scissors mark on the edge
            const mx = lerp(KX[k], leafX(k, j), 0.5), my = lerp(KY, LY, 0.5), q = p(t, cutT[k] + j * 0.1, 0.3);
            line(ctx, mx - 22, my - 12, mx + 22, my + 12, { color: C.bad, w: 6, prog: q });
          }
        }
        if (e1(k) > 0.9) {
          const done = t > tk[k], bound = isAB && k > 0 ? '≤' + MINS[k] : MINS[k];
          node(ctx, KX[k], KY, 50, done ? bound : '', { color: C.o, la: p(t, tk[k], 0.3), tc: C.o });
        }
      }
      node(ctx, ROOT.x, ROOT.y, 56, t > tr ? 3 : (isAB && t > tk[0] ? '≥3' : ''), { color: C.x, tc: C.x, la: isAB && t < tr ? p(t, tk[0], 0.3) : p(t, tr, 0.3) });
      // best line
      const bl = p(t, tr + 0.4, 0.6);
      { const dx = KX[0] - ROOT.x, dy = KY - ROOT.y, L = Math.hypot(dx, dy);
        line(ctx, ROOT.x + dx / L * 60, ROOT.y + dy / L * 60, KX[0] - dx / L * 54, KY - dy / L * 54, { color: C.accent, w: 8, prog: bl, a: 0.9 }); }
      // row labels
      txt(ctx, 'X to move', 90, ROOT.y, { size: 30, align: 'left', color: C.x, weight: 600 });
      txt(ctx, 'takes the highest', 90, ROOT.y + 40, { size: 26, align: 'left', color: C.dim, a: isAB ? 1 : fade(t, s('s3') + 9.5) });
      txt(ctx, 'O to move', 90, KY, { size: 30, align: 'left', color: C.o, weight: 600, a: e1(0) });
      txt(ctx, 'takes the lowest', 90, KY + 40, { size: 26, align: 'left', color: C.dim, a: isAB ? 1 : fade(t, s('s3') + 6.5) });
      txt(ctx, 'score', 90, LY, { size: 30, align: 'left', color: C.dim, weight: 600, a: isAB ? 1 : fade(t, s('s3') + 0.5) });
      if (isAB) {
        txt(ctx, 'O can already hold X to 1 here.  X has 3 elsewhere.  Stop.', W / 2, 960, { size: 38, color: C.fg, a: fade(t, s5 + 9.4, s('s6') + 0.2) });
        txt(ctx, '4 of 9 positions never looked at', W / 2, 960, { size: 38, color: C.good, a: fade(t, s('s6') + 2.6) });
      } else {
        txt(ctx, 'minimax', W / 2, 960, { size: 46, weight: 700, color: C.accent, a: fade(t, s('s3') + 12.5) });
      }
      ctx.restore();
    }

    // s4: growth, using the engine's real perft counts from the empty board
    const a4 = fade(t, s('s4') + 0.1, s('s5') - 0.4), PERFT = [81, 720, 6336, 55080, 473256];
    if (a4 > 0) {
      txt(ctx, 'Positions after n moves, from the empty board', W / 2, 250, { size: 38, color: C.dim, a: a4 });
      PERFT.forEach((n, i) => {
        const q = p(t, s('s4') + 1.0 + i * 0.9, 0.7, ease.out);
        hbar(ctx, `${i + 1} move${i ? 's' : ''}`, n.toLocaleString('en-US'), 560, 370 + i * 100, 900, n / 473256, { a: a4 * clamp(q * 3), prog: q, h: 46, size: 36, color: C.x });
      });
      txt(ctx, '× 9 for every extra move', W / 2, 900, { size: 46, weight: 600, a: a4 * p(t, s('s4') + 6.0, 0.5) });
    }

    // s6 (second half): the ordering scores the engine really uses
    const s6 = s('s6'), a6 = fade(t, s6 + 5.4, s('s7') - 0.3);
    if (a6 > 0) {
      txt(ctx, 'Move ordering: which move do we try first?', W / 2, 220, { size: 44, weight: 600, a: a6 });
      const rows = [['best move last time we saw this position', '+1000', 1, C.good, 9.5], ['wins a small board', '+100', 0.28, C.good, 13.6],
        ['blocks a small-board win', '+75', 0.24, C.good, 14.2], ['makes two in a row', '+50', 0.2, C.good, 14.8], ['hands over a free move', '−250', 0.42, C.bad, 16.0]];
      rows.forEach(([lab, v, fr, col, dt], i) => {
        const q = p(t, s6 + dt, 0.6, ease.out);
        hbar(ctx, lab, v, 900, 360 + i * 95, 620, fr, { a: a6 * clamp(q * 3), prog: q, color: col, h: 40, size: 34 });
      });
      txt(ctx, 'ordering bonuses from the current engine', W / 2, 900, { size: 28, color: C.dim, a: a6 * fade(t, s6 + 10) });
    }

    // s7
    const a7 = fade(t, s('s7') + 0.3, s('s8') - 0.3);
    stat(ctx, '+422 Elo', 'remembered best move first · 25 January 2024 · ± 90', W / 2, 480, { a: a7 * p(t, s('s7') + 4.2, 0.5) });
    txt(ctx, 'one change', W / 2, 300, { size: 40, color: C.dim, a: a7 });

    // s8: iterative deepening against a fixed clock (schematic)
    const s8 = s('s8'), a8 = fade(t, s8 + 0.2, s('s9') - 0.3);
    if (a8 > 0) {
      const x0 = 260, x1 = 1660, y = 560, N = 16, clock = clamp((t - s8 - 1.2) / 8.5);
      txt(ctx, 'Iterative deepening', W / 2, 250, { size: 48, weight: 600, a: a8 });
      line(ctx, x0, y + 120, x1, y + 120, { color: C.grid, w: 4, a: a8 });
      txt(ctx, '0 ms', x0, y + 165, { size: 28, color: C.dim, a: a8 }); txt(ctx, '90 ms', x1, y + 165, { size: 28, color: C.dim, a: a8 });
      line(ctx, lerp(x0, x1, clock), y - 150, lerp(x0, x1, clock), y + 120, { color: C.accent, w: 3, a: a8 });
      let best = 0;
      for (let d = 1; d <= N; d++) {
        const end = Math.pow(1.42, d - N), start = d === 1 ? 0 : Math.pow(1.42, d - 1 - N);
        if (clock <= start) break;
        const q = clamp((clock - start) / (end - start)); if (q >= 1) best = d;
        rrect(ctx, lerp(x0, x1, start), y - 100, Math.max(3, (x1 - x0) * (end - start) * q - 3), 200, { fill: d % 2 ? C.x : C.o, a: a8 * 0.75, r: 4 });
        if ((end - start) * (x1 - x0) > 46) txt(ctx, String(d), lerp(x0, x1, (start + end) / 2), y, { size: 34, weight: 700, color: C.bg, a: a8 * clamp(q * 2) });
      }
      txt(ctx, `completed depth: ${best}`, W / 2, 820, { size: 44, mono: true, a: a8 });
      txt(ctx, 'schematic: each extra move of depth costs a constant factor more time', W / 2, 900, { size: 26, color: C.faint, a: a8 });
    }

    // s9
    const a9 = fade(t, s('s9') + 0.2);
    stat(ctx, '1st place', 'UVicAI tournament · February 2024', W / 2, 470, { a: a9, color: C.accent });
  },
};
