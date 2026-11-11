import { W, H, C, p, fade, ease, clamp, lerp, txt, uttt, replay, rrect, line, heading, cellXY, boardXY, arrow, circle } from '../lib.js';

// A hand-built demonstration game: three opening moves, a jump ahead, then a
// small-board win and the free move that follows it.
const OPEN = [[0, 4, 2], [1, 2, 6], [0, 6, 4]];                   // [side, board, sq]
const SKIP = [[1, 4, 0], [0, 0, 0], [1, 0, 2], [0, 0, 4], [1, 6, 0], [0, 8, 1], [1, 8, 4], [0, 2, 2], [1, 3, 0]];
const LATE = [[0, 0, 8], [1, 8, 0], [0, 4, 4]];
const M = [...OPEN, ...SKIP, ...LATE].map(([side, board, sq]) => ({ side, board, sq }));

export default {
  id: 'rules', title: 'The rules',
  segs: [
    { id: 'a', text: 'The rules take thirty seconds. The first move can go anywhere. After that, the cell you pick decides which board your opponent has to play in.' },
    { id: 'b', text: 'X plays in the top right cell, so O must play in the top right board. O picks the bottom left cell, and X is sent to the bottom left board.' },
    { id: 'c', text: 'Get three in a row on a small board and you win that board.' },
    { id: 'd', text: 'If you are sent to a board that is already finished, you can play anywhere.' },
    { id: 'e', text: 'Win three small boards in a row, and you win the game.' },
    { id: 'f', text: 'That sending rule is what makes this hard. Every move is two decisions at once: where do I play, and where do I let my opponent play? A move that looks great locally can hand them exactly the board they wanted.' },
    { id: 'g', text: 'People find this game confusing. Computers are well suited to it, for the same reason they are suited to chess: you can search.', pause: 0.8 },
  ],
  draw(ctx, T) {
    const t = T.t, cx = 620, cy = 570, size = 820;
    heading(ctx, 'Ultimate Tic-Tac-Toe', fade(t, 0.3));
    const b = T.s('b'), c = T.s('c'), d = T.s('d'), e = T.s('e');
    const times = [T.s('a') + 6.3, b + 5.6, b + 9.2,
      ...SKIP.map((_, i) => c - 1.6 + i * 0.12), c + 2.2, d + 0.6, d + 3.6];
    const st = replay(M, t, i => times[i]);
    // Where the player to move is allowed to play.
    const act = [];
    const hl = (bd, t0, t1) => { const a = fade(t, t0, t1, 0.3); if (a > 0) act.push({ b: bd, a }); };
    hl(2, b + 2.4, times[1] + 0.3); hl(6, times[1] + 0.9, times[2] + 0.3);
    hl(0, c - 0.4, times[12] + 0.2); hl(8, times[12] + 1.0, times[13] + 0.2);
    for (let k = 1; k < 9; k++) hl(k, d + 2.6, times[14] + 0.3);
    st.active = act;
    if (t > e - 0.2) { // illustrate the game-winning diagonal
      [4, 8].forEach((k, i) => { st.macro[k] = 1; st.macroProg[k] = p(t, e + 0.6 + i * 0.5, 0.5, ease.out); });
    }
    uttt(ctx, cx, cy, size, st, { gridProg: p(t, 0.2, 1.0) });
    if (t > e) {
      const [x1, y1] = boardXY(cx, cy, size, 0), [x2, y2] = boardXY(cx, cy, size, 8);
      line(ctx, x1 - 90, y1 - 90, x2 + 90, y2 + 90, { color: C.accent, w: 10, prog: p(t, e + 1.7, 0.6), a: 0.9 });
    }
    // Arrow from the chosen cell to the board it sends the opponent to.
    const send = (i, t0) => {
      const m = M[i], [x1, y1] = cellXY(cx, cy, size, m.board, m.sq), [x2, y2] = boardXY(cx, cy, size, m.sq);
      const a = fade(t, t0, t0 + 2.6, 0.3);
      if (a > 0) arrow(ctx, x1, y1, lerp(x1, x2, 0.82), lerp(y1, y2, 0.82), { color: C.accent, w: 5, prog: p(t, t0, 0.6), a });
    };
    send(0, b + 1.6); send(1, times[1] + 0.5);

    // Right-hand column: the rule being shown.
    const RX = 1180, rule = (n, s, t0, t1, y) => {
      const a = fade(t, t0, t1);
      txt(ctx, n, RX, y, { size: 30, color: C.accent, align: 'left', weight: 700, a });
      txt(ctx, s, RX + 56, y, { size: 34, align: 'left', a });
    };
    const f = T.s('f'), end = f - 0.3;
    rule('1', 'Your cell picks their board', T.s('a') + 5.5, end, 330);
    rule('2', 'Three in a row wins a small board', c + 0.3, end, 430);
    rule('3', 'Sent to a finished board: play anywhere', d + 0.3, end, 530);
    rule('4', 'Three boards in a row wins the game', e + 0.3, end, 630);

    const fa = fade(t, f + 0.2, T.s('g') - 0.2);
    txt(ctx, 'Every move is two decisions', RX, 380, { size: 50, weight: 600, align: 'left', a: fa });
    txt(ctx, 'Where do I play?', RX, 490, { size: 42, align: 'left', a: fa * p(t, f + 4.6, 0.4), color: C.x });
    txt(ctx, 'Where do I send them?', RX, 570, { size: 42, align: 'left', a: fa * p(t, f + 6.0, 0.4), color: C.o });
    const ga = fade(t, T.s('g') + 0.2);
    txt(ctx, 'Hard for people.', RX, 430, { size: 54, weight: 600, align: 'left', a: ga });
    txt(ctx, 'Searchable for computers.', RX, 520, { size: 54, weight: 600, align: 'left', a: ga * p(t, T.s('g') + 2.6, 0.5), color: C.good });
  },
};
