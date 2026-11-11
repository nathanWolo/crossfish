import { W, H, C, p, fade, ease, clamp, lerp, txt, line, rrect, heading, titleCard, arrow, stat, code, rnd } from '../lib.js';
import { PAYLOAD } from '../payload_sample.js';

export const title = { id: 't_pack', segs: [{ id: 'x', text: '', silence: 2.8 }], draw: (ctx, T) => titleCard(ctx, T.t, T.dur, 'CHAPTER 8', 'Fitting in the box', '100,000 characters') };

const PARTS = [['engine code', 37868, C.o], ['network', 42682, C.accent], ['', 1641, C.dim]];
const BITS = Array.from({ length: 60 }, (_, i) => (rnd(i + 5) > 0.5 ? '1' : '0')).join('');

export default {
  id: 'pack', title: 'Chapter 8: Fitting in the box',
  segs: [
    { id: 'p1', text: 'There was one more constraint. CodinGame accepts a single source file of at most one hundred thousand characters. The engine, and every weight of the network, has to fit inside that.', say: 'There was one more constraint. Coding Game accepts a single source file of at most one hundred thousand characters. The engine, and every weight of the network, has to fit inside that.' },
    { id: 'p2', text: 'But the limit counts characters, not bytes. So the weights are compressed, and then written out fifteen bits at a time, mostly as Chinese characters. The neural network is a string of forty-two thousand of them, sitting in the middle of the source code.' },
    { id: 'p3', text: 'This is the other reason for the pattern trick. The tables the engine uses while searching take twenty-five megabytes. The source file only carries the small network that generates them.' },
    { id: 'p4', text: 'And the final submission is not even C++. It is a Python script that unpacks a binary built ahead of time with a different compiler, and runs it. That was worth another seven percent of speed.', say: 'And the final submission is not even C plus plus. It is a Python script that unpacks a binary built ahead of time with a different compiler, and runs it. That was worth another seven percent of speed.', pause: 1.2 },
  ],
  draw(ctx, T) {
    const t = T.t, s = id => T.s(id);
    heading(ctx, 'Chapter 8 · Fitting in the box', fade(t, 0.3));
    const p1 = s('p1'), p2 = s('p2'), p3 = s('p3'), p4 = s('p4');
    // the budget bar stays up for p1-p3
    const ab = fade(t, p1 + 3.0, p4 - 0.4), x0 = 260, w = 1400, y = lerp(480, 210, p(t, p2 - 0.2, 0.8));
    if (ab > 0) {
      rrect(ctx, x0, y, w, 90, { stroke: C.gridHi, a: ab, w: 3, r: 10 });
      txt(ctx, '100,000 characters', x0 + w, y - 34, { size: 30, color: C.dim, align: 'right', a: ab });
      let acc = 0;
      PARTS.forEach(([lab, n, col], i) => {
        const q = p(t, p1 + 7.5 + i * 1.6, 0.8, ease.out), bw = (n / 100000) * w * q;
        rrect(ctx, x0 + (acc / 100000) * w + 3, y + 3, Math.max(0, bw - 3), 84, { fill: col, a: ab * 0.85, r: 6 });
        if (lab) txt(ctx, `${lab} · ${n.toLocaleString('en-US')}`, x0 + ((acc + n / 2) / 100000) * w, y + 45, { size: 30, weight: 700, color: C.bg, a: ab * clamp(q * 2 - 1) });
        acc += n;
      });
      txt(ctx, '17,809 to spare', x0 + w * 0.91, y + 45, { size: 24, color: C.dim, a: ab * p(t, p1 + 12.5, 0.5) });
    }
    // p2: weights -> bits -> characters
    const a2 = fade(t, p2 + 0.3, p3 - 0.4);
    if (a2 > 0) {
      const lx = 520;
      txt(ctx, 'weights', lx, 400, { size: 30, color: C.dim, align: 'right', a: a2 });
      txt(ctx, '0.0372  −0.1184  0.2510  −0.0046  0.0931  …', lx + 40, 400, { size: 32, mono: true, align: 'left', a: a2 });
      const qb = a2 * p(t, p2 + 3.6, 0.5);
      txt(ctx, 'compressed bits', lx, 490, { size: 30, color: C.dim, align: 'right', a: qb });
      for (let g = 0; g < 4; g++) {
        const gx = lx + 40 + g * 320;
        txt(ctx, BITS.slice(g * 15, g * 15 + 15), gx, 490, { size: 28, mono: true, align: 'left', a: qb, color: g % 2 ? C.o : C.x });
        const qc = a2 * p(t, p2 + 6.0 + g * 0.5, 0.4);
        arrow(ctx, gx + 126, 520, gx + 126, 566, { color: C.grid, w: 3, a: qc, head: 12 });
        txt(ctx, PAYLOAD[g], gx + 126, 615, { size: 60, a: qc, color: g % 2 ? C.o : C.x });
      }
      txt(ctx, '15 bits per character', lx, 615, { size: 30, color: C.dim, align: 'right', a: a2 * p(t, p2 + 6.0, 0.4) });
      // the wall of text, straight from the header
      const wa = a2 * p(t, p2 + 10.5, 0.6), reveal = clamp((t - p2 - 10.5) / 6);
      rrect(ctx, 260, 700, 1400, 300, { fill: C.panel, stroke: C.grid, a: wa, w: 2 });
      for (let r = 0; r < 6; r++) txt(ctx, PAYLOAD.slice(r * 44, r * 44 + Math.floor(44 * clamp(reveal * 6 - r))), 290, 740 + r * 44, { size: 30, align: 'left', a: wa, color: C.accent });
      txt(ctx, 'cpp_impl/nnue_b64_net.hpp', 1640, 980, { size: 20, color: C.faint, align: 'right', a: wa });
    }
    // p3
    const a3 = fade(t, p3 + 0.3, p4 - 0.4);
    stat(ctx, '42,682 characters', 'in the source file', 520, 560, { a: a3, size: 70, color: C.accent, small: 30 });
    arrow(ctx, 860, 560, 1060, 560, { color: C.gridHi, w: 6, prog: p(t, p3 + 4.0, 0.6), a: a3 });
    stat(ctx, '25.6 MB of tables', 'rebuilt at start-up, every game', 1400, 560, { a: a3 * p(t, p3 + 4.4, 0.5), size: 70, color: C.fg, small: 30 });
    // p4
    const a4 = fade(t, p4 + 0.3);
    if (a4 > 0) {
      code(ctx, [['# submitted as Python 3 (simplified)', C.dim], ['elf = lzma.decompress(decode(PAYLOAD))   # built with clang', C.fg], ['fd  = os.memfd_create("bot")', C.fg], ['os.write(fd, elf)', C.fg], ['os.execv(f"/proc/self/fd/{fd}", ["bot"])', C.good]], 310, 220, { w: 1300, size: 32, a: a4 });
      stat(ctx, '+7.5% speed', 'same search tree, compared with CodinGame\'s own g++ build', W / 2, 700, { a: a4 * p(t, p4 + 9.0, 0.5), size: 120, small: 30 });
    }
  },
};
