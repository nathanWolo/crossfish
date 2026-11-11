#!/usr/bin/env python3
"""Referee two CodinGame-protocol bots and record the game as JSON.

Usage: play_game.py BOT_X BOT_O OUT.json
Each move record holds the 9x9 (row, col) plus the bot's own depth / eval /
node count as printed after its move ("r c D<depth> E<score> N<nodes>").
"""
import json, re, subprocess, sys, time

LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(cells):
    for a, b, c in LINES:
        if cells[a] and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return 3 if all(cells) else 0   # 3 = drawn / full

def legal(grid, macro, last):
    boards = [b for b in range(9) if not macro[b]]
    if last is not None and not macro[last[1]]:
        boards = [last[1]]
    return [(b, s) for b in boards for s in range(9) if not grid[b][s]]

def main():
    cmds, out = sys.argv[1:3], sys.argv[3]
    procs = [subprocess.Popen([c], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1) for c in cmds]
    grid = [[0]*9 for _ in range(9)]; macro = [0]*9
    last = None; moves = []; side = 0; result = 0
    while True:
        lm = legal(grid, macro, last)
        if not lm: result = 3; break
        p = procs[side]
        if last is None: p.stdin.write("-1 -1\n")
        else:
            b, s = last; p.stdin.write(f"{(b//3)*3+s//3} {(b%3)*3+s%3}\n")
        p.stdin.write(f"{len(lm)}\n" + "".join(f"{(b//3)*3+s//3} {(b%3)*3+s%3}\n" for b, s in lm)); p.stdin.flush()
        t0 = time.time(); line = p.stdout.readline().strip(); ms = (time.time()-t0)*1000
        r, c = map(int, line.split()[:2]); b, s = (r//3)*3+c//3, (r%3)*3+c%3
        assert (b, s) in lm, (line, lm)
        info = {k: int(v) for k, v in re.findall(r"\b([DEN])(-?\d+)", line)}
        grid[b][s] = side+1; macro[b] = winner(grid[b])
        moves.append(dict(side=side, board=b, sq=s, ms=round(ms, 1), n_legal=len(lm), **info))
        last = (b, s); w = winner([m if m != 3 else 0 for m in macro])
        if w in (1, 2): result = w; break
        if all(macro):  # CodinGame tiebreak: most miniboards
            a, o = macro.count(1), macro.count(2); result = 1 if a > o else 2 if o > a else 3; break
        side ^= 1
    for p in procs: p.kill()
    json.dump(dict(bots=cmds, result=result, moves=moves), open(out, "w"))
    print(out, "result", result, "plies", len(moves))

main()
