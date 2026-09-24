"""Full-eval Stockfish-style NNUE for UTTT (exploration).

Features per perspective P (sparse, summed into a shared-weight accumulator):
  cells of LIVE miniboards: (mb*9+sq)*2 + (owner != P)            0..161
  decided miniboards:       162 + mb*3 + {P won, other won, draw}  162..188
  constraint:               189 + (0..8 forced, 9 free)            189..198
Net: acc = EB(feat) + b0 (A); h = [crelu(acc_stm), crelu(acc_ntm)] -> L1 (32) -> crelu -> L2 (1).
Loss: MSE between sigmoid(pred/K) and sigmoid(teacher/K).
Dumps: NNUEWDL1, int32 = teacher search score (stm-relative), float = current static eval (baseline).
"""
import argparse, struct, sys, time
import numpy as np
import torch, torch.nn as nn

PAD = 199
NF = 200
REC = 101

def load(paths):
    arrs = []
    for p in paths:
        b = open(p, "rb").read()
        assert b[:8] == b"NNUEWDL1"
        n = struct.unpack_from("<Q", b, 8)[0]
        arrs.append(np.frombuffer(b, dtype=np.uint8, offset=16, count=n * REC).reshape(n, REC))
    a = np.concatenate(arrs)
    states = a[:, :93]
    base = np.ascontiguousarray(a[:, 93:97]).view("<f4").reshape(-1).astype(np.float32)
    teacher = np.ascontiguousarray(a[:, 97:101]).view("<i4").reshape(-1).astype(np.float32)
    return states, base, teacher

def features(states):
    n = len(states)
    cells = (states[:, :81] - ord("0")).astype(np.int16)          # 0 empty, 1 P0, 2 P1
    sup = (states[:, 81:90] - ord("0")).astype(np.int16)          # 0 live, 1 P0, 2 P1, 3 draw
    stm = (states[:, 90] - ord("1")).astype(np.int16)             # 0 = P0 to move
    con = (states[:, 91] - ord("0")).astype(np.int16)
    live_cell = np.repeat(sup == 0, 9, axis=1)                    # cell's miniboard live
    cell_idx = np.arange(81, dtype=np.int16)[None, :].repeat(n, 0)
    out = []
    for pers in (stm, 1 - stm):                                   # stm view, then ntm view
        owner = cells - 1                                         # -1 empty, 0 P0, 1 P1
        occ = (cells > 0) & live_cell
        f_cell = np.where(occ, cell_idx * 2 + (owner != pers[:, None]), PAD)
        mbs = np.arange(9, dtype=np.int16)[None, :]
        cls = np.where(sup == 3, 2, np.where(sup - 1 == pers[:, None], 0, 1))
        f_sup = np.where(sup > 0, 162 + mbs * 3 + cls, PAD)
        f_con = (189 + con)[:, None]
        f = np.concatenate([f_cell, f_sup, f_con], axis=1).astype(np.int16)
        f.sort(axis=1)                                            # PAD (199) last
        out.append(f)
    return out[0], out[1]

class Net(nn.Module):
    def __init__(self, A=256, L1=32):
        super().__init__()
        self.ft = nn.EmbeddingBag(NF, A, mode="sum", padding_idx=PAD)
        self.b0 = nn.Parameter(torch.zeros(A))
        self.l1 = nn.Linear(2 * A, L1)
        self.l2 = nn.Linear(L1, 1)
        nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad(): self.ft.weight[PAD].zero_()
    def forward(self, fs, fn):
        a = torch.clamp(self.ft(fs) + self.b0, 0, 1)
        b = torch.clamp(self.ft(fn) + self.b0, 0, 1)
        h = torch.clamp(self.l1(torch.cat([a, b], 1)), 0, 1)
        return self.l2(h).squeeze(1) * 1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--A", type=int, default=256)
    ap.add_argument("--L1", type=int, default=32)
    ap.add_argument("--K", type=float, default=2000.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=int, default=50000)
    ap.add_argument("--subsample", type=int, default=0, help="train on the first N shuffled rows (nested subsets)")
    ap.add_argument("--loss", choices=["wdl", "huber"], default="wdl")
    ap.add_argument("--huber", type=float, default=1500.0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    states, base, teacher = load(args.data)
    fs, fn = features(states)
    n = len(teacher)
    rng = np.random.default_rng(42); perm = rng.permutation(n)
    va, tr = perm[:args.val], perm[args.val:]
    if args.subsample: tr = tr[:args.subsample]
    K = args.K
    T = torch.from_numpy(teacher); B = torch.from_numpy(base)
    FS = torch.from_numpy(fs.astype(np.int64)); FN = torch.from_numpy(fn.astype(np.int64))
    sig = lambda x: torch.sigmoid(x / K)
    def metrics(pred, idx):
        t = T[idx]; nm = t.abs() < 8000
        return (float(((sig(pred) - sig(t)) ** 2).mean()),
                float((pred[nm] - t[nm]).abs().mean()), float(np.corrcoef(pred[nm].numpy(), t[nm].numpy())[0, 1]))
    vidx = torch.from_numpy(va)
    bl = metrics(B[vidx], vidx)
    print(f"rows={n} train={len(tr)} val={len(va)}  BASELINE current static eval: wdl_mse={bl[0]:.6f} mae={bl[1]:.1f} corr={bl[2]:.4f}", flush=True)
    net = Net(args.A, args.L1)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = None
    for ep in range(args.epochs):
        net.train(); t0 = time.time()
        order = torch.from_numpy(rng.permutation(tr))
        tot = 0.0; nb = 0
        for i in range(0, len(order) - args.batch + 1, args.batch):
            b = order[i:i + args.batch]
            if args.loss == "wdl":
                loss = ((sig(net(FS[b], FN[b])) - sig(T[b])) ** 2).mean()
            else:
                t = T[b]; keep = t.abs() < 8000
                loss = torch.nn.functional.huber_loss(net(FS[b][keep], FN[b][keep]), t[keep], delta=args.huber) / 1e6
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad(): net.ft.weight[PAD].zero_()
            tot += float(loss.detach()); nb += 1
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(FS[vidx[j:j+65536]], FN[vidx[j:j+65536]]) for j in range(0, len(vidx), 65536)])
        m = metrics(pv, vidx)
        print(f"epoch {ep:02d} train={tot/nb:.6f} val wdl_mse={m[0]:.6f} mae={m[1]:.1f} corr={m[2]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        score = m[0] if args.loss == "wdl" else m[1]
        if best is None or score < best:
            best = score; torch.save(net.state_dict(), args.out)
    print(f"best val score={best:.6f} vs baseline wdl_mse {bl[0]:.6f} mae {bl[1]:.1f}")
    sd = torch.load(args.out)
    with open(args.out.replace(".pt", ".bin"), "wb") as f:
        f.write(b"FNN1"); f.write(struct.pack("<ii", args.A, args.L1))
        for k in ["ft.weight", "b0", "l1.weight", "l1.bias", "l2.weight", "l2.bias"]:
            f.write(sd[k].numpy().astype("<f4").tobytes())

if __name__ == "__main__":
    main()
