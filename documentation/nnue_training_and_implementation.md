# NNUE training and runtime implementation

This document describes the learned evaluation stack Crossfish has shipped
since round seven. The network weights have not changed since then; the
runtime around them was made faster in rounds nine and ten (improvement log
sections 44 and 48) without changing any score. It covers the data format, teacher-label generation, feature
encoding, PyTorch models, training objectives, checkpoint formats, compression,
generated C++ headers, runtime evaluation, and strength validation.

The short version is:

```text
position
  │
  ├─ incremental handcrafted evaluation (HCE)
  │
  ├─ local-pattern D16/H8 MiniNet residual
  │    └─ nine 3×3 boards, side-to-move relative
  │
  └─ compact macro-context residual
       └─ nine won/drawn/live classes + forced-board constraint

qsearch stand pat = HCE + local residual + macro residual
```

The current evaluator is deliberately hybrid. The HCE remains a fast,
well-behaved baseline. The learned heads predict corrections to that baseline
instead of replacing it. This was stronger and much easier to fit than asking a
small network to relearn all of the known Ultimate Tic-Tac-Toe structure.

## 1. What “NNUE” means in this repository

The name is historical and should not be read as “the current evaluator is a
Stockfish-style accumulator updated on every move.”

Crossfish has experimented with several learned evaluators:

- a sparse 199-feature dual-accumulator network;
- the D8/H4 MiniNet that first shipped as an HCE residual;
- the current D16/H8 MiniNet;
- a separate macro-context residual head;
- a full Stockfish-style NNUE (199 features, 256-wide accumulator) that
  replaced HCE, MiniNet and macro together. It fitted holdout labels better
  and lost 193-296 Elo at equal depth; improvement log section 53 and
  `tools/experiments/full_nnue/` record it.

The current local MiniNet is evaluated from the board at selected search nodes,
but its first layer is heavily preprojected, and the two pieces of its input
that change rarely (each miniboard's centroid code and the decided-miniboard
term) are kept up to date by make/unmake. The macro network is reduced to an
exact table lookup whose key is maintained incrementally. The HCE itself also
has an incremental local-board accumulator.

An experiment that maintained both local MiniNet perspectives on every
make/unmake was bit-exact but slower overall. Ultimate Tic-Tac-Toe evaluates
fewer nodes than it traverses, so paying update cost on every search edge was
more expensive than reconstructing the small preprojected network only when it
was needed.

## 2. Shipped score composition

Let:

- `H(p)` be the side-to-move-relative handcrafted evaluation;
- `R_local(p)` be the D16/H8 local MiniNet output;
- `R_macro(p)` be the macro-context output;
- `C(p)` be the search correction-history adjustment.

Three correction histories are learned online during search, each an exact
table of running static-eval errors: a structural one keyed by side to move,
forced miniboard and the decided-miniboard mask; one keyed by the shape of the
forced miniboard; and one keyed by the 18-bit macro state (improvement log
sections 14, 33 and 42).

The full public evaluator used by tests and probes is:

```text
E(p) = H(p) + R_local(p) + R_macro(p)
```

The qsearch stand-pat path additionally applies the structural correction
history only (the cheapest of the three):

```text
stand_pat(p) = H(p) + C(p) + R_local(p) + R_macro(p)
```

The code avoids running both learned heads when bounds already prove that their
exact value cannot matter:

1. If `H(p) + C(p) - 640 >= beta`, qsearch returns immediately.
2. If `H(p) + C(p) + MINI_MAX + MACRO_CLIP < alpha`, it uses that safe upper
   bound instead of executing the networks.
3. Otherwise it evaluates the local residual and performs the cached macro
   lookup.

`MINI_MAX` is 8000 and `MACRO_CLIP` is 2000 in the shipped engine.

Interior reverse-futility pruning normally uses HCE corrected by all three
histories. At depth one, a selective second check evaluates the D16 local head
before pruning a borderline fail-high. The macro residual is not paid on that
interior pruning path. Reverse futility, futility and qsearch delta pruning do
not run against mate-range bounds at all, where a static eval says nothing
(improvement log section 51).

Relevant runtime files:

- `cpp_impl/crossfish_dev.hpp`
- `cpp_impl/crossfish_prev.hpp`
- `cpp_impl/codingame_nnue.cpp`
- `cpp_impl/mini_eval_d16.hpp`
- `cpp_impl/macro_eval.hpp`

The final two files are generated artifacts. Do not hand-edit their packed
weights.

## 3. Training-data format

### 3.1 `NNUEWDL1`

The local and macro trainers consume the repository’s `NNUEWDL1` binary format.
It starts with:

| Field | Size | Meaning |
| --- | ---: | --- |
| Magic | 8 bytes | ASCII `NNUEWDL1` |
| Record count | 8 bytes | little-endian `uint64_t` |

Each record is 101 bytes:

| Field | Size | Meaning after search labeling |
| --- | ---: | --- |
| UTTTAI state | 93 bytes | board, side to move, constraint, result |
| Static score | 4 bytes | little-endian `float32`, normally HCE |
| Teacher score | 4 bytes | little-endian `int32`, normally search score |

The 93-byte state is ASCII encoded:

| Byte range | Meaning |
| --- | --- |
| `0..80` | 81 local cells, nine consecutive cells per miniboard |
| `81..89` | nine super-board cells |
| `90` | side to move: `'1'` for player zero, `'2'` for player one |
| `91` | forced miniboard `'0'..'8'`, or `'9'` for free choice |
| `92` | game result field; unused by MiniNet feature extraction |

Local cell values are:

- `'0'`: empty;
- `'1'`: player zero;
- `'2'`: player one.

Super-board values are:

- `'0'`: live miniboard;
- `'1'`: won by player zero;
- `'2'`: won by player one;
- `'3'`: drawn.

The training loader rejects a dataset whose first score field still resembles
WDL probabilities (`mean(abs(y)) < 2`). This catches the most common pipeline
mistake, but it cannot detect every semantically bad label set.

### 3.2 Raw self-play data

Build the harness first:

```bash
make -C cpp_impl verify
```

Generate self-play positions:

```bash
cpp_impl/bin/test_bots dump nnue 8000 20 datasets/nnue_pos.bin
```

This starts every game with four to eight random moves, then lets the current
Dev engine play at the requested time. The ordinary `nnue` mode stores final
WDL in the float field and static HCE in the integer field. It is useful as a
position source, but it is not directly a search-score training set.

Generate random legal-game positions for broader coverage:

```bash
cpp_impl/bin/test_bots dump hce 2000000 datasets/nnue_hce_rand.bin
```

The fixed-depth search dumper expects files named `nnue_pos.bin` and
`nnue_hce_rand.bin` under `datasets/` unless equivalent paths can be found by
the harness.

### 3.3 Search-score labels

There are three supported labeling routes.

#### Mixed fixed-depth labels

```bash
cpp_impl/bin/test_bots dump search 8 800000 datasets/nnue_search_d8.bin
```

This deterministically samples half of the requested positions from random
legal play and half from engine self-play, shuffles them, and runs a full-window
fixed-depth search. The output already has:

```text
float32 = static HCE
int32   = fixed-depth search score
```

By default, the teacher search is forced to use HCE at its leaves. This is
useful when training a residual that learns what deeper search sees beyond the
static HCE without recursively teaching from the candidate network itself.

#### Relabel an existing position set

```bash
cpp_impl/bin/test_bots dump relabel 12 80000 \
  datasets/source.bin datasets/relabel_d12.bin hce
```

Use `current` instead of `hce` as the final argument to label with the current
full evaluator:

```bash
cpp_impl/bin/test_bots dump relabel 12 80000 \
  datasets/source.bin datasets/relabel_current_d12.bin current
```

Relabeling is useful for controlled distribution experiments because it keeps
the positions fixed while changing teacher depth or teacher evaluator.

#### Timed root-score labels

```bash
cpp_impl/bin/test_bots dump root 8000 20 datasets/nnue_root.bin
cpp_impl/bin/test_bots dump annotate \
  datasets/nnue_root.bin datasets/nnue_root_annotated.bin
```

`dump root` stores completed root depth in the float field and completed root
score in the integer field. `dump annotate` replaces the float field with
static HCE while preserving the integer search score.

Do not run `dump annotate` on an ordinary WDL dump and assume it has created
search labels. In an ordinary dump the integer field is already HCE, so the
result would effectively train HCE against itself.

### 3.4 Label clipping and mate handling

Search scores near the engine’s mate bounds dominate a regression loss while
providing little useful calibration for ordinary qsearch leaves.

`nnue_train_mininet.py` supports two policies:

- clip both HCE and teacher scores to `±mate_clip`;
- drop rows with `abs(search) >= mate_clip`.

The round-seven D16/H8 artifact used the drop-mates path with a threshold of
8000. The macro trainer independently removes `abs(search) >= 8000`, then
clips its remaining residual target by dropping rows outside
`±target_clip` (4000 by default).

### 3.5 Distribution quality

Offline loss is only meaningful when the position distribution resembles the
nodes where the engine actually calls the evaluator.

Useful coverage includes:

- early positions where no miniboard has been decided;
- forced-board and free-choice states;
- random legal play, which reaches shapes self-play may avoid;
- engine self-play, which emphasizes realistic tactical structures;
- positions near local and macro captures;
- multiple teacher depths;
- independent qsearch-leaf samples.

Round seven tested an additional 80,000 targeted depth-12 labels. They improved
MAE on their own targeted set by 1.77 points but worsened independent depth-12,
depth-8, and qsearch-leaf sets. The broader training distribution was retained.

The current trainer uses a deterministic random 90/10 position split. Adjacent
positions from the same game can therefore appear on both sides of the split.
Treat that validation loss as an optimizer/early-stopping signal, not as proof
of generalization. Use a separately generated dump for honest model
comparison, and use SPRT for the final decision.

## 4. Side-to-move-relative feature encoding

Both learned heads are side-to-move relative. One network handles both players.

For each local square:

```text
0 = empty
1 = occupied by the side to move
2 = occupied by the opponent
```

The nine digits of one miniboard form a ternary index:

```text
local_index = Σ digit[square] × 3^square
```

There are exactly:

```text
3^9 = 19,683
```

possible local patterns. Square zero is the least significant ternary digit,
matching `mini_index` in C++.

Each super-board cell is also converted to the side-to-move-relative class:

| Class | Meaning |
| ---: | --- |
| 0 | live |
| 1 | won by the side to move |
| 2 | won by the opponent |
| 3 | drawn |

The constraint is an integer from 0 through 9:

- `0..8`: the next move is forced into that miniboard;
- `9`: free choice, including the initial position and a send to a finished
  miniboard.

This canonicalization is important. Without it, the model would need separate
parameters for positions that are identical after swapping player colors.

## 5. Local D16/H8 MiniNet

### 5.1 Architecture

The active round-seven local network has:

- embedding width `D = 16`;
- hidden width `H = 8`;
- 19,683 local-pattern embeddings;
- four super-state embeddings;
- nine location embeddings;
- two active/inactive embeddings;
- ten constraint embeddings.

For miniboard `m`, define:

```text
v[m] =
    local_embedding[local_index[m]]
  + super_embedding[super_class[m]]
  + location_embedding[m]
  + active_embedding[constraint == m]
```

The nine vectors are concatenated with one global constraint embedding:

```text
x = concat(v[0], v[1], ..., v[8], constraint_embedding[constraint])
```

Since there are ten D-wide blocks:

```text
x has 10 × 16 = 160 elements
```

The residual is:

```text
hidden  = ReLU(W1 × x + b1)      # 8 values
R_local = W2 × hidden + b2       # one scalar in eval units
```

The active inference path contains 316,625 trainable scalar parameters:

```text
19,683×16 local embeddings
4×16      super embeddings
9×16      location embeddings
2×16      active embeddings
10×16     constraint embeddings
8×160     first-layer weights
8         first-layer biases
8         output weights
1         output bias
```

Most parameters are in the local-pattern table. The hidden mixer is
intentionally tiny because it runs at many qsearch leaves and must fit inside
the CodinGame source limit after packing.

### 5.2 Residual objective

The winning training mode is `--residual`. For each position:

```text
prediction = static_HCE + MiniNet(position)
target     = teacher_search_score
loss       = Huber(prediction, target)
```

Equivalently, the network learns the residual:

```text
teacher_search_score - static_HCE
```

The loss is expressed directly in engine evaluation units. The default Huber
delta is 1500. Huber loss behaves quadratically for ordinary errors and
linearly in the tails, reducing the influence of tactical and mate outliers.

The trainer also supports:

- direct replacement of HCE instead of residual learning;
- `asinh(score / S)` label compression;
- L2 regularization on the residual;
- residual upsampling by error magnitude;
- an output ReLU;
- a frozen baked-HCE additive head.

Those are experiment controls, not part of the shipped D16/H8 inference path.

### 5.3 Empty-board anchoring

HCE already supplies the empty-position tempo score. A residual network must
therefore contribute zero on the empty board.

The trainer can enforce this after every optimizer step with `--pin-empty`.
Regardless of that option, final MiniNet export shifts `b2` so the empty-board
residual is exactly zero in the float checkpoint.

The packing step checks the empty output again after embedding compression and
adjusts `b2` by:

```text
original_empty_output - packed_empty_output
```

This prevents a compression artifact from silently changing the opening
baseline.

### 5.4 Expanding an existing network

`--init-mini old.bin` can initialize a larger MiniNet from a smaller CFM2
checkpoint.

When D grows, each of the ten old input blocks is copied into the beginning of
the corresponding wider block. New columns start outside the inherited
function.

When H grows, old hidden rows and output weights are copied. New hidden
neurons keep randomized incoming weights but start with zero output weights.
The expanded model therefore reproduces the old model at initialization while
allowing additional capacity to learn.

The initializer rejects a request whose target D or H is smaller than the
source checkpoint.

### 5.5 Training loop

The MiniNet trainer:

1. loads the `NNUEWDL1` records;
2. verifies that the float field resembles an eval, not WDL;
3. drops or clips mate scores;
4. optionally upsamples large residuals;
5. creates or reuses a `.npz` feature cache;
6. splits records 90/10 with NumPy seed 42;
7. trains with Adam;
8. applies cosine learning-rate annealing;
9. early-stops on validation Huber loss;
10. restores the best validation checkpoint;
11. reports MAE, median absolute error, correlation, and HCE baseline metrics;
12. zero-centers the empty residual;
13. writes a CFM2 checkpoint.

Feature caching is keyed by data path, D, row count, mate policy, and upsampling
configuration. Delete the cache if the underlying dataset is replaced in
place; the filename alone cannot detect changed file contents.

### 5.6 Representative D16/H8 command

The round-seven artifact is a D16/H8 linear residual with mate rows dropped and
a learning rate of `2e-4`. A representative invocation is:

```bash
python3 tools/nnue_train_mininet.py \
  --data datasets/nnue_search.bin \
  --out artifacts/mininet_d16h8.bin \
  --arch mini \
  --residual \
  --mini-d 16 \
  --hidden 8 \
  --label linear \
  --mate-clip 8000 \
  --drop-mates \
  --upsample 0 \
  --huber 1500 \
  --lr 2e-4 \
  --epochs 40 \
  --batch 2048 \
  --patience 8
```

This is a documented template, not a byte-for-byte reconstruction command for
the checked-in header. The generated header does not contain the original
dataset path, dataset hash, or complete optimizer invocation.

## 6. CFM2 MiniNet checkpoint

`nnue_train_mininet.py` writes a little-endian float32 CFM2 file:

```text
4 bytes   magic "CFM2"
int32     D
int32     H
float32   local embeddings[19683][D]
float32   super embeddings[4][D]
float32   location embeddings[9][D]
float32   constraint embeddings[10][D]
float32   active embeddings[2][D]
float32   W1[H][10*D]
float32   b1[H]
float32   W2[H]
float32   b2
```

The trainer appends legacy scalar additive-head tables and a bake flag after
`b2`. They support older frozen-HCE experiments. The D16 header emitter reads
only the active MiniNet fields listed above.

CFM2 is an inference checkpoint, not a complete experiment record. It does not
store optimizer state, epoch, data provenance, command-line arguments, or
validation metrics.

## 7. Packing the local network for C++

The full local embedding table alone is:

```text
19,683 × 16 × 4 bytes = 1,259,712 bytes
```

Embedding it literally would exceed CodinGame’s 100,000-character source cap.
`tools/nnue_emit_mininet_header.py` compresses and preprojects it.

### 7.1 Projection-aware clustering

Ordinary k-means in embedding space treats every embedding dimension equally.
The engine only cares about errors after the first layer and output mixer.

The emitter reshapes the local portion of W1 into:

```text
[hidden=8][miniboard=9][D=16]
```

For every local embedding it computes all 72 first-layer projections, weighted
by the absolute output weight of each hidden neuron:

```text
projection_feature[index, hidden, miniboard] =
    abs(W2[hidden])
    × dot(local_embedding[index], W1[hidden, miniboard, :])
```

K-means runs in this flattened 72-dimensional projection space. This directs
capacity toward embedding differences that can actually affect the output.

There are 256 centroid codes:

- code 0 is reserved for the empty local board;
- the other 19,682 patterns are clustered into codes 1 through 255;
- each stored centroid is the mean of its members in original embedding space.

Keeping centroids in embedding space preserves the additive decomposition with
super-state, location, and active-board embeddings.

### 7.2 Packed payload

The generated payload contains:

- 19,683 one-byte centroid codes;
- 256 D16 float32 centroids;
- super, location, constraint, and active embeddings;
- W1, b1, W2, and b2.

It is CJK14 encoded into `cpp_impl/mini_eval_d16.hpp`: each 14 bits of
payload become one character in U+4E00..U+8DFF. CodinGame counts UTF-16 code
units and each of those characters is one unit, so the header stores 14
payload bits per counted character, against 6.4 for the ASCII85 encoding it
replaced. The macro payload uses the same decoder. See
[minification.md](minification.md) section 4 for the encoding details.

The raw C++ string uses `~` as its delimiter. Every payload character is
non-ASCII, so payload text cannot accidentally terminate the literal. At startup the header decodes the payload
into static storage and builds runtime tables.

### 7.3 Mask-to-code lookup

The board stores each miniboard as two 9-bit bitboards. The runtime creates:

```text
D16_MN_MASK_CODE[1 << 18]
```

indexed by:

```text
(mine_mask << 9) | opponent_mask
```

Valid disjoint masks map to the ternary index and then to the centroid code.
Overlapping masks map to code zero as a defensive fallback.

This removes ternary-index arithmetic from the hot inference loop.

### 7.4 First-layer factorization

The float model appears to require reconstruction of 160 features followed by
an `8 × 160` matrix multiply. Almost every term is constant for a small
categorical choice, so the emitter/runtime preprojects those terms once.

The shipped tables are:

```text
D16_MN_FACTOR_INIT[constraint][hidden]
D16_MN_FACTOR_CODE[miniboard][centroid][hidden]
D16_MN_FACTOR_SUPER[miniboard][super_class][hidden]
D16_MN_FACTOR_ACTIVE[miniboard][hidden]
```

`FACTOR_CODE` includes the common live/inactive local contribution.
`FACTOR_SUPER` stores a delta from the live class.
`FACTOR_ACTIVE` stores a delta from inactive.

Every projection is rounded to int32 with a fixed scale of 192. Runtime
inference is approximately:

```text
hidden = FACTOR_INIT[constraint] + super_acc[side_to_move]

for each miniboard:
    hidden += FACTOR_CODE[miniboard][mini_code[side_to_move][miniboard]]

if a miniboard is forced:
    hidden += FACTOR_ACTIVE[forced_miniboard]

output =
    b2
    + dot(W2, ReLU(float(hidden))) / 192
```

Two inputs are maintained by the search rather than looked up per leaf:

- `mini_code[perspective][miniboard]` caches each miniboard's centroid code
  (round nine). Only the miniboard a move is played in can change, so make
  updates two bytes and unmake restores them.
- `super_acc[perspective]` is the running sum of `FACTOR_SUPER` over all nine
  miniboards (round ten). It changes only when a miniboard becomes decided,
  exactly where the macro key changes. This is exact because the live-class
  rows are exactly zero (each is `lround(x - x)`).

This is deliberately not a full incremental accumulator: maintaining the whole
hidden vector on every make/unmake was measured slower (section 1).

All eight hidden lanes fit in one AVX2 vector. The main code table occupies
72 KiB:

```text
9 × 256 × 8 × 4 bytes = 73,728 bytes
```

The scalar path reconstructs the 160 float features and serves as a reference.
The unit test permits at most eight eval units of difference between the
scalar packed model and the int32 factored path across randomized games.

### 7.5 Header generation

```bash
python3 tools/nnue_emit_mininet_header.py \
  artifacts/mininet_d16h8.bin \
  -o cpp_impl/mini_eval_d16.hpp
```

The emitter currently requires exactly D16/H8 for the factored runtime.
`--no-reserve-empty` exists for experiments but was weaker than reserving the
empty pattern exactly.

## 8. Macro-context residual

The local MiniNet has detailed 3×3 pattern information, but its tiny mixer is
not an efficient way to learn every interaction among decided miniboards and
the forced-board constraint. The macro head targets the remaining error.

### 8.1 Target construction

`tools/nnue_train_macro_context.py` loads:

- the annotated search-label dataset;
- the trained local CFM2 checkpoint.

It computes:

```text
macro_target =
    teacher_search_score
  - static_HCE
  - float_local_MiniNet
```

Rows with `abs(search) >= 8000` are removed. Rows with
`abs(macro_target) > target_clip` are also removed.

The macro head therefore learns only what remains after both existing
evaluators.

### 8.2 Features and architecture

The macro model sees:

- nine side-to-move-relative super-board classes;
- the forced-board constraint.

It does not see local stones.

For miniboard `m`, the class embedding is selected from a location-specific
table entry `4*m + class[m]`. The nine embeddings and one constraint embedding
are summed:

```text
u =
    Σ macro_embedding[4*m + class[m]]
  + macro_constraint[constraint]
```

The network is:

```text
hidden  = ReLU(W_hidden × u + b_hidden)  # 16 values
raw     = W_out × hidden + b_out
R_macro = raw(position) - raw(empty_position)
```

The accepted model uses embedding width 8 and hidden width 16. Subtracting the
empty-position output in `forward()` makes zero anchoring structural rather
than an optimizer preference.

### 8.3 Training

The macro trainer uses:

- deterministic PyTorch and NumPy seeds;
- a deterministic 90/10 split;
- AdamW with weight decay `1e-4`;
- batch size 4096;
- Huber loss, delta 800 by default;
- early stopping, patience 30 by default;
- up to 200 epochs;
- 16 CPU threads by default.

A representative command is:

```bash
python3 tools/nnue_train_macro_context.py \
  --data datasets/nnue_search.bin \
  --net artifacts/mininet_d16h8.bin \
  --out artifacts/macro_d8h16.pt \
  --d 8 \
  --hidden 16 \
  --epochs 200 \
  --patience 30 \
  --lr 3e-3 \
  --huber 800 \
  --target-clip 4000 \
  --threads 16
```

The `.pt` checkpoint stores the model state, D, H, best validation MAE, base
net path, and data path.

### 8.4 Preprojection

The macro header emitter folds every categorical embedding through the hidden
weight matrix:

```text
projected_embedding = embedding × W_hiddenᵀ
projected_constraint = constraint × W_hiddenᵀ
```

The runtime payload therefore contains hidden-space contributions directly:

- hidden bias: 16 floats;
- ten projected constraints: `10 × 16` floats;
- nine-by-four projected super classes: `9 × 4 × 16` floats;
- output weights: 16 floats;
- output bias: one float.

The raw float payload is 3,076 bytes.

The accepted export multiplies output weights and the empty-adjusted output
bias by 1.25, then clips the final value to `[-2000, 2000]`.

```bash
python3 tools/nnue_emit_macro_header.py \
  artifacts/macro_d8h16.pt \
  -o cpp_impl/macro_eval.hpp \
  --scale 1.25 \
  --clip 2000
```

## 9. Exact macro lookup

Even a 16-hidden-unit AVX2 macro MLP was expensive at every qsearch leaf.
There are only:

```text
4^9 = 2^18 = 262,144
```

possible side-to-move-relative super-board states and ten possible
constraints. The generated header evaluates every combination once and stores:

```text
int16_t MACRO_SCORE[10][1 << 18]
```

The table occupies exactly 5 MiB:

```text
10 × 262,144 × 2 bytes = 5,242,880 bytes
```

Each macro key uses two bits per miniboard:

```text
key = Σ class[miniboard] << (2 × miniboard)
```

`FastBoard` carries one key for each player perspective:

```text
uint32_t macro_key[2]
```

For a miniboard won by player zero:

- player-zero key stores class 1;
- player-one key stores class 2.

For a draw, both keys store class 3. Live boards remain class 0.

The key changes only when a miniboard becomes decided or when that move is
unmade. Ordinary moves inside a live miniboard do not touch it. At evaluation,
the engine selects `macro_key[side_to_move]`, computes the current constraint,
and performs one indexed load.

The public/reference evaluator can still execute the original macro MLP from
board state. Unit tests compare that path against the lookup across randomized
games and require exact equality.

Table construction is a one-time initialization cost. It uses static storage,
not an engine-instance member, avoiding stack overflows in match workers and
the CodinGame process.

The lookup recovered roughly 6% start-position NPS within the D16 build.

## 10. Search integration

### 10.1 Initialization

At the start of a search:

1. `FastBoard` is copied from `GlobalBoard`;
2. HCE local accumulators are initialized;
3. both macro keys are reconstructed;
4. generated packed weights/tables initialize lazily on first evaluation.

### 10.2 Make/unmake

For an ordinary move in a still-live miniboard:

- local HCE state is updated;
- that miniboard's two cached centroid codes are refreshed;
- no macro key changes.

When a move wins or draws a miniboard:

- the super-board state changes;
- the cached out-of-play mask and terminal flag change;
- the relevant two-bit macro class is updated in both perspective keys, and
  the matching `FACTOR_SUPER` rows are added to both `super_acc` sums.

Make records everything it overwrites (hash, HCE scores and threat maps, the
miniboard's HCE entry, both centroid codes, active board, terminal flag and
decided state) in a 32-byte undo record, and unmake restores from it rather
than re-deriving each value (round ten).

### 10.3 Qsearch

The learned stack is primarily a leaf evaluator. Qsearch:

1. detects terminal positions;
2. computes incremental HCE plus structural correction;
3. applies the cheap HCE fail-high shortcut;
4. applies the safe upper-bound shortcut;
5. computes D16 MiniNet and cached macro residual when needed;
6. uses the result as stand pat;
7. searches local capture moves.

This placement matters. A network can have better full-position MAE yet lose
Elo if it is trained on states unlike actual qsearch leaves or costs enough
nodes to reduce search depth.

### 10.4 Interior pruning

The normal reverse-futility and futility checks remain HCE based. This avoids
paying the full learned evaluator at every interior node. A guarded depth-one
reverse-futility path adds the D16 local residual only when a cheap coarse
bound says it might confirm the cutoff.

That split is part of the accepted design; moving the networks to every static
evaluation call is a different search experiment and must be SPRT tested.

## 11. Generating a CodinGame submission

After generating both evaluator headers:

```bash
make -C cpp_impl test
make -C cpp_impl verify
make -C cpp_impl cg-input
python3 -c "s=open('cpp_impl/cg_input.cpp',encoding='utf-8').read(); print(len(s.encode('utf-16-le'))//2)"
```

`tools/cg_minify.py --inline-local` recursively expands the local generated
headers into `codingame_nnue.cpp`, then strips and renames the combined source.

The current `cg_input.cpp` is 90,095 UTF-16 code units, leaving 9,905 below
the 100,000-unit limit; the evaluator payloads account for 26,247 of them and
the opening book for 13,456. `wc -c` reports bytes, which overstate the count
because each payload character is three UTF-8 bytes; the minifier prints the
unit count. The lossless CJK14 payload encoding keeps both evaluator payloads
compact; their decoded data remains byte-for-byte identical to the accepted
round-seven networks.

Always compile both the readable and minified sources. Packing bugs can preserve
Python validation metrics while producing a broken submission. CodinGame
compiles without `-O`, so a regenerated header must keep its hot helpers
(`d16_mini_hsum256`, `evaluate_macro_key`) `always_inline`; the emitters write
the attribute ([minification.md](minification.md) section 13).

## 12. Correctness and equivalence tests

Run:

```bash
make test
make -C cpp_impl verify
```

The relevant checks include:

- scalar packed MiniNet versus the fast D16 factor path;
- original macro MLP versus exact macro-key lookup;
- macro clipping bounds;
- HCE linear/LUT consistency;
- board make/unmake and legal move generation;
- NNUE incremental-versus-refresh checks for the older sparse path;
- readable and minified CodinGame compilation.

The D16 unit test currently allows at most eight eval units between scalar
centroid inference and the int32 factor path. Macro lookup must match the float
macro evaluator exactly because the table is generated from that evaluator at
startup.

Do not enable FMA casually. The reference and generated paths are designed
around the repository’s AVX2 mul-plus-add behavior, and changing floating-point
association can change the effective network.

## 13. Strength validation

Offline validation is diagnostic. It is not the ship criterion.

For an evaluation change:

1. freeze `crossfish_prev.hpp`;
2. change only Dev and intended generated headers;
3. run correctness tests;
4. run equal-depth testing to isolate leaf quality;
5. optionally run a cheap 20 ms screen;
6. run the authoritative 90 ms SPRT with the external 100 ms referee;
7. freeze and port only after a pass.

The timed referee measures wall-clock response time outside the engine. A move
returned after CodinGame's 100 ms limit is scored as an immediate loss and
included in the printed timeout totals. This catches internal timer
regressions that ordinary W/D/L testing would otherwise misclassify as extra
search strength. Timed tests reserve one physical core for scheduler and
referee headroom; fixed-depth tests remain exempt.

The timeout-hardened direct validation against merged round six used the same
90 ms allocation as the CodinGame bot:

```text
N: 2954 W: 1100 D: 937 L: 917
Elo diff: +21.55 +/- 10.37
LLR: +3.110 (H0=0, H1=+5) — PASS
Timeouts: Prev=0 Dev=0
Maximum response: Prev=97.99 ms Dev=90.19 ms
```

The accepted direct round-seven result against the merged round-six engine was:

```text
95 ms: N 5152 W 2012 D 1591 L 1549
Elo diff: +31.31 +/- 7.91
LLR: +3.063, H0=+20, H1=+25 — PASS
Prev NPS: 13,333,760
Dev NPS: 11,787,264
```

The gain is a mixture:

- algorithmic quality from the larger local residual;
- algorithmic quality from explicit learned macro context;
- implementation speed from projection-aware centroid packing;
- implementation speed from int32 first-layer factors;
- implementation speed from exact macro lookup;
- search integration that avoids paying learned inference where HCE already
  proves the bound.

## 14. Failure modes and lessons

### Better MAE can still lose Elo

The engine chooses moves, not regression examples. Calibration improvements
that do not alter useful move ordering may be neutral. A slower evaluator can
also erase its equal-depth gain at fixed time.

### More capacity is not automatically useful

Wider hidden layers can sharply reduce NPS. Earlier H128 experiments were much
too expensive. D16/H8 is a balance among representation, packed source size,
and leaf cost.

### Incremental updates are not automatically faster

Maintaining learned state on every make/unmake only wins if enough descendant
evaluations amortize that work. The rejected dual-perspective D16 accumulator
updated far more often than the evaluator was called.

### Targeted data can overfit its own slice

The rejected depth-12 fine-tune improved its targeted set and worsened three
independent sets. Preserve broad data and evaluate on datasets generated by a
different run.

### The macro head can duplicate HCE ideas

An explicit active macro-target HCE bonus added cost and was nearly neutral.
The learned macro head already represented much of that interaction.

### Packing is part of the model

Centroid assignment, empty-board reservation, projection rounding, output
scale, clipping, and arithmetic order all define the deployed evaluator. Test
the packed C++ network, not only the PyTorch checkpoint.

## 15. Reproducibility checklist for future nets

Large training dumps and the accepted round-seven CFM2/PyTorch checkpoints are
not currently tracked in Git. The generated headers are the versioned runtime
source of truth, but they are insufficient to reconstruct training.

For each future accepted net, record:

- base commit;
- exact data-generation command;
- teacher evaluator and search depth/time;
- dataset record count and SHA-256;
- feature-cache provenance;
- exact training command;
- Python, NumPy, and PyTorch versions;
- random seeds;
- checkpoint SHA-256;
- emitter command;
- generated-header SHA-256;
- scalar-versus-packed error statistics;
- minified source character count;
- equal-depth result;
- timed SPRT result.

A practical artifact layout is:

```text
artifacts/
  round-N/
    training-command.txt
    dataset.sha256
    mininet.cfm2
    mininet.cfm2.sha256
    macro.pt
    macro.pt.sha256
    metrics.txt
```

Whether those large files live in Git, release storage, or external artifact
storage is a repository policy decision. The hashes and commands should still
be committed with the generated headers.

## 16. Source map

| File | Responsibility |
| --- | --- |
| `cpp_impl/test_bots.cpp` | data generation, search labeling, probes, SPRT |
| `tools/nnue_train_mininet.py` | local MiniNet feature extraction and training |
| `tools/nnue_train_macro_context.py` | residual macro-head training |
| `tools/nnue_emit_mininet_header.py` | D16/H8 clustering, packing, and factor runtime generation |
| `tools/nnue_emit_macro_header.py` | macro preprojection, scaling, clipping, and lookup generation |
| `tools/nnue_emit_mininet_cg.py` | legacy D8/H4 packer and shared packing helpers |
| `cpp_impl/mini_eval_d16.hpp` | generated local evaluator |
| `cpp_impl/macro_eval.hpp` | generated macro evaluator and exact lookup |
| `cpp_impl/crossfish_dev.hpp` | local search integration |
| `cpp_impl/codingame_nnue.cpp` | readable CodinGame integration |
| `tools/cg_minify.py` | header inlining and final source minification |
| `cpp_impl/unit_tests.cpp` | packed-eval equivalence checks |

For the chronological experiment history and rejected alternatives, see
[the improvement log](improvement_log.md).
