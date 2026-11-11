# Crossfish explainer video (first draft)

A 12-minute animated explainer about how Crossfish works and how it got here.
Everything is generated from code in this directory: the narration script, a
scratch voice-over, the animation frames and the final MP4.

- `SCRIPT.md` is the narration, generated from the scenes. Read this first.
- `src/scenes/*.js` holds one file per chapter: the narration segments and a
  `draw(ctx, T)` function that paints one 1920x1080 frame for any time `T.t`.
- `src/lib.js` has the drawing helpers (boards, text, arrows, bars, easing).
- `src/games.js` holds three real games recorded with `tools/play_game.py`.
  The depth and node counts on screen in the cold open are what the bot printed.
- `src/payload_sample.js` is the start of the real network payload from
  `cpp_impl/nnue_b64_net.hpp`, used in chapter 8.

## Build

Needs Node 18+, Playwright with Chromium, ffmpeg, and Python 3 with numpy.
The scratch narration also needs `pip install piper-tts` and a voice:

```bash
mkdir -p build/voice && cd build/voice
base=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high
curl -LO $base/en_US-ryan-high.onnx -LO $base/en_US-ryan-high.onnx.json
```

Then:

```bash
make video      # build/crossfish_explainer.mp4 (+ .srt subtitles), about 10 minutes on 4 cores
make sheet      # build/sheet_<scene>.png contact sheets, for a quick visual review
make preview    # scrub and play the animation in a browser
node tools/render.mjs --scene nnue      # one scene, no audio
node tools/render.mjs --still 431.5     # one frame as PNG
```

`build/` is ignored by git.

## Replacing the scratch narration with your own voice

The animation is timed against the narration, one segment at a time.

1. Record each segment of `SCRIPT.md` as `build/audio/<scene>.<segment>.wav`
   (mono, 22,050 Hz). The keys are listed in `build/script.json`.
2. Run `python3 tools/tts.py --measure` to rebuild `build/timings.json` from
   your recordings, then `make video` (skip its `audio` step, or it will
   re-synthesize any segment whose text changed).

Inside a scene, cues are written as seconds after a segment starts, for example
`T.s('n4') + 9.8`. They were tuned for the scratch voice at about 180 words per
minute. If you read much slower or faster, the cues inside long segments will
drift and need nudging. `make sheet` shows where.

## Editing

- Change wording: edit the `text` of a segment in its scene file. `say` is an
  optional respelling for the synthetic voice only ("E-low", "Coding Game").
- Add a scene: write a file exporting `{ id, title, segs, draw }` and add it to
  `src/scenes/index.js`.
- Chapter title cards are the small `title` exports next to each scene.

## Claims to check before publishing

The numbers come from `documentation/improvement_log.md`, the README and the
NNUE document. These are the places where the draft goes beyond them or
simplifies:

- **The tournament.** The docs only say "UVicAI Ultimate Tic-Tac-Toe
  tournament", January to February 2024, first place, "about two weeks". The
  script calls the organiser "the UVic AI club". Fix the name and add the
  details only you know (format, number of entrants).
- **AI coding agents.** Chapter 3 says you came back "working with AI coding
  agents", based on the commit trailers and branch names. Say as much or as
  little about that as you like.
- **"Number one on a worldwide ladder".** From the log: the bot submitted on
  2026-09-27 finished placement at rank 1 (34.27). The docs do not say how long
  it stayed there, and later placements are #3 to #9.
- **"The largest single step in the project"** for the first network
  (+267 ± 12). True among the 2026 measurements. The 2024 Python-era
  "+422 ± 90" for hash-move ordering is larger but was measured on a much
  weaker engine.
- **Elo numbers do not add.** Each is against the version just before it. The
  video says so on screen in chapters 4 and 9.
- **Chapter 5's handcrafted weights** are the values in today's source. The
  narration says "about ten weights, tuned on self-play games".
- **Chapter 7 simplifications.** The encoder's 32 outputs go through a
  per-place projection to 64 numbers (plus one extra lane); the video folds
  that into "the encoder". Only the 11,093 patterns that can occur on a live
  board are used, though the table has room for all 19,683 per place. There are
  two running totals, one per player. "About eighty million positions" is the
  round-13 data set; the first network (the +267 one) trained on 9.34 million.
  The "+1,866" on screen is a root score the engine printed in that game, used
  here as an example output.
- **Chapter 6 node counts** (700k and 147k per move) are the log's figures for
  that day's builds.
- **Schematic figures**: the iterative-deepening bar in chapter 1 and the two
  SPRT curves in chapter 4 are illustrations and are labelled as such.
