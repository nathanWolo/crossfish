#!/usr/bin/env python3
"""Place each narration wav at its timeline position -> build/narration.wav, and write subtitles (.srt)."""
import json, wave, numpy as np
tl = json.load(open("build/timeline.json"))
RATE = 22050
out = np.zeros(int((tl["total"] + 1) * RATE), dtype=np.int16)
srt = []; n = 0
def ts(t): return f"{int(t//3600):02d}:{int(t%3600//60):02d}:{int(t%60):02d},{int(t*1000%1000):03d}"
for sc in tl["scenes"]:
    for sg in sc["list"]:
        try: w = wave.open(f"build/audio/{sg['key']}.wav")
        except FileNotFoundError: continue
        assert w.getframerate() == RATE and w.getnchannels() == 1, sg["key"]
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        i = int((sc["start"] + sg["start"]) * RATE); out[i:i+len(a)] = a
        n += 1; srt.append(f"{n}\n{ts(sc['start']+sg['start'])} --> {ts(sc['start']+sg['end'])}\n{sg['text']}\n")
with wave.open("build/narration.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(out.tobytes())
open("build/crossfish_explainer.srt", "w").write("\n".join(srt))
print("narration.wav", f"{len(out)/RATE/60:.2f} min,", n, "segments")
