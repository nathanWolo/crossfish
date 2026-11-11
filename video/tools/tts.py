#!/usr/bin/env python3
"""Scratch narration: synthesize every segment in build/script.json with Piper.

Writes build/audio/<key>.wav (cached by text hash) and build/timings.json, the
map from segment key to duration in seconds that the animation is timed against.
To use a human recording instead, replace the wav files (same names) and run
`tools/tts.py --measure` to rebuild timings.json without synthesizing.
"""
import hashlib, json, os, sys, wave

VOICE = os.environ.get("CF_VOICE", "build/voice/en_US-ryan-high.onnx")
SPEED = float(os.environ.get("CF_VOICE_LENGTH", "1.25"))   # >1 is slower

def wav_seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()

def main():
    segs = json.load(open("build/script.json"))
    os.makedirs("build/audio", exist_ok=True)
    measure = "--measure" in sys.argv
    voice = None
    timings = {}
    for s in segs:
        path = f"build/audio/{s['key']}.wav"
        tag = hashlib.sha1(f"{VOICE}|{SPEED}|{s['say']}".encode()).hexdigest()
        if not measure and not (os.path.exists(path) and open(path + ".tag").read() == tag if os.path.exists(path + ".tag") else False):
            if voice is None:
                from piper import PiperVoice, SynthesisConfig
                voice = PiperVoice.load(VOICE)
                cfg = SynthesisConfig(length_scale=SPEED)
            with wave.open(path, "wb") as w:
                voice.synthesize_wav(s["say"], w, syn_config=cfg)
            open(path + ".tag", "w").write(tag)
            print("tts", s["key"], f"{wav_seconds(path):.1f}s")
        timings[s["key"]] = round(wav_seconds(path), 3)
    json.dump(timings, open("build/timings.json", "w"), indent=1)
    print(f"{len(timings)} segments, {sum(timings.values())/60:.1f} min of speech")

main()
