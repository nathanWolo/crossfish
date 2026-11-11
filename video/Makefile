# Crossfish explainer video. See README.md.
.PHONY: all script audio video preview sheet clean
OUT := build/crossfish_explainer.mp4

all: video

script:            ## narration -> build/script.json and SCRIPT.md
	node tools/export_script.mjs

audio: script      ## scratch narration with Piper -> build/audio/*.wav, build/timings.json
	python3 tools/tts.py

video: audio       ## frames + narration -> $(OUT)
	node tools/render.mjs
	python3 tools/mix_audio.py
	ffmpeg -y -loglevel error -i build/video_only.mp4 -i build/narration.wav -c:v copy -c:a aac -b:a 160k -shortest $(OUT)
	@echo wrote $(OUT)

sheet:             ## contact sheets of every scene -> build/sheet_*.png
	node tools/render.mjs --sheet --n 12 --cols 3

preview:           ## scrub the animation in a browser at http://localhost:8080
	npx http-server . -p 8080 -c-1

clean:
	rm -rf build/audio build/*.mp4 build/*.png build/*.wav build/*.json build/*.srt
