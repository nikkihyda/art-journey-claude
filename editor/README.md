# Whimsical art journey vlog editor

Builds **"Episode : First video of my hand-painted mirror"**, a 9:16 mini vlog (59.5 s, 1080x1920, 30 fps)
from the 3 raw phone clips and 10 photos of finished hand-painted mirrors.

## What the edit does

| time | section | what you see |
|---|---|---|
| 0.0–3.7 | title | Reference-2 style title held over a beat-synced montage (her with the blank mirror, studio, finished mirrors): `[episode 01]` serif label, a tight bold pink stack *first video / of my / hand-painted / mirror ☆*, `mirror diaries no. 1 [a whimsical art journey]`, a dark `ep. 01 art journey` tag, cut-out stickers and chrome stars. The stickers cut away at 2.5 s and the title at 3.4 s. The ep. 01 tag only appears here. |
| 3.7–15.0 | unboxing | Box-cutting and frame footage with captions, `(yet!)` doodle, mirror + brush stickers on "mirror painting", paint splats, a heart doodle |
| 15.0–18.9 | "my creative journey" | Polaroids of past mirrors drop in with washi tape (camera-shutter flash) over a blurred background |
| 18.9–22.4 | intro | "oops!" on *lupa nak cakap*, then a close-up of her face with the name tag |
| 22.4–24.3 | sneak peek | Finished mirrors shown fast inside arched gold mirror frames |
| 24.3–43.7 | studio vlog | Mirror sparkles, "eep!" for *nervous*, arch/oval/round/tall shapes drawn for *bentuk cermin*, "fighting!", an animated follow button, a like heart burst |
| 43.7–53.9 | outro talk | "episode 02 ???", then a mirror-selfie photo with "rajin mode: on" |
| 53.9–59.5 | end card | Arched photo collage, "see you in episode 02", "follow for more mirror magic" |

Transitions: mirror-shine glint sweeps, painterly brush-stroke wipes (pink / butter / lavender) and zoom punches.
Accent colours: pink, butter yellow, periwinkle blue and lavender purple (no green/mint).
Grade: lifted pastel blacks, warm pink highlights, soft bloom, pink vignette, light leaks, fine grain.
Captions follow the voice-over in the reference-2 style: lowercase white Inter Display SemiBold, tight tracking, soft shadow, no outline, 2-4 word chunks.

Audio: the original voice-over plus an **original** synthesized kalimba + ukulele-pluck track
(F – Dm – B♭ – C, 100 BPM). The music ducks under the voice. Cute SFX (pops, sparkles, brush swish,
shutter, boing, ding) are generated in code. The final mix is loudness-normalised to −14 LUFS.

## Files

- `make_video.py`: the timeline, segments, overlays, captions and renderer
- `fx.py`: grading, blitting, glint, brush wipe, doodle paths, light leaks
- `stickers.py`: procedural glossy stickers (flower/star buttons, sparkle, heart, butterfly, palette, brush, mirror, bow, washi tape, polaroid, arched gold frame)
- `textfx.py`: text rendering with outline, glow and shadow
- `audio.py`: music, SFX and voice-over mix
- `mux.py`: two-pass loudnorm and mux
- `transcript.json`: word-level timestamps of the voice-over (faster-whisper large-v3)
- `fonts/`: Inter Display, Liberation Serif, Fraunces, Fredoka, Quicksand, Gaegu (all SIL OFL)

## Rebuild

```bash
pip install numpy pillow opencv-python-headless
./build.sh /path/to/uploads /path/to/workdir
```

To change caption wording, edit `CHUNKS` in `make_video.py`. To fix the name tag, edit the `vicky` text in `Assets`.
