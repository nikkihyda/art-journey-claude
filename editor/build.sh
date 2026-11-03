#!/usr/bin/env bash
# Rebuild "Episode : First video of my hand-painted mirror" from the raw clips + photos.
#
#   ./build.sh <uploads_dir> <work_dir>
#
# uploads_dir must contain the 3 .mov clips and the 10 photos (names listed in make_video.py).
# Needs: ffmpeg, python3 with numpy, pillow, opencv-python-headless, faster-whisper (only for re-transcribing).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UP="${1:?uploads dir}"
WORK="${2:?work dir}"
mkdir -p "$WORK/clips" "$WORK/audio"

V1="$UP/0695ee88-copy_8F67E416-F41E-4E4A-B171-747E7064AB85.mov"
V2="$UP/a7339ae5-copy_D4679EF6-F9B3-4117-A651-14DB10D67DD9.mov"
V3="$UP/d216cfd9-copy_509DACD7-AB95-4F61-BF9A-EADC42DA5F15.mov"

# 1. frames
i=1
for f in "$V1" "$V2" "$V3"; do
  mkdir -p "$WORK/clips/v$i"
  [ -f "$WORK/clips/v$i/00001.jpg" ] || ffmpeg -v error -y -i "$f" -vf fps=30 -q:v 2 "$WORK/clips/v$i/%05d.jpg"
  ffmpeg -v error -y -i "$f" -vn -ac 1 -ar 16000 "$WORK/audio/v$i.wav"
  i=$((i+1))
done

# 2. transcript (word timestamps) -- reuse the committed one unless missing
TR="$HERE/transcript.json"

# 3. video (no audio) + sfx cue list
python3 "$HERE/make_video.py" --frames "$WORK/clips" --photos "$UP" --transcript "$TR" \
  --out "$WORK/video_noaudio.mp4" --cues "$WORK/cues.json"

# 4. music + sfx + voice-over mix
python3 "$HERE/audio.py" --clips "$V1" "$V2" "$V3" --cues "$WORK/cues.json" --meta "$WORK/cues_meta.json" \
  --out "$WORK/mix.wav"

# 5. loudness-normalise (-14 LUFS, TikTok/IG friendly) and mux
python3 "$HERE/mux.py" "$WORK/video_noaudio.mp4" "$WORK/mix.wav" "$WORK/final.mp4"
echo "done -> $WORK/final.mp4"
