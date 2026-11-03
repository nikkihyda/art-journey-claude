"""Original whimsical background music + SFX synthesis, and the final VO mix.

Usage: python3 audio.py --clips v1.mov v2.mov v3.mov --cues cues.json --out mix.wav
"""
import argparse
import json
import math
import subprocess

import numpy as np

SR = 44100
BPM = 100
BEAT = 60.0 / BPM
BAR = BEAT * 4
SONG_OFFSET = 1.1          # song time = t + SONG_OFFSET  (bar 2 lands on t=3.7, the vlog "drop")

rng = np.random.default_rng(11)


def note(n):
    """note name like 'A4', 'Bb3' -> Hz"""
    names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
             "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    name, octv = n[:-1], int(n[-1])
    midi = 12 * (octv + 1) + names[name]
    return 440.0 * 2 ** ((midi - 69) / 12)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def env_ad(t, a=0.004, d=3.0):
    e = np.exp(-t * d)
    if a > 0:
        e *= np.clip(t / a, 0, 1)
    return e


# ------------------------------------------------------------------ instruments
def kalimba(f, dur=1.4, vel=1.0):
    t = tt(dur)
    s = np.sin(2 * np.pi * f * t) * env_ad(t, 0.003, 3.4)
    s += 0.22 * np.sin(2 * np.pi * f * 5.93 * t) * env_ad(t, 0.001, 22)
    s += 0.12 * np.sin(2 * np.pi * f * 2.0 * t) * env_ad(t, 0.002, 7)
    s += 0.05 * rng.normal(0, 1, len(t)) * env_ad(t, 0.0005, 300)
    return s * vel


def pluck(f, dur=1.6, vel=1.0, bright=1.0):
    t = tt(dur)
    s = np.zeros_like(t)
    for k in range(1, 9):
        s += (1.0 / k ** (1.5 - 0.3 * bright)) * np.sin(2 * np.pi * k * f * t + k) * np.exp(-t * (1.8 + k * 1.1))
    s *= np.clip(t / 0.002, 0, 1)
    return s * vel * 0.5


def bell(f, dur=2.0, vel=1.0, ratio=3.5, idx=3.0, decay=2.4):
    t = tt(dur)
    I = idx * np.exp(-t * 7)
    s = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t)) * env_ad(t, 0.002, decay)
    return s * vel


def bass(f, dur=0.55, vel=1.0):
    t = tt(dur + 0.15)
    e = env_ad(t, 0.008, 3.0) * np.clip((dur + 0.15 - t) / 0.15, 0, 1)
    s = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)) * e
    return s * vel


def pad(freqs, dur, vel=1.0):
    t = tt(dur)
    s = np.zeros_like(t)
    for f in freqs:
        for det in (-0.0035, 0.0, 0.0035):
            s += np.sin(2 * np.pi * f * (1 + det) * t + rng.uniform(0, 6))
    att = np.clip(t / 0.6, 0, 1)
    rel = np.clip((dur - t) / 0.6, 0, 1)
    trem = 1 + 0.08 * np.sin(2 * np.pi * 4.5 * t)
    return s * att * rel * trem * vel / (len(freqs) * 3)


def band_noise(n, lo, hi):
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1 / SR)
    X[(fr < lo) | (fr > hi)] = 0
    y = np.fft.irfft(X, n)
    return y / (np.abs(y).max() + 1e-9)


def kick(vel=1.0):
    t = tt(0.32)
    f = 45 + 85 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env_ad(t, 0.001, 11) * vel


def snap(vel=1.0):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    return band_noise(n, 1300, 5200) * env_ad(t, 0.0008, 38) * vel


def shaker(vel=1.0):
    n = int(0.07 * SR)
    t = np.arange(n) / SR
    e = np.clip(t / 0.012, 0, 1) * np.exp(-t * 55)
    return band_noise(n, 6000, 15000) * e * vel


# ------------------------------------------------------------------ SFX
def sfx_pop(hi=False):
    t = tt(0.16)
    f0, f1 = (1500, 650) if hi else (950, 330)
    f = f1 + (f0 - f1) * np.exp(-t * 45)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env_ad(t, 0.001, 28) * 0.9


def sfx_chime():
    out = np.zeros(int(2.2 * SR))
    for i, n in enumerate(["C6", "E6", "G6", "C7", "E7"]):
        b = bell(note(n), 1.8, 0.35, ratio=2.0, idx=1.2, decay=2.6)
        o = int(i * 0.055 * SR)
        out[o:o + len(b)] += b
    return out


def sfx_sparkle():
    out = np.zeros(int(1.4 * SR))
    pent = ["C7", "D7", "E7", "G7", "A7", "C8"]
    for i in range(12):
        n = pent[int(rng.integers(0, len(pent)))]
        b = bell(note(n) * (1 + rng.uniform(-0.004, 0.004)), 0.7, 0.18 * (1 - i / 14), ratio=1.41, idx=1.5, decay=6)
        o = int(i * 0.035 * SR)
        out[o:o + len(b)] += b
    return out


def sfx_whoosh(dur=0.38):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = band_noise(n, 300, 7000)
    # time varying brightness via moving average blend
    k = 9
    smooth = np.convolve(x, np.ones(k) / k, mode="same")
    p = t / dur
    bright = np.sin(np.pi * p)
    y = smooth * (1 - bright) + x * bright
    e = np.sin(np.pi * np.clip(p, 0, 1)) ** 1.5
    return y * e * 0.55


def sfx_swish():
    dur = 0.55
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = band_noise(n, 1800, 9000)
    am = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 23 * t + 3 * np.sin(2 * np.pi * 3 * t)))
    e = np.clip(t / 0.06, 0, 1) * np.clip((dur - t) / 0.25, 0, 1)
    return x * am * e * 0.42


def sfx_shutter():
    out = np.zeros(int(0.25 * SR))
    for o, v in ((0.0, 1.0), (0.065, 0.7)):
        n = int(0.012 * SR)
        c = band_noise(n, 1000, 9000) * np.exp(-np.arange(n) / SR * 300) * v
        i = int(o * SR)
        out[i:i + n] += c
    t = tt(0.08)
    out[:len(t)] += np.sin(2 * np.pi * 180 * t) * env_ad(t, 0.001, 50) * 0.3
    return out * 0.8


def sfx_boing():
    t = tt(0.55)
    f = 260 + 160 * np.exp(-t * 6) * np.cos(2 * np.pi * 9 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env_ad(t, 0.004, 5) * 0.6


def sfx_tap():
    n = int(0.03 * SR)
    c = band_noise(n, 2000, 8000) * np.exp(-np.arange(n) / SR * 200)
    return np.concatenate([c * 0.5, sfx_pop(True)[: int(0.1 * SR)] * 0.5])


def sfx_ding():
    return bell(note("E6"), 1.4, 0.5, ratio=2.0, idx=1.0, decay=2.8) + bell(note("B6"), 1.4, 0.25, ratio=2.0, idx=1.0, decay=3)


SFX = {
    "pop": lambda: sfx_pop(False), "pop_hi": lambda: sfx_pop(True), "chime": sfx_chime, "sparkle": sfx_sparkle,
    "whoosh": sfx_whoosh, "swish": sfx_swish, "shutter": sfx_shutter, "boing": sfx_boing, "tap": sfx_tap, "ding": sfx_ding,
}


# ------------------------------------------------------------------ composition
CHORDS = [  # (pad/strum voicing, bass root)
    (["F3", "A3", "C4", "F4", "A4"], "F2"),
    (["D3", "F3", "A3", "D4", "F4"], "D2"),
    (["Bb2", "D3", "F3", "Bb3", "D4"], "Bb1"),
    (["C3", "E3", "G3", "Bb3", "E4"], "C2"),
]
MEL_A = [["A5", None, "C6", "A5", "G5", None, "F5", None],
         ["F5", None, "A5", "F5", "E5", None, "D5", None],
         ["D5", None, "F5", "D5", "C5", "D5", "F5", None],
         ["E5", None, "G5", None, "C6", None, "Bb5", "A5"]]
MEL_B = [["C6", "A5", "F5", "A5", "C6", None, "D6", "C6"],
         ["A5", None, "F5", None, "D5", "E5", "F5", None],
         ["Bb5", "A5", "G5", "F5", "D5", None, "F5", None],
         ["G5", None, "E5", None, "C5", None, "E5", "G5"]]
MEL_C = [["F5", None, None, "A5", None, None, "C6", None],   # sparse (under heavy talking)
         ["D5", None, None, "F5", None, None, "A5", None],
         ["D5", None, None, "F5", None, None, "Bb5", None],
         ["C5", None, None, "E5", None, None, "G5", None]]


def add(buf, sig, t0, gain=1.0, pan=0.0):
    i = int(round(t0 * SR))
    if i < 0:
        sig = sig[-i:]
        i = 0
    j = min(len(buf), i + len(sig))
    if j <= i:
        return
    l = math.cos((pan + 1) * math.pi / 4)
    r = math.sin((pan + 1) * math.pi / 4)
    buf[i:j, 0] += sig[:j - i] * gain * l * 1.414
    buf[i:j, 1] += sig[:j - i] * gain * r * 1.414


def compose(total):
    buf = np.zeros((int((total + 3) * SR), 2))
    nbars = int(math.ceil((total + SONG_OFFSET) / BAR)) + 1
    for b in range(nbars):
        bt = b * BAR - SONG_OFFSET          # bar start in video time
        if bt > total:
            break
        voicing, root = CHORDS[b % 4]
        intro = b < 2
        outro_final = bt >= 56.0
        outro = 53.5 <= bt < 56.0
        # pad
        add(buf, pad([note(n) for n in voicing[1:4]], BAR + 0.4, 0.55 if intro else 0.35), bt)
        if outro_final:
            # final resolving chord
            fv = ["F3", "A3", "C4", "F4", "A4", "C5"]
            for i, n in enumerate(fv):
                add(buf, pluck(note(n), 3.0, 0.75), bt + i * 0.025, pan=-0.3 + i * 0.12)
            for i, n in enumerate(["F5", "A5", "C6", "F6", "A6"]):
                add(buf, bell(note(n), 2.2, 0.28, ratio=2.0, idx=1.0, decay=2.0), bt + 0.1 + i * 0.12, pan=0.3)
            add(buf, bass(note("F2"), 1.8, 0.7), bt)
            add(buf, kalimba(note("F5"), 2.0, 0.5), bt)
            break
        # strums (ukulele-ish), syncopated
        if not intro or b == 1:
            for k, (beat_pos, vel) in enumerate([(0, 0.9), (1.5, 0.55), (2, 0.75), (3.5, 0.55)]):
                st = bt + beat_pos * BEAT
                order = voicing[1:] if k % 2 == 0 else voicing[1:][::-1]
                for i, n in enumerate(order):
                    add(buf, pluck(note(n), 1.3, vel * 0.5, bright=0.8), st + i * 0.014, pan=-0.25 + i * 0.12)
        # bass + drums
        if not intro:
            add(buf, bass(note(root), 0.5, 0.85), bt)
            add(buf, bass(note(root) * 1.5, 0.3, 0.55), bt + 2 * BEAT)
            add(buf, bass(note(root), 0.25, 0.5), bt + 3.5 * BEAT)
            for beat in range(4):
                if beat in (0, 2):
                    add(buf, kick(0.75), bt + beat * BEAT)
                else:
                    add(buf, snap(0.32), bt + beat * BEAT, pan=0.1)
            for e in range(8):
                swing = 0.03 if e % 2 else 0.0
                add(buf, shaker(0.16 if e % 2 else 0.24), bt + e * BEAT / 2 + swing, pan=0.35)
        # melody
        if intro:
            mel = MEL_C[b % 4] if b == 1 else None
        elif outro:
            mel = MEL_B[b % 4]
        else:
            phrase = (b - 2) // 4
            mel = [MEL_A, MEL_C, MEL_B, MEL_C, MEL_A, MEL_B][phrase % 6][b % 4]
        if mel:
            for e, n in enumerate(mel):
                if n:
                    add(buf, kalimba(note(n), 1.2, 0.55), bt + e * BEAT / 2, pan=0.15 * math.sin(e))
        # sparkle glock on every 4th bar end
        if not intro and b % 4 == 3:
            for i, n in enumerate(["C6", "E6", "G6", "C7"]):
                add(buf, bell(note(n), 1.0, 0.12, ratio=1.41, idx=1.2, decay=4), bt + 3 * BEAT + i * BEAT / 4, pan=0.5)
    # intro shimmer
    for i, n in enumerate(["A5", "C6", "F6", "A6", "C7"]):
        add(buf, bell(note(n), 2.0, 0.16, ratio=2.0, idx=1.0, decay=2.0), 0.02 + i * 0.09, pan=-0.4 + 0.2 * i)
    return buf


# ------------------------------------------------------------------ mix
def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def highpass(x, fc=90):
    X = np.fft.rfft(x, axis=0)
    fr = np.fft.rfftfreq(x.shape[0], 1 / SR)
    g = 1 / np.sqrt(1 + (fc / np.maximum(fr, 1)) ** 4)
    return np.fft.irfft(X * g[:, None], x.shape[0], axis=0)


def envelope(x, att=0.02, rel=0.35, win=0.03):
    mono = np.abs(x).mean(axis=1)
    k = int(win * SR)
    rms = np.sqrt(np.convolve(mono ** 2, np.ones(k) / k, mode="same"))
    # downsample for a fast attack/release follower
    hop = 441
    r = rms[::hop]
    out = np.zeros_like(r)
    a_att = math.exp(-hop / SR / att)
    a_rel = math.exp(-hop / SR / rel)
    v = 0.0
    for i, x_ in enumerate(r):
        a = a_att if x_ > v else a_rel
        v = a * v + (1 - a) * x_
        out[i] = v
    return np.interp(np.arange(len(mono)), np.arange(len(out)) * hop, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs=3, required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--music", action="store_true", help="add the synthesized background music (off by default)")
    ap.add_argument("--ambient-clip", default=None, help="outdoor opening clip; its natural sound goes under the outro")
    args = ap.parse_args()
    meta = json.load(open(args.meta))
    total = meta["total"]
    off = {int(k): v for k, v in meta["vo_off"].items()}
    N = int(total * SR)

    # --- voice over
    vo = np.zeros((N + SR, 2))
    for i, p in enumerate(args.clips, start=1):
        a = load_audio(p)
        if i == 3:
            a = a[: int(meta["vo_end_v3"] * SR)]
            fl = int(0.12 * SR)
            a[-fl:] *= np.linspace(1, 0, fl)[:, None]
        f = int(0.004 * SR)
        a[:f] *= np.linspace(0, 1, f)[:, None]
        a[-f:] *= np.linspace(1, 0, f)[:, None]
        s = int(off[i] * SR)
        vo[s:s + len(a)] += a[: max(0, len(vo) - s)]
    vo = highpass(vo, 85)
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.89

    # --- music with sidechain ducking
    mus = compose(total)[: len(vo)]
    mus = mus / (np.abs(mus).max() + 1e-9)
    env = envelope(vo)
    talk = np.clip(env / 0.05, 0, 1)
    gain = 0.55 - 0.31 * talk              # ~-23 dBFS in gaps, ~-30 dBFS (18 dB under the voice) while talking
    # louder during the title tail and the end card
    t = np.arange(len(vo)) / SR
    gain = np.where(t > meta["vo_off"]["3"] + meta["vo_end_v3"] - 0.2, 0.95, gain)
    mus *= gain[:, None]
    # fade out
    fo = int(1.2 * SR)
    mus[N - fo:N] *= np.linspace(1, 0, fo)[:, None]
    mus[N:] = 0

    # --- sfx
    sfx = np.zeros_like(vo)
    for (tc, name, g) in json.load(open(args.cues)):
        if name in SFX:
            sig = SFX[name]()
            add(sfx, sig, tc, gain=0.33 * g, pan=float(rng.uniform(-0.3, 0.3)))

    if not args.music:
        mus[:] = 0          # voice-over + sound effects only

    # --- natural ambience of the outdoor clip under the outro (so the ending never drops to digital silence)
    amb = np.zeros_like(vo)
    if args.ambient_clip and meta.get("ambient"):
        src = load_audio(args.ambient_clip)
        src = src / (np.sqrt(np.mean(src ** 2)) + 1e-9) * 10 ** (-33 / 20)   # quiet natural bed, ~-33 dBFS RMS
        for piece in meta["ambient"]:
            a0, a1 = piece["src_from"], piece["src_to"]
            lo, hi = sorted((a0, a1))
            seg = src[int(lo * SR):int(hi * SR)].copy()
            if a0 > a1:
                seg = seg[::-1]
            f = int(0.08 * SR)
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
            s0 = int(piece["at"] * SR)
            n = min(len(seg), len(amb) - s0)
            amb[s0:s0 + n] += seg[:n]
        fo = int(0.5 * SR)
        amb[N - fo:N] *= np.linspace(1, 0, fo)[:, None]
        amb[N:] = 0
    mix = vo + mus + sfx + amb
    mix = mix[:N]
    # soft limiter
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    import wave
    with wave.open(args.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    # also export music-only for reference
    print("wrote", args.out, f"{N / SR:.2f}s")


if __name__ == "__main__":
    main()
