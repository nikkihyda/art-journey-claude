"""Whimsical art-journey mini vlog renderer.

Usage:
  python3 make_video.py --frames <dir with v1/ v2/ v3/ jpg sequences> --photos <dir with uploads>
                        --transcript <whisper json> --out <video_noaudio.mp4> --cues <sfx_cues.json>
                        [--start S --end E --step N]  (preview helpers)
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import fx
import stickers as st
import textfx as T
from fx import W, H, FPS, blit, pop, fade, clamp01, ease_out_cubic, ease_in_out, ease_out_back

# --------------------------------------------------------------------------------------------
# Source material
# --------------------------------------------------------------------------------------------
PHOTO_FILES = {
    "title": "7b338bea-image.png",
    2: "2382bf24-image.jpg",   # three mirrors in the garden
    3: "6c50544e-image.jpg",   # Ayu & Adzrie / Shazwan & Aqila arches
    4: "148e5bc0-image.jpg",   # round mirror Aina & Ikhwan
    5: "6da66a69-image.jpg",   # Haizzul & Hajar gold ornate
    6: "3efe039d-image.jpg",   # three: Afiq, Nazihah, birthday
    7: "4ed39779-image.jpg",   # Man & Mawaddah / Rayyan / Nazira
    8: "e32ebaa2-image.jpg",   # Azimah & Amirul / Nuryn (artist reflection)
    9: "94b894ec-image.jpg",   # Nur & Irsyad close-up
    10: "5bb551e7-image.jpg",  # engagement, white ornate oval
}

CLIP_DUR = {1: 20.571995, 2: 22.615011, 3: 11.6}
VO_OFF = {1: 0.5}
VO_OFF[2] = VO_OFF[1] + CLIP_DUR[1]
VO_OFF[3] = VO_OFF[2] + CLIP_DUR[2]
VO_END_V3 = 10.45
TOTAL = 59.5

# --------------------------------------------------------------------------------------------
# Timeline: visual segments (global seconds)
# --------------------------------------------------------------------------------------------
SEGS = [
    dict(name="title", kind="title", t0=0.0, t1=3.70),
    dict(name="cut_box", kind="clip", t0=3.70, t1=8.40, src=1, a=1.50, b=6.20, zoom=(1.06, 1.0)),
    dict(name="mirror_box", kind="clip", t0=8.40, t1=11.40, src=1, a=6.20, b=11.00, zoom=(1.0, 1.08)),
    dict(name="frame_pull", kind="clip", t0=11.40, t1=15.00, src=1, a=11.00, b=15.40, zoom=(1.08, 1.0)),
    dict(name="polaroids", kind="clip", t0=15.00, t1=18.90, src=1, a=15.40, b=19.30, zoom=(1.0, 1.05), blurbg=True),
    dict(name="oops", kind="clip", t0=18.90, t1=21.07, src=1, a=0.0, b=1.45, zoom=(1.0, 1.22), zoom_ease=True),
    dict(name="hello", kind="photo", t0=21.07, t1=22.37, photo="title", zoom=(1.55, 1.70), focus=(0.36, 0.42)),
    dict(name="portfolio", kind="portfolio", t0=22.37, t1=24.30, photos=[9, 5, 3, 2]),
    dict(name="arrange", kind="clip", t0=24.30, t1=27.60, src=2, a=3.40, b=6.70, zoom=(1.0, 1.06), sparkle=True),
    dict(name="nervous", kind="clip", t0=27.60, t1=29.60, src=2, a=6.70, b=8.70, zoom=(1.04, 1.10), shake=True),
    dict(name="shapes", kind="clip", t0=29.60, t1=32.20, src=2, a=20.45, b=22.60, zoom=(1.0, 1.07), sparkle=True),
    dict(name="nextmonth", kind="clip", t0=32.20, t1=33.70, src=2, a=10.50, b=12.00, zoom=(1.06, 1.0)),
    dict(name="siapkan", kind="clip", t0=33.70, t1=36.40, src=2, a=12.70, b=15.40, zoom=(1.0, 1.06)),
    dict(name="follow", kind="clip", t0=36.40, t1=39.10, src=2, a=15.40, b=18.10, zoom=(1.0, 1.04)),
    dict(name="like", kind="clip", t0=39.10, t1=41.00, src=2, a=18.10, b=20.00, zoom=(1.05, 1.0)),
    dict(name="studio", kind="clip", t0=41.00, t1=43.69, src=3, a=0.0, b=2.69, zoom=(1.0, 1.08), sparkle=True),
    dict(name="ep2", kind="clip", t0=43.69, t1=46.40, src=3, a=2.69, b=5.40, zoom=(1.0, 1.05)),
    dict(name="rajin", kind="clip", t0=46.40, t1=49.90, src=3, a=5.40, b=9.00, zoom=(1.0, 1.08)),
    dict(name="rajin2", kind="photo", t0=49.90, t1=53.90, photo=8, zoom=(1.12, 1.30), focus=(0.62, 0.52)),
    dict(name="end", kind="end", t0=53.90, t1=TOTAL),
]

# transitions at segment boundaries (time -> kind, colour)
TRANS = {
    3.70: ("glint", None),
    8.40: ("punch", None),
    11.40: ("brush", st.PINK),
    15.00: ("glint", None),
    18.90: ("punch", None),
    21.07: ("flash", None),
    22.37: ("brush", st.BUTTER),
    24.30: ("glint", None),
    27.60: ("punch", None),
    29.60: ("brush", st.MINT),
    32.20: ("punch", None),
    33.70: ("glint", None),
    36.40: ("punch", None),
    39.10: ("punch", None),
    41.00: ("brush", st.PINK),
    43.69: ("glint", None),
    46.40: ("punch", None),
    49.90: ("glint", None),
    53.90: ("brush", st.BUTTER),
}
BRUSH_HALF = 0.27

# --------------------------------------------------------------------------------------------
# Captions (chunks map display words onto whisper word timings)
# --------------------------------------------------------------------------------------------
# (clip, display text, first whisper word idx, last idx, style, keywords)
CHUNKS = [
    (1, "hi! this is", 0, 2, "n", []),
    (1, "my first episode", 3, 5, "n", ["first", "episode"]),
    (1, "hmm, i'm not famous", 6, 9, "n", ["famous"]),
    (1, "but i still nak buat", 10, 14, "n", []),
    (1, "floor mirror painting", 15, 17, "big", []),
    (1, "because i think", 18, 20, "n", []),
    (1, "it's time to create", 21, 24, "n", ["create"]),
    (1, "something for my own", 25, 28, "n", ["own"]),
    (1, "just want to share", 29, 32, "n", []),
    (1, "to korang", 33, 34, "n", ["korang"]),
    (1, "my creative journey", 35, 37, "big", []),
    (1, "sebelum tu", 38, 39, "n", []),
    (1, "lupa nak cakap...", 40, 42, "n", ["lupa"]),
    (2, "i'm vicky", 0, 1, "n", ["vicky"]),
    (2, "and i buat", 2, 4, "n", []),
    (2, "hand-painted mirror", 5, 7, "pink", []),
    (2, "sekarang tengah", 8, 9, "n", []),
    (2, "susun-susun lagi cermin", 10, 13, "n", ["cermin"]),
    (2, "untuk next month punya", 14, 17, "n", ["next", "month"]),
    (2, "agak nervous lah juga", 18, 21, "n", ["nervous"]),
    (2, "sebab banyak", 22, 23, "n", ["banyak"]),
    (2, "bentuk cermin", 24, 25, "big", []),
    (2, "untuk next month punya", 26, 29, "n", ["next", "month"]),
    (2, "tapi kita kena", 30, 32, "n", []),
    (2, "siapkan juga!", 33, 34, "pink", []),
    (2, "kalau nak tengok", 35, 37, "n", []),
    (2, "saya lukis,", 38, 39, "n", ["lukis,"]),
    (2, "follow tau!", 40, 41, "pink", []),
    (2, "at least like lah", 42, 45, "n", ["like"]),
    (2, "kita tengok", 46, 47, "n", []),
    (2, "jadi apa nanti video ni", 48, 52, "n", ["video"]),
    (3, "nanti akan ada lagi", 0, 3, "n", []),
    (3, "ke tak nanti?", 4, 6, "n", ["tak"]),
    (3, "sampai sini sahaja", 7, 9, "n", []),
    (3, "tapi harapnya", 10, 11, "n", []),
    (3, "saya ni rajin lah", 12, 15, "n", ["rajin"]),
    (3, "untuk edit video", 16, 18, "n", ["edit", "video"]),
]

CAP_Y_DEFAULT = 1330
CAP_Y = {  # per segment overrides
    "title": 1520, "portfolio": 1580, "hello": 1500, "polaroids": 1440, "follow": 1360, "rajin2": 1470,
}

INK = (95, 52, 82)          # deep plum for text on light backgrounds
TEAL = (52, 140, 120)
HOT = st.HOTPINK

CUES = []  # (time, sfx name, gain)


def cue(t, name, gain=1.0):
    CUES.append((round(t, 3), name, gain))


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
class ClipReader:
    def __init__(self, root):
        self.root = root
        self.cache = {}

    def frame(self, src, t):
        n = len(os.listdir(os.path.join(self.root, f"v{src}"))) if (src, "n") not in self.cache else self.cache[(src, "n")]
        self.cache[(src, "n")] = n
        f = t * FPS
        i0 = int(math.floor(f))
        frac = f - i0
        a = self._get(src, min(max(i0, 0), n - 1))
        if frac > 0.15 and i0 + 1 < n:
            b = self._get(src, min(i0 + 1, n - 1))
            return cv2.addWeighted(a, 1 - frac, b, frac, 0)
        return a

    def _get(self, src, i):
        k = (src, i)
        if k in self.cache:
            return self.cache[k]
        im = cv2.imread(os.path.join(self.root, f"v{src}", f"{i + 1:05d}.jpg"))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im.shape[1] != W or im.shape[0] != H:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        # keep cache small
        if len(self.cache) > 24:
            for kk in list(self.cache.keys())[:8]:
                if kk[1] != "n":
                    del self.cache[kk]
        self.cache[k] = im
        return im


def S(img):
    return fx.to_sprite(img)


def wobble(t, i, amp=2.5, speed=1.6):
    return amp * math.sin(t * speed * 2 * math.pi * 0.5 + i * 1.7)


def bob(t, i, amp=6, speed=1.2):
    return amp * math.sin(t * speed * 2 * math.pi * 0.5 + i * 2.3)


# --------------------------------------------------------------------------------------------
# Asset build
# --------------------------------------------------------------------------------------------
class Assets:
    def __init__(self, photo_dir):
        self.photo = {}
        for k, fn in PHOTO_FILES.items():
            im = Image.open(os.path.join(photo_dir, fn)).convert("RGB")
            if k == "title":
                a = np.asarray(im).mean(axis=(1, 2))
                rows = np.where(a > 12)[0]
                im = im.crop((0, rows.min(), im.width, rows.max() + 1))
            self.photo[k] = im
        self.photo_np = {k: np.asarray(v) for k, v in self.photo.items()}

        # stickers
        self.btn_pink = S(st.button_flower(130, st.PINK))
        self.btn_star = S(st.button_star(120))
        self.btn_mint = S(st.button_flower(115, st.MINT, dots=True, holes=4))
        self.btn_round = S(st.button_round(90, st.BUTTER))
        self.btn_pink_s = S(st.button_flower(64, st.PINK))
        self.sparkle = S(st.sparkle(110))
        self.sparkle_y = S(st.sparkle(90, color=(255, 240, 170)))
        self.heart = S(st.heart(120))
        self.heart_big = S(st.heart(330))
        self.heart_s = S(st.heart(60, st.PINK, outline=False))
        self.daisy = S(st.daisy(120))
        self.tulip = S(st.tulip(100, st.PINK))
        self.butterfly = S(st.butterfly(120))
        self.butterfly2 = S(st.butterfly(100, st.MINT, st.PINK))
        self.palette = S(st.palette(230))
        self.palette_s = S(st.palette(170))
        self.brush = S(st.brush(230, st.HOTPINK, st.MINT))
        self.brush2 = S(st.brush(200, st.YELLOW, st.PINK))
        self.mirror = S(st.mirror_icon(120))
        self.mirror_oval = S(st.mirror_icon(110, shape="oval"))
        self.bow = S(st.bow(130))
        self.splats = [S(st.paint_splat(150, c, i)) for i, c in enumerate([st.PINK, st.BUTTER, st.MINT, st.LILAC])]
        self.washi = [S(st.washi(180, 50, st.MINT, "dots", seed=1)), S(st.washi(180, 50, st.PINK, "stripes", seed=2)),
                      S(st.washi(180, 50, st.BUTTER, "grid", seed=3))]

        # title texts
        self.t_ep = S(T.render("episode :", T.QUICKSAND(62, 650), st.BUTTER,
                               shadow=(0, 4, 5, (70, 40, 90, 170))))
        self.t_first = S(T.render("first video", T.FRAUNCES(168, 430), st.MINT, stroke=4, stroke_fill=(62, 150, 128),
                                  glow=16, glow_color=(255, 255, 255), glow_strength=0.75,
                                  shadow=(0, 7, 10, (30, 50, 80, 130))))
        self.t_line3 = S(T.render("of my hand-painted", T.FREDOKA(94, 650), (248, 178, 228), stroke=5,
                                  stroke_fill=(214, 96, 176), glow=12, glow_color=(255, 255, 255), glow_strength=0.8,
                                  shadow=(0, 6, 8, (60, 30, 70, 120))))
        self.t_mirror = S(T.render("mirror", T.FRAUNCES(172, 430), st.MINT, stroke=4, stroke_fill=(62, 150, 128),
                                   glow=16, glow_color=(255, 255, 255), glow_strength=0.75,
                                   shadow=(0, 7, 10, (30, 50, 80, 130))))
        self.t_sub1 = S(T.render("(a whimsical", T.QUICKSAND(50, 600), st.BUTTER, shadow=(0, 3, 4, (0, 0, 0, 160))))
        self.t_sub2 = S(T.render("art journey)", T.QUICKSAND(50, 600), st.BUTTER, shadow=(0, 3, 4, (0, 0, 0, 160))))
        g = np.zeros((780, W, 4), np.uint8)
        a = (np.linspace(1, 0, 780) ** 1.4 * 120).astype(np.uint8)
        g[..., 0], g[..., 1], g[..., 2] = 70, 50, 120
        g[..., 3] = a[:, None]
        self.top_shade = g
        gb = np.zeros((900, W, 4), np.uint8)
        gb[..., 0], gb[..., 1], gb[..., 2] = 60, 30, 70
        gb[..., 3] = (np.linspace(0, 1, 900) ** 1.2 * 205).astype(np.uint8)[:, None]
        self.bottom_shade = gb

        # ep tag
        tag_txt = T.render("ep.01  art journey", T.QUICKSAND(34, 700), INK)
        tag = Image.new("RGBA", (tag_txt.width + 58, tag_txt.height), (0, 0, 0, 0))
        tag.alpha_composite(st.button_flower(44, st.PINK).resize((44, 44)), (0, max(0, (tag_txt.height - 44) // 2)))
        tag.alpha_composite(tag_txt, (54, 0))
        self.tag = S(st.pill(tag, fill=(255, 246, 236), pad=(20, 10)))

        # name tag
        self.t_hi = S(T.render("hi, i'm", T.QUICKSAND(66, 650), st.BUTTER, shadow=(0, 4, 6, (60, 30, 70, 170))))
        self.t_name = S(T.render("vicky", T.FRAUNCES(200, 430), st.MINT, stroke=4, stroke_fill=(62, 150, 128),
                                 glow=18, glow_color=(255, 255, 255), glow_strength=0.8,
                                 shadow=(0, 8, 12, (30, 50, 80, 140))))
        self.t_role = S(T.render("the mirror painter", T.FREDOKA(64, 620), (248, 178, 228), stroke=4,
                                 stroke_fill=(214, 96, 176), glow=10, glow_color=(255, 255, 255), glow_strength=0.8))

        # portfolio
        self.port_head = S(T.render("sneak peek", T.FREDOKA(96, 650), (248, 178, 228), stroke=5, stroke_fill=(214, 96, 176),
                                    glow=12, glow_color=(255, 255, 255), glow_strength=0.85, shadow=(0, 6, 8, (60, 30, 70, 120))))
        self.port_head2 = S(T.render("my works", T.FRAUNCES(130, 430), st.MINT, stroke=4, stroke_fill=(62, 150, 128),
                                     glow=14, glow_color=(255, 255, 255), glow_strength=0.8, shadow=(0, 6, 8, (30, 50, 80, 120))))
        self.arches = {}
        for k in (9, 5, 3, 2):
            self.arches[k] = S(st.arch_frame_photo(self.photo[k], 600, 980, focus=(0.5, 0.40)))
        self.port_bg = {}
        for k in (9, 5, 3, 2):
            bg = fx.cover_np(self.photo[k], 1.25)
            bg = cv2.resize(cv2.GaussianBlur(cv2.resize(bg, (W // 8, H // 8)), (0, 0), 3), (W, H))
            wash = np.empty_like(bg)
            wash[:] = (255, 214, 230)
            bg = cv2.addWeighted(bg, 0.55, wash, 0.45, 0)
            self.port_bg[k] = bg

        # polaroids
        self.polas = [S(st.polaroid(self.photo[4], 420, ratio=1.05)), S(st.polaroid(self.photo[10], 400, ratio=1.15)),
                      S(st.polaroid(self.photo[6], 430, ratio=1.0))]

        # misc labels
        hand = lambda s, size=72, fill=(255, 255, 255), sf=st.HOTPINK: S(T.render(s, T.HAND(size), fill, stroke=6, stroke_fill=sf,
                                                                                 shadow=(0, 5, 5, (110, 40, 80, 120))))
        self.l_yet = hand("(yet!)", 70, st.BUTTER)
        self.l_oops = S(T.render("oops!", T.FREDOKA(120, 700), (255, 255, 255), stroke=7, stroke_fill=st.HOTPINK,
                                 shadow=(0, 7, 8, (110, 40, 80, 140))))
        self.l_forme = hand("for me", 66, st.BUTTER)
        self.l_eep = hand("eep!", 80, st.BUTTER)
        self.l_shapes = [hand("arch", 52), hand("oval", 52), hand("round", 52), hand("tall", 52)]
        self.l_fight = S(st.pill(T.render("fighting!", T.FREDOKA(64, 650), (255, 255, 255)), fill=st.HOTPINK, pad=(30, 14)))
        self.l_follow = S(st.pill(T.render("+ follow", T.FREDOKA(66, 650), (255, 255, 255)), fill=st.HOTPINK, pad=(40, 16)))
        self.l_following = S(st.pill(T.render("following", T.FREDOKA(66, 650), TEAL), fill=st.MINT, pad=(40, 16)))
        self.l_like = hand("+1 like", 72, st.BUTTER)
        self.l_ep2 = S(T.render("episode 02", T.FRAUNCES(140, 430), st.MINT, stroke=4, stroke_fill=(62, 150, 128),
                                glow=14, glow_color=(255, 255, 255), glow_strength=0.8, shadow=(0, 6, 8, (30, 50, 80, 140))))
        self.l_q = S(T.render("?", T.FREDOKA(130, 700), st.BUTTER, stroke=6, stroke_fill=st.HOTPINK))
        self.l_rajin = S(st.pill(T.render("rajin mode: on", T.FREDOKA(58, 650), INK), fill=st.BUTTER, pad=(30, 14)))
        self.l_next = hand("next month!", 60, st.BUTTER)

        # end card
        bg = np.zeros((H, W, 3), np.float32)
        gy = np.linspace(0, 1, H)[:, None]
        top = np.array((255, 224, 236), np.float32)
        bot = np.array((255, 245, 214), np.float32)
        bg[:] = (top * (1 - gy) + bot * gy)[:, None, :] if False else (top[None, :] * (1 - gy) + bot[None, :] * gy)[:, None, :]
        bgu = np.clip(bg, 0, 255).astype(np.uint8)
        for y in range(40, H, 90):
            for x in range(40 + (45 if (y // 90) % 2 else 0), W, 90):
                cv2.circle(bgu, (x, y), 7, (196, 238, 214), -1, cv2.LINE_AA)
        self.end_bg = bgu
        self.e_see = S(T.render("see you in", T.QUICKSAND(70, 700), HOT))
        self.e_ep = S(T.render("episode 02", T.FRAUNCES(176, 430), st.MINT, stroke=5, stroke_fill=(62, 150, 128),
                               glow=12, glow_color=(255, 255, 255), glow_strength=0.9, shadow=(0, 7, 10, (90, 60, 90, 90))))
        self.e_follow = S(T.render("follow for more", T.FREDOKA(76, 650), (248, 178, 228), stroke=5, stroke_fill=(214, 96, 176),
                                   glow=8, glow_color=(255, 255, 255), glow_strength=0.9))
        self.e_magic = S(T.render("mirror magic", T.FRAUNCES(128, 430), st.MINT, stroke=4, stroke_fill=(62, 150, 128),
                                  glow=10, glow_color=(255, 255, 255), glow_strength=0.9, shadow=(0, 6, 8, (90, 60, 90, 90))))
        self.e_thanks = S(T.render("thank you for watching", T.HAND(64), INK))
        self.e_arch_c = S(st.arch_frame_photo(self.photo[3], 430, 700, focus=(0.22, 0.42)))
        self.e_arch_l = S(st.arch_frame_photo(self.photo[7], 300, 500, focus=(0.15, 0.45)))
        self.e_arch_r = S(st.arch_frame_photo(self.photo[5], 300, 500, focus=(0.5, 0.40)))

        self.brushes = {}

    def brush_wipe(self, color):
        if color not in self.brushes:
            self.brushes[color] = fx.BrushWipe(color, seed=len(self.brushes) * 11 + 3)
        return self.brushes[color]


# --------------------------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------------------------
class Captions:
    def __init__(self, transcript):
        self.chunks = []
        words = {}
        for k, v in transcript.items():
            i = int(os.path.basename(k)[1])
            words[i] = [w for s in v for w in s["words"]]
        for (clip, text, a, b, style, kws) in CHUNKS:
            ws = words[clip][a:b + 1]
            disp = text.split()
            off = VO_OFF[clip]
            times = []
            if len(disp) == len(ws):
                times = [off + w["s"] for w in ws]
            else:
                s0, s1 = ws[0]["s"], ws[-1]["e"]
                times = [off + s0 + (s1 - s0) * i / len(disp) for i in range(len(disp))]
            # whisper sometimes stretches a word start backwards over a pause; keep monotonic
            for i in range(1, len(times)):
                times[i] = max(times[i], times[i - 1] + 0.06)
            start = times[0] - 0.06
            end = off + ws[-1]["e"]
            self.chunks.append(dict(clip=clip, words=disp, times=times, start=start, end=end, style=style,
                                    kws=set(kws)))
        # each chunk lasts until next one starts (max hold)
        for i, c in enumerate(self.chunks):
            nxt = self.chunks[i + 1]["start"] if i + 1 < len(self.chunks) else c["end"] + 0.6
            c["end"] = min(max(c["end"] + 0.25, c["start"] + 0.6), nxt - 0.02, c["end"] + 0.9)
        self.cache = {}

    # Subtitle look follows reference 2: lowercase white sans (Inter Display SemiBold), tight tracking,
    # soft shadow, no outline, the whole 2-4 word chunk appears at once, centred in the lower middle.
    CAP_SIZE = 66
    CAP_TRACK = -2.4
    CAP_MAXW = 900

    def _line(self, text):
        text = text.replace("'", "\u2019")
        key = ("l", text)
        if key in self.cache:
            return self.cache[key]
        f = T.font("InterDisplay-SemiBold.otf", self.CAP_SIZE)
        asc, desc = f.getmetrics()
        pad = 26
        tw = int(f.getlength(text) + self.CAP_TRACK * max(0, len(text) - 1)) + 2
        Wd, Hd = tw + pad * 2, asc + desc + pad * 2
        txt = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
        d = ImageDraw.Draw(txt)
        for i, ch in enumerate(text):
            x = pad + f.getlength(text[:i]) + self.CAP_TRACK * i
            d.text((x, pad), ch, font=f, fill=(255, 255, 255, 255))
        a = np.asarray(txt.split()[3]).astype(np.float32)
        soft = cv2.GaussianBlur(cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, 3]]), (Wd, Hd)), (0, 0), 7) * 0.55
        tight = cv2.GaussianBlur(cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, 1.5]]), (Wd, Hd)), (0, 0), 1.6) * 0.30
        sh = np.clip(soft + tight, 0, 255).astype(np.uint8)
        out = Image.new("RGBA", (Wd, Hd), (20, 14, 18, 0))
        out.putalpha(Image.fromarray(sh))
        out = Image.alpha_composite(out, txt)
        self.cache[key] = out
        return out

    def sprite(self, ci, k=None):
        key = ("c", ci)
        if key in self.cache:
            return self.cache[key]
        c = self.chunks[ci]
        words = c["words"]
        full = " ".join(words)
        lines = [full]
        if self._line(full).width - 52 > self.CAP_MAXW and len(words) > 1:
            best = None
            for cut in range(1, len(words)):
                l1, l2 = " ".join(words[:cut]), " ".join(words[cut:])
                wmax = max(self._line(l1).width, self._line(l2).width)
                if best is None or wmax < best[0]:
                    best = (wmax, [l1, l2])
            lines = best[1]
        imgs = [self._line(l) for l in lines]
        step = int(self.CAP_SIZE * 1.12)
        tw = max(im.width for im in imgs)
        th = imgs[0].height + step * (len(imgs) - 1)
        canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        for i, im in enumerate(imgs):
            canvas.alpha_composite(im, ((tw - im.width) // 2, i * step))
        spr = S(canvas)
        self.cache[key] = spr
        return spr

    def draw(self, frame, t, seg_name):
        for ci, c in enumerate(self.chunks):
            if c["start"] <= t <= c["end"]:
                spr = self.sprite(ci)
                a = min(clamp01((t - c["start"]) / 0.07), clamp01((c["end"] - t) / 0.07))
                dy = 10 * (1 - ease_out_cubic((t - c["start"]) / 0.18))
                y = CAP_Y.get(seg_name, CAP_Y_DEFAULT)
                rot = 0.0
                dx = 0
                if seg_name == "nervous" and "nervous" in " ".join(c["words"]):
                    dx = 4 * math.sin(t * 70)
                    rot = 1.0 * math.sin(t * 53)
                blit(frame, spr, W / 2 + dx, y + dy, rot=rot, alpha=a)


# --------------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------------
class Renderer:
    def __init__(self, args):
        self.A = Assets(args.photos)
        self.clips = ClipReader(args.frames)
        self.cap = Captions(json.load(open(args.transcript)))
        self.rng = random.Random(1)
        self._register_cues()

    # ---------------- cues for the audio mix
    def _register_cues(self):
        cue(0.05, "chime", 0.8)
        for t in (0.35, 0.65, 1.15, 1.55, 2.0):
            cue(t, "pop", 0.7)
        cue(0.9, "pop_hi", 0.5); cue(1.3, "pop_hi", 0.5); cue(1.75, "pop_hi", 0.5); cue(2.2, "pop", 0.5); cue(2.4, "pop", 0.5)
        for t, (kind, _) in TRANS.items():
            if kind == "glint":
                cue(t - 0.12, "sparkle", 0.8)
            elif kind == "brush":
                cue(t - BRUSH_HALF, "swish", 0.75)
            elif kind == "punch":
                cue(t - 0.05, "whoosh", 0.55)
            elif kind == "flash":
                cue(t - 0.03, "sparkle", 0.6)
        cue(4.55, "pop_hi", 0.5)                 # (yet!)
        cue(7.62, "sparkle", 0.6)                # floor mirror painting
        cue(12.32, "pop", 0.5); cue(12.45, "pop_hi", 0.4)
        for i in range(3):
            cue(15.05 + i * 0.6, "shutter", 0.7)
        cue(19.95, "boing", 0.7)                 # oops
        cue(21.12, "chime", 0.6)                 # hi i'm vicky
        for i in range(4):
            cue(22.37 + i * 0.4825, "pop", 0.55)
        cue(28.0, "boing", 0.5)                  # nervous
        for i in range(4):
            cue(30.25 + i * 0.32, "pop_hi", 0.45)  # shapes
        cue(35.1, "pop", 0.7)                    # fighting
        cue(38.15, "pop", 0.7); cue(38.55, "tap", 0.8); cue(38.62, "ding", 0.5)   # follow
        cue(40.16, "pop", 0.7); cue(40.3, "ding", 0.7)                            # like
        cue(44.5, "pop", 0.6); cue(44.8, "pop_hi", 0.4); cue(45.0, "pop_hi", 0.4)  # ep 02 ?
        cue(50.9, "pop", 0.6)                    # rajin
        for t in (54.25, 54.45, 54.65, 55.0, 55.3, 56.0, 56.3):
            cue(t, "pop", 0.45)
        cue(54.2, "chime", 0.7)
        cue(58.7, "chime", 0.6)

    # ---------------- base frames per segment
    def seg_at(self, t):
        for i, s in enumerate(SEGS):
            if s["t0"] <= t < s["t1"]:
                return i, s
        return len(SEGS) - 1, SEGS[-1]

    def base(self, s, t):
        lt = t - s["t0"]
        dur = s["t1"] - s["t0"]
        p = lt / dur
        if s["kind"] == "clip":
            src_t = s["a"] + (s["b"] - s["a"]) * p
            fr = self.clips.frame(s["src"], src_t)
            z0, z1 = s.get("zoom", (1.0, 1.0))
            pe = ease_in_out(p) if s.get("zoom_ease") else p
            z = z0 + (z1 - z0) * pe
            dx = dy = rot = 0.0
            if s.get("shake"):
                dx = 7 * math.sin(t * 37) + 4 * math.sin(t * 61)
                dy = 5 * math.sin(t * 43)
                rot = 0.5 * math.sin(t * 29)
            return fx.camera(fr, z, dx, dy, rot)
        if s["kind"] == "title":
            z = 1.13 - 0.10 * ease_out_cubic(p)
            return fx.cover_np(self.A.photo_np["title"], z, 0.5, 0.42)
        if s["kind"] == "photo":
            z0, z1 = s["zoom"]
            z = z0 + (z1 - z0) * ease_out_cubic(p)
            return fx.cover_np(self.A.photo_np[s["photo"]], z, *s["focus"])
        if s["kind"] == "portfolio":
            n = len(s["photos"])
            k = min(n - 1, int(p * n))
            fr = self.A.port_bg[s["photos"][k]].copy()
            return fr
        if s["kind"] == "end":
            return self.A.end_bg.copy()
        raise ValueError(s)

    # ---------------- segment overlays
    def overlay(self, fr, s, t):
        lt = t - s["t0"]
        name = s["name"]
        A = self.A
        if name == "title":
            self.title_overlay(fr, lt)
        elif name == "cut_box":
            # (yet!) after "not famous"
            p = pop(t, 4.55, 0.3)
            if 4.55 <= t < 6.3:
                blit(fr, A.l_yet, 860, 1205, scale=p, rot=10 + wobble(t, 1), alpha=fade(t, 4.55, 6.3, 0.05, 0.15))
                fx.draw_doodle(fr, fx.star_path(950, 1120, 26), clamp01((t - 4.7) / 0.35), color=st.BUTTER, width=6, shadow=st.HOTPINK)
                fx.draw_doodle(fr, fx.star_path(760, 1140, 16), clamp01((t - 4.85) / 0.3), color=(255, 255, 255), width=5, shadow=st.HOTPINK)
            # mirror + brush stickers for "floor mirror painting"
            self.floor_mirror(fr, t)
        elif name == "mirror_box":
            self.floor_mirror(fr, t)
            # sparkles over the mirror surface in the box
            for i, (x, y) in enumerate([(300, 1000), (720, 1080), (520, 930)]):
                ph = (t * 1.3 + i * 0.37) % 1.0
                blit(fr, A.sparkle, x, y, scale=math.sin(math.pi * ph) * 0.9, rot=ph * 90)
        elif name == "frame_pull":
            # "create" -> paint splats, "for my own" -> heart doodle
            for i, (x, y, tt) in enumerate([(140, 380, 12.32), (960, 520, 12.45), (900, 1000, 12.6)]):
                if t >= tt:
                    blit(fr, A.splats[i], x, y, scale=pop(t, tt, 0.25) * 0.9, rot=i * 40 + wobble(t, i, 3))
            if t >= 13.30:
                fx.draw_doodle(fr, fx.heart_path(820, 1080, 230), clamp01((t - 13.3) / 0.55), color=(255, 255, 255), width=8, shadow=st.HOTPINK)
                blit(fr, A.l_forme, 820, 1075, scale=pop(t, 13.75, 0.3), rot=-6 + wobble(t, 2))
        elif name == "polaroids":
            self.polaroid_overlay(fr, lt, t)
        elif name == "oops":
            if t >= 19.92:
                p = pop(t, 19.92, 0.3)
                blit(fr, A.l_oops, 760, 640, scale=p, rot=-10 + wobble(t, 3, 4, 3))
                for i, (x, y) in enumerate([(940, 520), (985, 600), (600, 520)]):
                    pp = clamp01((t - 20.05 - i * 0.08) / 0.25)
                    fx.draw_doodle(fr, [(x, y), (x + 14, y + 30)], pp, color=(170, 220, 255), width=9, shadow=(255, 255, 255))
                blit(fr, A.heart_s, 560, 770, scale=pop(t, 20.2, 0.25), rot=-15)
        elif name == "hello":
            self.hello_overlay(fr, lt, t)
        elif name == "portfolio":
            self.portfolio_overlay(fr, lt, t, s)
        elif name == "arrange":
            u = clamp01((t - 24.6) / 2.9)
            if 0 < u < 1:
                flap = 0.35 + 0.65 * abs(math.cos(t * 13))
                blit(fr, A.butterfly2, 1150 - u * 1300, 700 - 260 * u + 50 * math.sin(u * 8), scale=0.85, sx=0.85 * flap, rot=12)
            self.mirror_sparkles(fr, t, [(860, 560), (980, 900), (720, 380), (900, 1450)])
        elif name == "nervous":
            if t >= 28.0:
                blit(fr, A.l_eep, 300, 1130, scale=pop(t, 28.0, 0.25), rot=8 + 4 * math.sin(t * 40))
                for i, (x, y) in enumerate([(560, 1060), (610, 1110), (180, 1050)]):
                    pp = clamp01((t - 28.1 - i * 0.07) / 0.2)
                    fx.draw_doodle(fr, [(x, y), (x + 10, y + 28)], pp, color=(170, 220, 255), width=9, shadow=(255, 255, 255))
        elif name == "shapes":
            self.shapes_overlay(fr, t)
            self.mirror_sparkles(fr, t, [(430, 1000), (760, 1200), (300, 1350)])
        elif name == "nextmonth":
            pass
        elif name == "siapkan":
            if t >= 35.05:
                blit(fr, A.l_fight, 760, 1110, scale=pop(t, 35.05, 0.3), rot=-6 + wobble(t, 4))
                blit(fr, A.btn_star, 960, 1020, scale=pop(t, 35.2, 0.3) * 0.8, rot=wobble(t, 5, 8))
                fx.draw_doodle(fr, fx.star_path(560, 1030, 22), clamp01((t - 35.25) / 0.3), color=st.BUTTER, width=6)
        elif name == "follow":
            self.follow_overlay(fr, t)
        elif name == "like":
            self.like_overlay(fr, t)
        elif name == "studio":
            self.mirror_sparkles(fr, t, [(160, 900), (420, 820), (700, 1000), (930, 760), (560, 1250)])
            if t > 42.0:
                fx.glint(fr, clamp01((t - 42.0) / 0.8), strength=0.45)
        elif name == "ep2":
            self.ep2_overlay(fr, t)
        elif name in ("rajin", "rajin2"):
            if name == "rajin2":
                blit(fr, A.bottom_shade, W / 2, H, anchor=(0.5, 1.0), alpha=clamp01((t - 49.9) / 0.3))
            if t >= 50.85:
                blit(fr, A.l_rajin, 560, 640, scale=pop(t, 50.85, 0.3), rot=-5 + wobble(t, 6))
                blit(fr, A.palette_s, 190, 780, scale=pop(t, 51.0, 0.3), rot=-12 + wobble(t, 7, 4))
                blit(fr, A.brush2, 910, 740, scale=pop(t, 51.15, 0.3), rot=28 + wobble(t, 8, 4))
            if t >= 47.0:
                fx.glint(fr, clamp01((t - 47.0) / 0.9), strength=0.35)
        elif name == "end":
            self.end_overlay(fr, lt, t)

    def mirror_sparkles(self, fr, t, pts):
        for i, (x, y) in enumerate(pts):
            ph = (t * 0.9 + i * 0.29) % 1.0
            sc = math.sin(math.pi * ph) ** 2
            spr = self.A.sparkle if i % 2 == 0 else self.A.sparkle_y
            blit(fr, spr, x, y, scale=sc * (0.9 if i % 2 == 0 else 0.7), rot=ph * 120)

    def floor_mirror(self, fr, t):
        # chunk "floor mirror painting" : g ~7.6 - 10.1
        A = self.A
        if 7.6 <= t <= 10.25:
            a = fade(t, 7.6, 10.25, 0.05, 0.15)
            blit(fr, A.mirror, 175, 1180, scale=pop(t, 7.62, 0.3), rot=-10 + wobble(t, 0), alpha=a)
            blit(fr, A.brush, 930, 1175, scale=pop(t, 8.1, 0.3), rot=25 + wobble(t, 1), alpha=a)
            blit(fr, A.sparkle, 255, 1075, scale=0.7 * abs(math.sin(t * 4)), alpha=a)
            blit(fr, A.sparkle_y, 860, 1080, scale=0.6 * abs(math.sin(t * 4 + 1)), alpha=a)

    def title_overlay(self, fr, t):
        A = self.A
        blit(fr, A.top_shade, W / 2, 0, anchor=(0.5, 0.0))
        # sparkles on the blank mirror
        for i, (x, y) in enumerate([(612, 705), (870, 1010), (640, 1380), (875, 1700)]):
            ph = (t * 0.8 + i * 0.31) % 1.0
            blit(fr, A.sparkle, x, y, scale=math.sin(math.pi * ph) ** 2 * 0.9, rot=ph * 90)
        # butterfly flying across
        if t > 0.2:
            u = clamp01((t - 0.2) / 3.5)
            bx = -80 + u * 1240
            by = 900 - 380 * u + 60 * math.sin(u * 7)
            flap = 0.35 + 0.65 * abs(math.cos(t * 13))
            blit(fr, A.butterfly, bx, by, scale=0.9, sx=0.9 * flap, rot=-15 + 10 * math.sin(t * 3))
        blit(fr, A.t_ep, 230 + bob(t, 0, 3), 150, scale=pop(t, 0.35), rot=-5 + wobble(t, 0, 1.5))
        blit(fr, A.t_first, 520, 285 + bob(t, 1, 4), scale=pop(t, 0.65, 0.36), rot=-3 + wobble(t, 1, 1.2))
        blit(fr, A.btn_pink, 965, 225 + bob(t, 2), scale=pop(t, 0.9), rot=wobble(t, 2, 8) + 10)
        blit(fr, A.t_line3, 540, 430 + bob(t, 3, 4), scale=pop(t, 1.15, 0.34), rot=-2 + wobble(t, 3, 1.2))
        blit(fr, A.btn_star, 78, 420 + bob(t, 4), scale=pop(t, 1.3) * 0.9, rot=wobble(t, 4, 10) - 12)
        blit(fr, A.t_mirror, 770, 560 + bob(t, 5, 4), scale=pop(t, 1.55, 0.36), rot=-4 + wobble(t, 5, 1.2))
        blit(fr, A.btn_mint, 470, 590 + bob(t, 6), scale=pop(t, 1.75) * 0.85, rot=wobble(t, 6, 10))
        a_sub = clamp01((t - 2.0) / 0.3)
        blit(fr, A.t_sub1, 742, 800 + (1 - ease_out_cubic(a_sub)) * 30, alpha=a_sub, rot=-2)
        blit(fr, A.t_sub2, 742, 862 + (1 - ease_out_cubic(clamp01((t - 2.12) / 0.3))) * 30,
             alpha=clamp01((t - 2.12) / 0.3), rot=-2)
        blit(fr, A.palette, 150, 1720, scale=pop(t, 2.2), rot=-14 + wobble(t, 7, 3))
        blit(fr, A.brush, 330, 1650, scale=pop(t, 2.4), rot=38 + wobble(t, 8, 3))

    def polaroid_overlay(self, fr, lt, t):
        A = self.A
        spots = [(300, 760, -9), (790, 700, 8), (560, 1040, -3)]
        for i, (x, y, r) in enumerate(spots):
            t0 = 0.05 + i * 0.6
            if lt < t0:
                continue
            u = clamp01((lt - t0) / 0.38)
            yy = y - (1 - ease_out_back(u, 1.4)) * 380
            sc = 1.15 - 0.15 * ease_out_cubic(u)
            blit(fr, A.polas[i], x, yy, scale=sc, rot=r + wobble(t, i, 1.0))
            # tape
            tu = clamp01((lt - t0 - 0.25) / 0.15)
            if tu > 0:
                tx = x + math.sin(math.radians(r)) * 0
                ty = yy - A.polas[i].shape[0] * 0.5 + 18
                blit(fr, A.washi[i], tx, ty, scale=tu, rot=r + (12 if i % 2 else -10))
        # flash on each shutter
        for i in range(3):
            d = lt - (0.05 + i * 0.6)
            if 0 <= d < 0.1:
                fx.flash(fr, 0.35 * (1 - d / 0.1))
        if lt > 1.9:
            blit(fr, A.daisy, 930, 1180, scale=pop(lt, 1.9) * 0.8, rot=wobble(t, 1, 6))
            blit(fr, A.heart, 150, 1170, scale=pop(lt, 2.05) * 0.8, rot=-12 + wobble(t, 2, 5))

    def hello_overlay(self, fr, lt, t):
        A = self.A
        blit(fr, A.top_shade, W / 2, 0, anchor=(0.5, 0.0), alpha=0.8)
        blit(fr, A.t_hi, 300, 220, scale=pop(lt, 0.05), rot=-6)
        blit(fr, A.t_name, 560, 360 + bob(t, 1, 4), scale=pop(lt, 0.12, 0.36), rot=-4 + wobble(t, 1, 1.2))
        blit(fr, A.bow, 860, 255, scale=pop(lt, 0.3) * 0.9, rot=18 + wobble(t, 2, 5))
        blit(fr, A.t_role, 560, 520, scale=pop(lt, 0.35, 0.3), rot=-2)
        fx.draw_doodle(fr, fx.squiggle_path(330, 790, 590, 9, 5), clamp01((lt - 0.45) / 0.4), color=st.BUTTER, width=7, shadow=st.HOTPINK)
        for i, (x, y) in enumerate([(140, 330), (980, 420), (890, 620)]):
            ph = (t * 1.1 + i * 0.33) % 1
            blit(fr, A.sparkle_y if i % 2 else A.sparkle, x, y, scale=math.sin(math.pi * ph) * 0.8, rot=ph * 90)

    def portfolio_overlay(self, fr, lt, t, s):
        A = self.A
        photos = s["photos"]
        dur = s["t1"] - s["t0"]
        n = len(photos)
        each = dur / n
        k = min(n - 1, int(lt / each))
        u = (lt - k * each) / each
        spr = A.arches[photos[k]]
        sc = 0.86 + 0.14 * ease_out_back(clamp01(u / 0.45), 1.6) + 0.03 * u
        rot = (-3 if k % 2 else 3) * (1 - 0.6 * u)
        blit(fr, spr, W / 2, 900, scale=sc, rot=rot)
        if u < 0.12 and k > 0:
            fx.flash(fr, 0.5 * (1 - u / 0.12))
        blit(fr, A.port_head, 380, 190, scale=pop(lt, 0.0, 0.3), rot=-4)
        blit(fr, A.port_head2, 700, 330, scale=pop(lt, 0.12, 0.3), rot=-4)
        blit(fr, A.btn_pink, 115, 300, scale=pop(lt, 0.2) * 0.8, rot=wobble(t, 0, 10))
        blit(fr, A.btn_star, 980, 1380, scale=pop(lt, 0.3) * 0.8, rot=wobble(t, 1, 10))
        blit(fr, A.btn_mint, 110, 1330, scale=pop(lt, 0.4) * 0.75, rot=wobble(t, 2, 10))
        for i, (x, y) in enumerate([(250, 520), (860, 640), (230, 1150), (880, 1180)]):
            ph = (t * 1.6 + i * 0.27) % 1
            blit(fr, A.sparkle, x, y, scale=math.sin(math.pi * ph) * 0.9, rot=ph * 90)

    def shapes_overlay(self, fr, t):
        A = self.A
        # chunk "bentuk cermin": ~30.24 - 31.7
        shapes = [
            (fx.arch_path(170, 270, 150, 260), st.PINK, A.l_shapes[0], 170),
            (fx.ellipse_path(410, 400, 80, 130), st.BUTTER, A.l_shapes[1], 410),
            (fx.ellipse_path(650, 400, 110, 110), st.MINT, A.l_shapes[2], 650),
            (fx.rect_path(890, 400, 130, 260), st.LILAC, A.l_shapes[3], 890),
        ]
        for i, (pts, col, lab, lx) in enumerate(shapes):
            t0 = 30.25 + i * 0.32
            if t < t0:
                continue
            a = fade(t, t0, 32.2, 0.01, 0.15)
            fx.draw_doodle(fr, pts, clamp01((t - t0) / 0.35), color=col, width=11, shadow=(255, 255, 255), alpha=a)
            blit(fr, lab, lx, 590, scale=pop(t, t0 + 0.2, 0.25), rot=wobble(t, i, 4), alpha=a)

    def follow_overlay(self, fr, t):
        A = self.A
        # "follow" word at ~38.15, tap at 38.55
        if t >= 38.1:
            sc = pop(t, 38.1, 0.3)
            if 38.5 <= t < 38.62:
                sc *= 0.92
            spr = A.l_follow if t < 38.62 else A.l_following
            blit(fr, spr, W / 2, 1010, scale=sc * (1 + (0.08 * (1 - clamp01((t - 38.62) / 0.2)) if t >= 38.62 else 0)), rot=-3)
            # finger cursor ripple
            if 38.35 <= t < 39.1:
                rp = clamp01((t - 38.55) / 0.35)
                if t >= 38.55:
                    r = int(30 + 90 * rp)
                    cv2.circle(fr, (W // 2 + 120, 1040), r, (255, 255, 255), max(1, int(8 * (1 - rp))), cv2.LINE_AA)
                blit(fr, A.heart_s, W / 2 + 140 + (1 - clamp01((t - 38.35) / 0.2)) * 80, 1065, scale=0.9)
        if 36.5 <= t < 38.1:
            blit(fr, A.palette_s, 210, 1020, scale=pop(t, 36.5, 0.3), rot=-10 + wobble(t, 1, 4), alpha=fade(t, 36.5, 39.1, 0.05, 0.1))
            blit(fr, A.brush, 860, 980, scale=pop(t, 36.65, 0.3), rot=30 + wobble(t, 2, 4))

    def like_overlay(self, fr, t):
        A = self.A
        if t >= 40.1:
            u = t - 40.1
            sc = pop(t, 40.1, 0.35) * (1 + 0.06 * math.sin(u * 12) * math.exp(-u * 3))
            blit(fr, A.heart_big, W / 2, 900, scale=sc, rot=-6)
            blit(fr, A.l_like, W / 2 + 10, 1130, scale=pop(t, 40.3, 0.3), rot=-4)
            rnd = random.Random(9)
            for i in range(9):
                st0 = 40.15 + i * 0.07
                if t < st0:
                    continue
                v = t - st0
                ang = rnd.uniform(-2.6, -0.5)
                dist = 220 + 260 * ease_out_cubic(v / 0.9)
                x = W / 2 + math.cos(ang) * dist
                y = 900 + math.sin(ang) * dist - 90 * v
                blit(fr, A.heart_s, x, y, scale=0.6 + 0.5 * rnd.random(), rot=rnd.uniform(-20, 20), alpha=clamp01(1 - v / 1.0))

    def ep2_overlay(self, fr, t):
        A = self.A
        if t >= 44.45:
            blit(fr, A.l_ep2, 520, 330 + bob(t, 0, 5), scale=pop(t, 44.45, 0.35), rot=-5)
            for i, (x, y, tt) in enumerate([(880, 250, 44.75), (960, 330, 44.95), (820, 420, 45.15)]):
                if t >= tt:
                    blit(fr, A.l_q, x, y + 10 * math.sin(t * 6 + i), scale=pop(t, tt, 0.25) * (0.8 - 0.15 * i),
                         rot=12 + 10 * math.sin(t * 5 + i))
            blit(fr, A.btn_mint, 140, 300, scale=pop(t, 44.6) * 0.8, rot=wobble(t, 3, 10))

    def end_overlay(self, fr, lt, t):
        A = self.A
        fx.light_leak(fr, t, 0.25, seed=3)
        blit(fr, A.e_arch_l, 215, 990, scale=pop(lt, 0.45, 0.36), rot=-9 + wobble(t, 1, 1))
        blit(fr, A.e_arch_r, 865, 990, scale=pop(lt, 0.6, 0.36), rot=9 + wobble(t, 2, 1))
        blit(fr, A.e_arch_c, 540, 920 + bob(t, 0, 4), scale=pop(lt, 0.3, 0.38), rot=wobble(t, 0, 1))
        blit(fr, A.bow, 540, 560 + bob(t, 0, 4), scale=pop(lt, 1.0, 0.3), rot=wobble(t, 3, 4))
        blit(fr, A.e_see, 540, 255, scale=pop(lt, 0.35, 0.3), rot=-2)
        blit(fr, A.e_ep, 540, 385 + bob(t, 4, 3), scale=pop(lt, 0.55, 0.36), rot=-3)
        blit(fr, A.e_follow, 540, 1385, scale=pop(lt, 1.1, 0.3), rot=-2)
        blit(fr, A.e_magic, 540, 1500 + bob(t, 5, 3), scale=pop(lt, 1.4, 0.36), rot=-3)
        blit(fr, A.e_thanks, 540, 1650, alpha=clamp01((lt - 2.0) / 0.4))
        blit(fr, A.btn_pink, 935, 250, scale=pop(lt, 0.7), rot=wobble(t, 6, 10))
        blit(fr, A.btn_star, 130, 430, scale=pop(lt, 0.8), rot=wobble(t, 7, 10))
        blit(fr, A.btn_mint, 960, 1560, scale=pop(lt, 1.6) * 0.9, rot=wobble(t, 8, 10))
        blit(fr, A.daisy, 115, 1520, scale=pop(lt, 1.7) * 0.9, rot=wobble(t, 9, 10))
        # butterflies
        for i, (bx0, by0, spd, sp) in enumerate([(-100, 1300, 0.28, A.butterfly), (1180, 1250, -0.22, A.butterfly2)]):
            u = lt * spd
            bx = bx0 + u * 1300 * (1 if spd > 0 else 1)
            by = by0 - abs(u) * 500 + 40 * math.sin(lt * 3 + i)
            flap = 0.35 + 0.65 * abs(math.cos(lt * 12 + i))
            blit(fr, sp, bx, by, scale=0.9, sx=0.9 * flap, rot=-10 if spd > 0 else 10)
        for i, (x, y) in enumerate([(300, 640), (800, 640), (540, 1270), (90, 1100), (990, 1150)]):
            ph = (t * 0.9 + i * 0.23) % 1
            blit(fr, A.sparkle if i % 2 else A.sparkle_y, x, y, scale=math.sin(math.pi * ph) * 0.9, rot=ph * 90)
        if t > TOTAL - 0.6:
            fx.flash(fr, clamp01((t - (TOTAL - 0.6)) / 0.6), color=(255, 247, 240))

    # ---------------- transitions
    def transitions(self, fr, t):
        for tt, (kind, col) in TRANS.items():
            d = t - tt
            if kind == "glint" and -0.2 <= d <= 0.25:
                fx.glint(fr, clamp01((d + 0.2) / 0.45), strength=0.9)
                fx.flash(fr, 0.55 * math.exp(-((d / 0.06) ** 2)))
            elif kind == "flash" and -0.05 <= d <= 0.15:
                fx.flash(fr, 0.6 * math.exp(-((d / 0.06) ** 2)))
            elif kind == "brush":
                if -BRUSH_HALF <= d < 0:
                    self.A.brush_wipe(col).cover(fr, (d + BRUSH_HALF) / BRUSH_HALF)
                elif 0 <= d < BRUSH_HALF:
                    self.A.brush_wipe(col).reveal(fr, d / BRUSH_HALF)

    def punch(self, fr, t):
        """zoom punch for incoming 'punch' cuts (applied before overlays)."""
        for tt, (kind, _) in TRANS.items():
            d = t - tt
            if kind == "punch" and 0 <= d < 0.22:
                z = 1 + 0.12 * (1 - ease_out_cubic(d / 0.22))
                fr[:] = fx.camera(fr, z)
                if d < 0.07:
                    fx.flash(fr, 0.25)
            if kind == "punch" and -0.08 <= d < 0:
                z = 1 + 0.06 * ((d + 0.08) / 0.08)
                fr[:] = fx.camera(fr, z)

    # ---------------- global overlays
    def global_overlay(self, fr, t, s):
        if 3.75 <= t < 53.6 and s["name"] not in ("portfolio", "hello"):
            a = fade(t, 3.75, 53.6, 0.4, 0.3)
            blit(fr, self.A.tag, 40, 150, anchor=(0.0, 0.5), alpha=a, rot=-2)

    def render_frame(self, fi):
        t = fi / FPS
        _, s = self.seg_at(t)
        fr = self.base(s, t)
        fr = np.ascontiguousarray(fr)
        self.punch(fr, t)
        fr = fx.grade(fr, fi, bloom=0.20 if s["kind"] != "end" else 0.0, sat=s["kind"] != "end")
        if s.get("blurbg"):
            lt = t - s["t0"]
            k = clamp01(lt / 0.3)
            if k > 0:
                small = cv2.resize(fr, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), 2.5)
                big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
                wash = np.empty_like(big)
                wash[:] = (255, 226, 236)
                big = cv2.addWeighted(big, 0.72, wash, 0.28, 0)
                cv2.addWeighted(big, 0.85 * k, fr, 1 - 0.85 * k, 0, dst=fr)
        if s["kind"] != "end":
            fx.light_leak(fr, t, 0.14, seed=1)
        self.overlay(fr, s, t)
        self.global_overlay(fr, t, s)
        if s["name"] != "end":
            self.cap.draw(fr, t, s["name"])
        self.transitions(fr, t)
        return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=TOTAL)
    ap.add_argument("--stills", default="", help="comma separated times -> png stills instead of video")
    ap.add_argument("--stilldir", default=".")
    args = ap.parse_args()

    R = Renderer(args)
    json.dump(sorted(CUES), open(args.cues, "w"), indent=0)
    json.dump({"vo_off": VO_OFF, "vo_end_v3": VO_END_V3, "total": TOTAL,
               "speech": [(c["start"], c["end"]) for c in R.cap.chunks]}, open(args.cues.replace(".json", "_meta.json"), "w"))

    if args.stills:
        for ts in args.stills.split(","):
            t = float(ts)
            fr = R.render_frame(int(round(t * FPS)))
            cv2.imwrite(os.path.join(args.stilldir, f"still_{t:06.2f}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return

    f0, f1 = int(round(args.start * FPS)), int(round(args.end * FPS))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        fr = R.render_frame(fi)
        proc.stdin.write(fr.tobytes())
        if fi % 60 == 0:
            print(f"frame {fi}/{f1}", flush=True)
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    main()
