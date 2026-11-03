"""Cute whimsical sticker assets, drawn procedurally with PIL + numpy.

Every function returns an RGBA PIL image (already antialiased).
Palette: pink / butter yellow / mint green, plus cream & gold accents.
"""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

PINK = (255, 150, 196)
HOTPINK = (255, 105, 168)
BLUSH = (255, 214, 230)
LILAC = (238, 176, 232)
YELLOW = (255, 222, 102)
BUTTER = (255, 238, 160)
MINT = (168, 236, 204)
GREEN = (126, 206, 150)
SAGE = (150, 196, 140)
CREAM = (255, 249, 238)
GOLD = (226, 178, 74)
SKY = (170, 214, 245)
WHITE = (255, 255, 255)

SS = 3  # supersampling factor


def _down(img, size):
    return img.resize(size, Image.LANCZOS)


def _np(img):
    return np.asarray(img).astype(np.float32) / 255.0


def _to_img(arr):
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def sticker_finish(img, border=6, border_color=WHITE, shadow=0.28, shadow_off=(5, 7), shadow_blur=7):
    """Add a die-cut white sticker border + soft drop shadow around an RGBA image."""
    pad = border + shadow_blur * 2 + max(abs(shadow_off[0]), abs(shadow_off[1])) + 2
    w, h = img.size
    canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    a = img.split()[3]
    a_big = Image.new("L", canvas.size, 0)
    a_big.paste(a, (pad, pad))
    if border > 0:
        # grow alpha => white cut border
        grown = a_big.point(lambda v: 255 if v > 20 else 0)
        grown = grown.filter(ImageFilter.MaxFilter(border * 2 + 1)).filter(ImageFilter.GaussianBlur(1.2))
    else:
        grown = a_big
    if shadow > 0:
        sh = grown.filter(ImageFilter.GaussianBlur(shadow_blur))
        sh = ImageChops.offset(sh, shadow_off[0], shadow_off[1]).point(lambda v: int(v * shadow))
        shadow_layer = Image.new("RGBA", canvas.size, (90, 40, 70, 0))
        shadow_layer.putalpha(sh)
        canvas = Image.alpha_composite(canvas, shadow_layer)
    if border > 0:
        bl = Image.new("RGBA", canvas.size, border_color + (0,))
        bl.putalpha(grown)
        canvas = Image.alpha_composite(canvas, bl)
    top = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    top.paste(img, (pad, pad))
    canvas = Image.alpha_composite(canvas, top)
    return canvas


def _shade_glossy(mask, color, depth=0.10, gloss=0.65):
    """Turn a supersampled L mask into a puffy glossy 3D shape."""
    w, h = mask.size
    r = max(2, min(w, h) * depth)
    hmap = cv2.GaussianBlur(_np(mask), (0, 0), r)
    gy, gx = np.gradient(hmap)
    lx, ly = -0.6, -0.8
    shade = -(gx * lx + gy * ly) * (min(w, h) * 0.9)
    shade = np.clip(shade, -1, 1)
    base = np.array(color, np.float32) / 255.0
    rgb = base[None, None, :] * (0.97 + 0.22 * shade[..., None])
    # rim darkening near the edge (puffy)
    rim = np.clip(hmap * 2.2, 0, 1)
    rgb = rgb * (0.88 + 0.12 * rim[..., None])
    # specular blob top-left
    spec = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(spec)
    d.ellipse([w * 0.18, h * 0.12, w * 0.52, h * 0.36], fill=int(255 * gloss))
    spec = spec.filter(ImageFilter.GaussianBlur(min(w, h) * 0.05))
    sp = _np(spec)[..., None] * np.clip(hmap, 0, 1)[..., None]
    rgb = rgb * (1 - sp) + sp
    alpha = _np(mask)
    out = np.dstack([np.clip(rgb, 0, 1), alpha])
    return _to_img(out)


def _holes(img, holes, color):
    d = ImageDraw.Draw(img)
    dark = tuple(int(c * 0.55) for c in color)
    for (x, y, r) in holes:
        d.ellipse([x - r, y - r, x + r, y + r], fill=dark + (255,))
        d.ellipse([x - r * 0.75, y - r * 0.45, x + r * 0.55, y + r * 0.85], fill=tuple(int(c * 0.40) for c in color) + (255,))
        d.arc([x - r, y - r, x + r, y + r], 20, 160, fill=(255, 255, 255, 170), width=max(1, int(r * 0.25)))
    return img


def button_flower(size=150, color=PINK, petals=5, dots=False, holes=2):
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    c = S / 2
    pr = S * 0.22
    for i in range(petals):
        a = -math.pi / 2 + i * 2 * math.pi / petals
        x, y = c + math.cos(a) * S * 0.24, c + math.sin(a) * S * 0.24
        d.ellipse([x - pr, y - pr, x + pr, y + pr], fill=255)
    d.ellipse([c - S * 0.26, c - S * 0.26, c + S * 0.26, c + S * 0.26], fill=255)
    img = _shade_glossy(m, color)
    if dots:
        dd = ImageDraw.Draw(img)
        rnd = random.Random(7)
        for _ in range(14):
            a = rnd.uniform(0, 2 * math.pi)
            rr = rnd.uniform(S * 0.08, S * 0.40)
            x, y = c + math.cos(a) * rr, c + math.sin(a) * rr
            if m.getpixel((int(x), int(y))) > 200:
                s = S * 0.022
                dd.ellipse([x - s, y - s, x + s, y + s], fill=(255, 255, 255, 230))
    hr = S * 0.045
    if holes == 2:
        hs = [(c - S * 0.07, c, hr), (c + S * 0.07, c, hr)]
    else:
        hs = [(c - S * 0.06, c - S * 0.06, hr), (c + S * 0.06, c - S * 0.06, hr),
              (c - S * 0.06, c + S * 0.06, hr), (c + S * 0.06, c + S * 0.06, hr)]
    img = _holes(img, hs, color)
    return sticker_finish(_down(img, (size, size)), border=0, shadow=0.30, shadow_off=(3, 5), shadow_blur=5)


def _star_points(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return pts


def button_star(size=150, color=BUTTER):
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    d.polygon(_star_points(S / 2, S * 0.53, S * 0.47, S * 0.24), fill=255)
    # round the corners
    m = m.filter(ImageFilter.GaussianBlur(S * 0.035)).point(lambda v: 255 if v > 110 else 0)
    m = m.filter(ImageFilter.GaussianBlur(1.5))
    img = _shade_glossy(m, color)
    c = S / 2
    hr = S * 0.04
    img = _holes(img, [(c - S * 0.06, c - S * 0.02, hr), (c + S * 0.06, c - S * 0.02, hr),
                       (c - S * 0.06, c + S * 0.10, hr), (c + S * 0.06, c + S * 0.10, hr)], color)
    return sticker_finish(_down(img, (size, size)), border=0, shadow=0.30, shadow_off=(3, 5), shadow_blur=5)


def button_round(size=120, color=MINT):
    S = size * SS
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse([S * 0.04, S * 0.04, S * 0.96, S * 0.96], fill=255)
    img = _shade_glossy(m, color)
    d = ImageDraw.Draw(img)
    c = S / 2
    d.ellipse([c - S * 0.30, c - S * 0.30, c + S * 0.30, c + S * 0.30], outline=tuple(int(v * 0.85) for v in color) + (255,), width=int(S * 0.025))
    hr = S * 0.05
    img = _holes(img, [(c - S * 0.08, c - S * 0.08, hr), (c + S * 0.08, c - S * 0.08, hr),
                       (c - S * 0.08, c + S * 0.08, hr), (c + S * 0.08, c + S * 0.08, hr)], color)
    return sticker_finish(_down(img, (size, size)), border=0, shadow=0.30, shadow_off=(3, 5), shadow_blur=5)


def sparkle(size=90, color=WHITE, glow=True, thin=0.16):
    """4-point twinkle star with soft glow."""
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    c = S / 2
    pts = []
    n = 64
    for i in range(n):
        a = i * 2 * math.pi / n
        # astroid-like concave star
        x = math.cos(a)
        y = math.sin(a)
        r = 0.48 * (abs(x) ** (2 / 3) + abs(y) ** (2 / 3)) ** -1.5 if False else None
        k = abs(math.cos(2 * a))
        rr = S * 0.62 * (thin + (0.48 - thin) * k ** 6)
        pts.append((c + x * rr, c + y * rr))
    d.polygon(pts, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(SS * 0.6))
    core = Image.new("RGBA", (S, S), color + (0,))
    core.putalpha(m)
    if glow:
        g = m.filter(ImageFilter.GaussianBlur(S * 0.05)).point(lambda v: min(255, int(v * 1.3)))
        gl = Image.new("RGBA", (S, S), (255, 246, 200, 0))
        gl.putalpha(g)
        core = Image.alpha_composite(gl, core)
    return _down(core, (size, size))


def heart(size=110, color=HOTPINK, outline=True, glossy=True):
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    pts = []
    for i in range(200):
        t = i / 200 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((S / 2 + x * S / 36, S * 0.47 - y * S / 36))
    d.polygon(pts, fill=255)
    img = _shade_glossy(m, color, depth=0.12, gloss=0.7) if glossy else None
    if img is None:
        img = Image.new("RGBA", (S, S), color + (0,))
        img.putalpha(m)
    img = _down(img, (size, size))
    if outline:
        img = sticker_finish(img, border=5, shadow=0.25, shadow_off=(3, 5), shadow_blur=5)
    return img


def daisy(size=120, petal=WHITE, center=YELLOW, petals=10):
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    c = S / 2
    for i in range(petals):
        a = i * 360 / petals
        p = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        pd = ImageDraw.Draw(p)
        pd.ellipse([c - S * 0.085, c - S * 0.47, c + S * 0.085, c - S * 0.06], fill=petal + (255,),
                   outline=tuple(int(v * 0.88) for v in petal) + (255,), width=SS * 2)
        img = Image.alpha_composite(img, p.rotate(a, resample=Image.BICUBIC, center=(c, c)))
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse([c - S * 0.15, c - S * 0.15, c + S * 0.15, c + S * 0.15], fill=255)
    ctr = _shade_glossy(m, center, depth=0.03, gloss=0.5)
    img = Image.alpha_composite(img, ctr)
    return sticker_finish(_down(img, (size, size)), border=4, shadow=0.22, shadow_off=(2, 4), shadow_blur=4)


def tulip(size=130, color=PINK):
    S = size * SS
    img = Image.new("RGBA", (S, int(S * 1.5)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    d.line([(c, S * 0.55), (c - S * 0.02, S * 1.45)], fill=GREEN + (255,), width=SS * 7)
    d.ellipse([c - S * 0.40, S * 0.85, c - S * 0.02, S * 1.05], fill=GREEN + (255,))
    d.pieslice([c - S * 0.30, S * 0.05, c + S * 0.30, S * 0.65], 0, 180, fill=color + (255,))
    d.polygon([(c - S * 0.30, S * 0.35), (c - S * 0.30, S * 0.05), (c - S * 0.12, S * 0.25), (c, S * 0.02),
               (c + S * 0.12, S * 0.25), (c + S * 0.30, S * 0.05), (c + S * 0.30, S * 0.35)], fill=color + (255,))
    d.line([(c, S * 0.08), (c, S * 0.55)], fill=tuple(int(v * 0.85) for v in color) + (255,), width=SS * 3)
    return sticker_finish(_down(img, (size, int(size * 1.5))), border=4, shadow=0.22)


def butterfly(size=140, c1=PINK, c2=BUTTER, body=(120, 80, 90)):
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    # upper wings
    d.ellipse([c - S * 0.46, S * 0.10, c - S * 0.02, S * 0.52], fill=c1 + (255,), outline=(255, 255, 255, 255), width=SS * 3)
    d.ellipse([c + S * 0.02, S * 0.10, c + S * 0.46, S * 0.52], fill=c1 + (255,), outline=(255, 255, 255, 255), width=SS * 3)
    # lower wings
    d.ellipse([c - S * 0.34, S * 0.46, c - S * 0.03, S * 0.82], fill=c2 + (255,), outline=(255, 255, 255, 255), width=SS * 3)
    d.ellipse([c + S * 0.03, S * 0.46, c + S * 0.34, S * 0.82], fill=c2 + (255,), outline=(255, 255, 255, 255), width=SS * 3)
    for (x, y, r) in [(c - S * 0.25, S * 0.30, S * 0.06), (c + S * 0.25, S * 0.30, S * 0.06),
                      (c - S * 0.18, S * 0.64, S * 0.04), (c + S * 0.18, S * 0.64, S * 0.04)]:
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 220))
    d.rounded_rectangle([c - S * 0.035, S * 0.22, c + S * 0.035, S * 0.80], radius=S * 0.035, fill=body + (255,))
    d.line([(c, S * 0.24), (c - S * 0.12, S * 0.08)], fill=body + (255,), width=SS * 3)
    d.line([(c, S * 0.24), (c + S * 0.12, S * 0.08)], fill=body + (255,), width=SS * 3)
    for x in (c - S * 0.12, c + S * 0.12):
        d.ellipse([x - S * 0.02, S * 0.06, x + S * 0.02, S * 0.10], fill=body + (255,))
    return sticker_finish(_down(img, (size, size)), border=3, shadow=0.20, shadow_off=(2, 4), shadow_blur=4)


def palette(size=200):
    S = size * SS
    m = Image.new("L", (S, int(S * 0.8)), 0)
    d = ImageDraw.Draw(m)
    d.ellipse([0, 0, S, S * 0.8], fill=255)
    d.ellipse([S * 0.12, S * 0.46, S * 0.30, S * 0.64], fill=0)  # thumb hole
    d.ellipse([S * 0.62, S * 0.56, S * 1.1, S * 0.95], fill=0)  # notch
    m = m.filter(ImageFilter.GaussianBlur(SS)).point(lambda v: 255 if v > 128 else 0)
    img = _shade_glossy(m, (250, 236, 214), depth=0.04, gloss=0.35)
    d = ImageDraw.Draw(img)
    blobs = [((0.30, 0.18), HOTPINK), ((0.50, 0.12), YELLOW), ((0.70, 0.20), GREEN),
             ((0.82, 0.38), SKY), ((0.48, 0.34), LILAC), ((0.20, 0.32), (255, 160, 120))]
    rnd = random.Random(3)
    for (fx, fy), col in blobs:
        x, y = fx * S, fy * S
        r = S * 0.065
        pts = []
        for i in range(24):
            a = i / 24 * 2 * math.pi
            rr = r * (1 + 0.18 * math.sin(3 * a + rnd.random() * 6))
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
        d.polygon(pts, fill=col + (255,))
        d.ellipse([x - r * 0.5, y - r * 0.6, x - r * 0.1, y - r * 0.25], fill=(255, 255, 255, 170))
    img = _down(img, (size, int(size * 0.8)))
    return sticker_finish(img, border=5, shadow=0.25)


def brush(size=220, tip=HOTPINK, handle=MINT):
    S = size * SS
    W = int(S * 0.22)
    img = Image.new("RGBA", (W, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = W / 2
    d.rounded_rectangle([c - W * 0.22, S * 0.40, c + W * 0.22, S * 0.98], radius=W * 0.22, fill=handle + (255,))
    d.rectangle([c - W * 0.06, S * 0.45, c - W * 0.01, S * 0.94], fill=(255, 255, 255, 120))
    d.rectangle([c - W * 0.25, S * 0.28, c + W * 0.25, S * 0.42], fill=GOLD + (255,))
    d.rectangle([c - W * 0.20, S * 0.30, c - W * 0.10, S * 0.40], fill=(255, 236, 170, 200))
    pts = [(c - W * 0.24, S * 0.29), (c - W * 0.20, S * 0.12), (c, S * 0.0), (c + W * 0.20, S * 0.12), (c + W * 0.24, S * 0.29)]
    d.polygon(pts, fill=(240, 220, 190, 255))
    d.polygon([(c - W * 0.19, S * 0.13), (c, S * 0.0), (c + W * 0.19, S * 0.13), (c + W * 0.10, S * 0.17), (c - W * 0.10, S * 0.17)], fill=tip + (255,))
    img = _down(img, (int(size * 0.22), size))
    return sticker_finish(img, border=4, shadow=0.22)


def mirror_icon(size=170, frame=GOLD, shape="arch"):
    S = size * SS
    H = int(S * 1.6)
    img = Image.new("RGBA", (S, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def arch(box, fill):
        x0, y0, x1, y1 = box
        r = (x1 - x0) / 2
        d.pieslice([x0, y0, x1, y0 + 2 * r], 180, 360, fill=fill)
        d.rectangle([x0, y0 + r, x1, y1], fill=fill)

    pad = S * 0.07
    if shape == "arch":
        arch([0, 0, S, H], frame + (255,))
        inner = Image.new("RGBA", (S, H), (0, 0, 0, 0))
        di = ImageDraw.Draw(inner)
        x0, y0, x1, y1 = pad, pad, S - pad, H - pad
        r = (x1 - x0) / 2
        di.pieslice([x0, y0, x1, y0 + 2 * r], 180, 360, fill=255)
        di.rectangle([x0, y0 + r, x1, y1], fill=255)
    else:
        d.ellipse([0, 0, S, H], fill=frame + (255,))
        inner = Image.new("RGBA", (S, H), (0, 0, 0, 0))
        ImageDraw.Draw(inner).ellipse([pad, pad, S - pad, H - pad], fill=255)
    m = inner.split()[0]
    g = np.linspace(0, 1, H)[:, None] * np.ones((1, S))
    glass = np.dstack([0.78 + 0.15 * g, 0.90 + 0.06 * g, 0.98 - 0.02 * g, np.ones_like(g)])
    gimg = _to_img(glass)
    # diagonal shine streaks
    sh = Image.new("L", (S, H), 0)
    sd = ImageDraw.Draw(sh)
    sd.polygon([(S * 0.15, H), (S * 0.45, H), (S * 1.0, H * 0.25), (S * 0.85, H * 0.10)], fill=150)
    sd.polygon([(S * 0.52, H), (S * 0.60, H), (S * 1.0, H * 0.48), (S * 1.0, H * 0.38)], fill=120)
    sh = sh.filter(ImageFilter.GaussianBlur(SS * 3))
    white = Image.new("RGBA", (S, H), (255, 255, 255, 0))
    white.putalpha(sh)
    gimg = Image.alpha_composite(gimg, white)
    gimg.putalpha(m)
    img = Image.alpha_composite(img, gimg)
    # gold highlight on frame
    d2 = ImageDraw.Draw(img)
    if shape == "arch":
        d2.arc([S * 0.03, S * 0.03, S * 0.97, S * 0.97], 200, 260, fill=(255, 240, 190, 255), width=SS * 3)
    img = _down(img, (size, int(size * 1.6)))
    return sticker_finish(img, border=5, shadow=0.25)


def bow(size=150, color=PINK):
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    dark = tuple(int(v * 0.82) for v in color) + (255,)
    # tails
    d.polygon([(c - S * 0.05, c), (c - S * 0.28, S * 0.92), (c - S * 0.16, S * 0.86), (c - S * 0.10, S * 0.95), (c + S * 0.02, c)], fill=dark)
    d.polygon([(c + S * 0.05, c), (c + S * 0.28, S * 0.92), (c + S * 0.16, S * 0.86), (c + S * 0.10, S * 0.95), (c - S * 0.02, c)], fill=dark)
    # loops
    d.polygon([(c, c), (S * 0.04, S * 0.22), (S * 0.02, S * 0.62)], fill=color + (255,))
    d.ellipse([S * 0.0, S * 0.20, S * 0.18, S * 0.64], fill=color + (255,))
    d.polygon([(c, c), (S * 0.96, S * 0.22), (S * 0.98, S * 0.62)], fill=color + (255,))
    d.ellipse([S * 0.82, S * 0.20, S * 1.0, S * 0.64], fill=color + (255,))
    d.line([(S * 0.1, S * 0.32), (c - S * 0.1, c - S * 0.02)], fill=(255, 255, 255, 140), width=SS * 4)
    d.line([(S * 0.9, S * 0.32), (c + S * 0.1, c - S * 0.02)], fill=(255, 255, 255, 140), width=SS * 4)
    d.rounded_rectangle([c - S * 0.09, c - S * 0.12, c + S * 0.09, c + S * 0.10], radius=S * 0.05, fill=dark)
    return sticker_finish(_down(img, (size, size)), border=4, shadow=0.22)


def washi(w=220, h=56, color=MINT, pattern="dots", alpha=200, seed=1):
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), color + (alpha,))
    d = ImageDraw.Draw(img)
    if pattern == "dots":
        for x in range(0, W, SS * 22):
            for j, y in enumerate(range(SS * 10, H, SS * 18)):
                xx = x + (SS * 11 if j % 2 else 0)
                r = SS * 3.5
                d.ellipse([xx - r, y - r, xx + r, y + r], fill=(255, 255, 255, 210))
    elif pattern == "stripes":
        for x in range(-H, W, SS * 18):
            d.polygon([(x, H), (x + SS * 8, H), (x + H + SS * 8, 0), (x + H, 0)], fill=(255, 255, 255, 140))
    elif pattern == "grid":
        for x in range(0, W, SS * 14):
            d.line([(x, 0), (x, H)], fill=(255, 255, 255, 120), width=SS)
        for y in range(0, H, SS * 14):
            d.line([(0, y), (W, y)], fill=(255, 255, 255, 120), width=SS)
    # jagged torn ends
    m = Image.new("L", (W, H), 255)
    md = ImageDraw.Draw(m)
    rnd = random.Random(seed)
    tooth = SS * 6
    for side in (0, 1):
        pts = []
        for y in range(0, H + tooth, tooth):
            dx = rnd.randint(0, SS * 7)
            pts.append((dx if side == 0 else W - dx, y))
        if side == 0:
            md.polygon([(0, 0)] + pts + [(0, H)], fill=0)
        else:
            md.polygon([(W, 0)] + pts + [(W, H)], fill=0)
    a = ImageChops.multiply(img.split()[3], m)
    img.putalpha(a)
    return _down(img, (w, h))


def polaroid(photo, width=420, ratio=1.18, bottom=0.20, tint=CREAM):
    """Put a photo in a polaroid-style card. photo: PIL RGB."""
    pw = int(width * 0.88)
    ph = int(pw * ratio)
    ph_img = cover_fit(photo, (pw, ph))
    m = int(width * 0.06)
    W = width
    H = ph + m + int(width * bottom)
    card = Image.new("RGBA", (W, H), tint + (255,))
    card.paste(ph_img, (m, m))
    return sticker_finish(card, border=0, shadow=0.35, shadow_off=(6, 10), shadow_blur=10)


def cover_fit(img, size, focus=(0.5, 0.5)):
    w, h = img.size
    tw, th = size
    s = max(tw / w, th / h)
    nw, nh = int(math.ceil(w * s)), int(math.ceil(h * s))
    r = img.resize((nw, nh), Image.LANCZOS)
    x = int((nw - tw) * focus[0])
    y = int((nh - th) * focus[1])
    return r.crop((x, y, x + tw, y + th))


def arch_mask(w, h, radius=None):
    m = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(m)
    W, H = w * SS, h * SS
    r = W / 2 if radius is None else radius * SS
    d.pieslice([0, 0, W, 2 * r], 180, 360, fill=255)
    d.rectangle([0, r - 1, W, H], fill=255)
    if radius is not None:
        pass
    return m.resize((w, h), Image.LANCZOS)


def arch_frame_photo(photo, w=520, h=860, frame=GOLD, border=16, focus=(0.5, 0.45)):
    """A photo inside an arched gold mirror frame (mirror vibe)."""
    outer = arch_mask(w, h)
    inner = arch_mask(w - border * 2, h - border * 2)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # gold frame with vertical gradient
    g = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    base = np.array(frame, np.float32)[None, None, :] / 255
    shade = 0.82 + 0.35 * np.abs(np.sin(g * 6.0 + 0.5))
    gold = np.dstack([np.clip(base * shade[..., None], 0, 1), np.ones_like(g)])
    gimg = _to_img(gold)
    gimg.putalpha(outer)
    img = Image.alpha_composite(img, gimg)
    ph = cover_fit(photo, (w - border * 2, h - border * 2), focus).convert("RGBA")
    ph.putalpha(inner)
    img.alpha_composite(ph, (border, border))
    # glass shine
    sh = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(sh)
    sd.polygon([(w * 0.05, h * 0.75), (w * 0.22, h * 0.75), (w * 0.95, h * 0.20), (w * 0.85, h * 0.12)], fill=70)
    sd.polygon([(w * 0.30, h * 0.80), (w * 0.36, h * 0.80), (w * 0.95, h * 0.38), (w * 0.95, h * 0.33)], fill=55)
    sh = sh.filter(ImageFilter.GaussianBlur(10))
    inner_full = Image.new("L", (w, h), 0)
    inner_full.paste(inner, (border, border))
    sh = ImageChops.multiply(sh, inner_full)
    wl = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    wl.putalpha(sh)
    img = Image.alpha_composite(img, wl)
    return sticker_finish(img, border=8, border_color=CREAM, shadow=0.35, shadow_off=(6, 12), shadow_blur=14)


def paint_splat(size=160, color=PINK, seed=0):
    S = size * SS
    rnd = random.Random(seed)
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    c = S / 2
    pts = []
    for i in range(40):
        a = i / 40 * 2 * math.pi
        rr = S * 0.22 * (1 + 0.25 * rnd.random() + (0.5 if rnd.random() < 0.18 else 0))
        pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
    d.polygon(pts, fill=255)
    for _ in range(9):
        a = rnd.uniform(0, 2 * math.pi)
        rr = rnd.uniform(S * 0.30, S * 0.46)
        r = rnd.uniform(S * 0.015, S * 0.05)
        x, y = c + math.cos(a) * rr, c + math.sin(a) * rr
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(SS * 2)).point(lambda v: 255 if v > 120 else 0).filter(ImageFilter.GaussianBlur(1))
    img = _shade_glossy(m, color, depth=0.05, gloss=0.45)
    return _down(img, (size, size))


def pill(text_img, fill=CREAM, pad=(26, 12), outline=None):
    w, h = text_img.size
    W, H = w + pad[0] * 2, h + pad[1] * 2
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W * SS - 1, H * SS - 1], radius=H * SS / 2, fill=fill + (255,),
                        outline=(outline + (255,)) if outline else None, width=SS * 3 if outline else 0)
    img = img.resize((W, H), Image.LANCZOS)
    img.alpha_composite(text_img, (pad[0], pad[1]))
    return sticker_finish(img, border=0, shadow=0.25, shadow_off=(3, 5), shadow_blur=5)


def cloud(size=220, color=WHITE):
    S = size * SS
    H = int(S * 0.6)
    m = Image.new("L", (S, H), 0)
    d = ImageDraw.Draw(m)
    for (x, y, r) in [(0.25, 0.62, 0.20), (0.45, 0.42, 0.26), (0.68, 0.50, 0.22), (0.82, 0.68, 0.16), (0.5, 0.72, 0.22)]:
        d.ellipse([(x - r) * S, y * H - r * S, (x + r) * S, y * H + r * S], fill=255)
    d.rectangle([0.2 * S, 0.68 * H, 0.86 * S, 0.92 * H], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(SS))
    img = _shade_glossy(m, color, depth=0.03, gloss=0.1)
    return sticker_finish(_down(img, (size, int(size * 0.6))), border=0, shadow=0.15)
