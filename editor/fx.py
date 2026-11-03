"""Frame-level effects: grading, blitting, glints, brush wipes, doodles, light leaks."""
import math
import random

import cv2
import numpy as np
from PIL import Image

W, H, FPS = 1080, 1920, 30


# ----------------------------------------------------------------------------- easing
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.9):
    x = clamp01(x)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def pop(t, t0, dur=0.32):
    """0->1 scale with overshoot starting at t0."""
    if t < t0:
        return 0.0
    return ease_out_back((t - t0) / dur)


def fade(t, t0, t1, fin=0.15, fout=0.15):
    if t < t0 or t > t1:
        return 0.0
    a = clamp01((t - t0) / fin) if fin > 0 else 1
    b = clamp01((t1 - t) / fout) if fout > 0 else 1
    return min(a, b)


# ----------------------------------------------------------------------------- sprites
def to_sprite(img):
    """PIL RGBA -> numpy RGBA uint8."""
    return np.asarray(img.convert("RGBA")).copy()


def transform_sprite(spr, scale=1.0, rot=0.0, sx=None):
    """Scale/rotate an RGBA numpy sprite, returns new sprite (expanded canvas)."""
    if scale <= 0.001:
        return None
    sxv = scale if sx is None else sx
    if abs(sxv - 1) < 1e-3 and abs(scale - 1) < 1e-3 and abs(rot) < 0.05:
        return spr
    h, w = spr.shape[:2]
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    # forward matrix: scale (sxv, scale) then rotate
    m00, m01 = sxv * ca, scale * sa
    m10, m11 = -sxv * sa, scale * ca
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32) - [w / 2, h / 2]
    xs = corners[:, 0] * m00 + corners[:, 1] * m01
    ys = corners[:, 0] * m10 + corners[:, 1] * m11
    nw = int(math.ceil(xs.max() - xs.min())) + 2
    nh = int(math.ceil(ys.max() - ys.min())) + 2
    M = np.float32([[m00, m01, nw / 2 - (m00 * w / 2 + m01 * h / 2)],
                    [m10, m11, nh / 2 - (m10 * w / 2 + m11 * h / 2)]])
    # premultiply to avoid dark fringes
    pm = spr.astype(np.float32)
    pm[..., :3] *= pm[..., 3:4] / 255.0
    out = cv2.warpAffine(pm, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a_ = np.maximum(out[..., 3:4], 1e-3)
    out[..., :3] = np.where(out[..., 3:4] > 0.5, out[..., :3] * 255.0 / a_, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def blit(dst, spr, cx, cy, scale=1.0, rot=0.0, alpha=1.0, sx=None, anchor=(0.5, 0.5)):
    """Alpha-blend sprite onto dst (RGB uint8) with its anchor at (cx, cy)."""
    if spr is None or alpha <= 0.003 or scale <= 0.01:
        return
    s = transform_sprite(spr, scale, rot, sx)
    if s is None:
        return
    h, w = s.shape[:2]
    x0 = int(round(cx - w * anchor[0]))
    y0 = int(round(cy - h * anchor[1]))
    x1, y1 = x0 + w, y0 + h
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sub = s[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    a = sub[..., 3:4].astype(np.float32) * (alpha / 255.0)
    region = dst[dy0:dy1, dx0:dx1].astype(np.float32)
    region = region * (1 - a) + sub[..., :3].astype(np.float32) * a
    dst[dy0:dy1, dx0:dx1] = region.astype(np.uint8)


def screen(dst, layer, opacity=1.0):
    """Screen-blend full-size uint8 layer onto dst in place."""
    inv = cv2.multiply(255 - dst, 255 - layer, scale=1 / 255.0)
    out = 255 - inv
    if opacity >= 0.999:
        dst[:] = out
    else:
        cv2.addWeighted(out, opacity, dst, 1 - opacity, 0, dst=dst)


# ----------------------------------------------------------------------------- grade
def make_lut():
    x = np.arange(256, dtype=np.float32) / 255.0
    # soft S-curve with lifted blacks and rolled highlights
    base = 0.055 + 0.92 * (x + 0.10 * np.sin(2 * math.pi * x) * (-0.25))
    base = np.clip(base, 0, 1)
    r = np.clip(base * 1.025 + 0.012, 0, 1)            # warm / pink
    g = np.clip(base * 0.995 + 0.004, 0, 1)
    b = np.clip(base * 0.985 + 0.018 * (1 - x), 0, 1)   # lifted blue shadows (dreamy)
    lut = np.stack([r, g, b], axis=-1)[:, None, :]
    return (lut * 255).astype(np.uint8)


LUT = make_lut()
SAT_LUT = np.clip(np.arange(256) * 1.10, 0, 255).astype(np.uint8)


def _vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    d = np.sqrt(nx ** 2 * 0.9 + ny ** 2 * 0.75)
    v = 1 - 0.28 * np.clip((d - 0.55) / 0.75, 0, 1) ** 1.6
    vr = v * 1.0 + (1 - v) * 0.15   # pinkish vignette
    vg = v
    vb = v * 1.0 + (1 - v) * 0.08
    return (np.dstack([vr, vg, vb]) * 255).astype(np.uint8)


VIGNETTE = _vignette()
_rng = np.random.default_rng(5)
GRAIN = [(_rng.normal(0, 4.5, (H // 2, W // 2)).astype(np.float32)) for _ in range(6)]


def grade(frame, fidx, bloom=0.22, sat=True):
    out = cv2.LUT(frame, LUT)
    if sat:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = cv2.LUT(hsv[..., 1], SAT_LUT)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if bloom > 0:
        small = cv2.resize(out, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        # only bright parts bloom
        small = cv2.subtract(small, np.full_like(small, 90))
        small = cv2.multiply(small, np.full_like(small, 1), scale=1.6)
        big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        screen(out, big, bloom)
    out = cv2.multiply(out, VIGNETTE, scale=1 / 255.0)
    g = cv2.resize(GRAIN[fidx % len(GRAIN)], (W, H), interpolation=cv2.INTER_NEAREST)
    out = np.clip(out.astype(np.float32) + g[..., None], 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- camera
def camera(frame, zoom=1.0, dx=0.0, dy=0.0, rot=0.0):
    if abs(zoom - 1) < 1e-4 and abs(dx) < 0.01 and abs(dy) < 0.01 and abs(rot) < 0.01:
        return frame
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def cover_np(img, zoom=1.0, cx=0.5, cy=0.5):
    """PIL/np image -> 1080x1920 frame, cover-fit with zoom and focus point."""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    h, w = img.shape[:2]
    s = max(W / w, H / h) * zoom
    M = np.float32([[s, 0, W / 2 - cx * w * s], [0, s, H / 2 - cy * h * s]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ----------------------------------------------------------------------------- glint / light
def _glint_band():
    bw, bh = 520, 2900
    x = np.linspace(-1, 1, bw)[None, :]
    core = np.exp(-(x / 0.12) ** 2) * 1.0 + np.exp(-(x / 0.45) ** 2) * 0.35
    thin = np.exp(-((x - 0.55) / 0.04) ** 2) * 0.6
    v = np.clip(core + thin, 0, 1)
    band = np.repeat(v, bh, axis=0)
    col = np.dstack([band * 255, band * 250, band * 240]).astype(np.uint8)
    return col


GLINT = _glint_band()


def glint(dst, p, angle=-28, strength=0.85):
    """Diagonal mirror-shine sweep, p in [0,1] across the frame."""
    if p <= 0 or p >= 1:
        return
    h, w = GLINT.shape[:2]
    cx = -500 + p * (W + 1000)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += cx - w / 2
    M[1, 2] += H / 2 - h / 2
    layer = cv2.warpAffine(GLINT, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    screen(dst, layer, strength * math.sin(math.pi * p) ** 0.5)


def flash(dst, amount, color=(255, 250, 245)):
    if amount <= 0.003:
        return
    layer = np.empty_like(dst)
    layer[:] = color
    cv2.addWeighted(layer, amount, dst, 1 - amount, 0, dst=dst)


def light_leak(dst, t, opacity=0.20, seed=0):
    sw, sh = 90, 160
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    layer = np.zeros((sh, sw, 3), np.float32)
    blobs = [((255, 140, 190), 0.13, 0.9), ((255, 220, 120), 0.09, 1.3), ((190, 178, 255), 0.07, 0.7)]
    for i, (col, sp, rad) in enumerate(blobs):
        ph = seed * 1.7 + i * 2.1
        bx = (0.5 + 0.6 * math.sin(t * sp * 2 * math.pi * 0.5 + ph)) * sw
        by = (0.5 + 0.55 * math.cos(t * sp * 2 * math.pi * 0.37 + ph * 1.3)) * sh
        r = rad * sw * 0.55
        g = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * r * r)))
        layer += g[..., None] * np.array(col, np.float32)[None, None, :]
    layer = np.clip(layer, 0, 255).astype(np.uint8)
    big = cv2.resize(layer, (W, H), interpolation=cv2.INTER_CUBIC)
    screen(dst, big, opacity)


# ----------------------------------------------------------------------------- brush wipe
class BrushWipe:
    """Painterly brush-stroke wipe. cover(p) paints colour over the frame, reveal(p) uncovers."""

    def __init__(self, color, seed=0, n=7, scale=4):
        self.color = np.array(color, np.uint8)
        self.scale = scale
        rnd = random.Random(seed)
        self.sw, self.sh = W // scale, H // scale
        self.strokes_a = self._make(rnd, n, flip=False)
        self.strokes_b = self._make(rnd, n, flip=True)
        tex = np.random.default_rng(seed).normal(0, 1, (self.sh, self.sw)).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
        tex = cv2.resize(tex, (W, H))
        self.tex = tex  # paint texture variation

    def _make(self, rnd, n, flip):
        strokes = []
        band = self.sh / n
        for i in range(n):
            y = band * (i + 0.5) + rnd.uniform(-band * 0.15, band * 0.15)
            ltr = (i % 2 == 0) ^ flip
            xs = np.linspace(-0.12 * self.sw, 1.12 * self.sw, 40)
            if not ltr:
                xs = xs[::-1]
            ph = rnd.uniform(0, 6)
            ys = y + np.sin(np.linspace(0, math.pi * rnd.uniform(1.2, 2.2), 40) + ph) * band * 0.22 - (np.linspace(0, 1, 40) - 0.5) * band * 0.4 * (1 if ltr else -1)
            thick = band * 1.55
            bristles = []
            for _ in range(10):
                off = rnd.uniform(-0.5, 0.5) * thick
                th = max(2, int(thick * rnd.uniform(0.18, 0.32)))
                end = rnd.uniform(0.85, 1.0)
                bristles.append((off, th, end))
            strokes.append((xs, ys, thick, bristles))
        return strokes

    def _mask(self, strokes, p):
        m = np.zeros((self.sh, self.sw), np.uint8)
        n = len(strokes)
        total = p * n
        for i, (xs, ys, thick, bristles) in enumerate(strokes):
            prog = clamp01(total - i * 0.82)
            if prog <= 0:
                continue
            k = max(2, int(len(xs) * prog))
            pts_main = np.stack([xs[:k], ys[:k]], 1)
            cv2.polylines(m, [pts_main.astype(np.int32)], False, 255, int(thick * 0.72), cv2.LINE_AA)
            for off, th, end in bristles:
                kk = max(2, int(k * end))
                pts = np.stack([xs[:kk], ys[:kk] + off], 1).astype(np.int32)
                cv2.polylines(m, [pts], False, 255, th, cv2.LINE_AA)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        return m

    def _paint(self):
        col = np.empty((H, W, 3), np.float32)
        col[:] = self.color
        col += self.tex[..., None] * 9
        return col

    def cover(self, dst, p):
        if p <= 0:
            return
        m = self._mask(self.strokes_a, ease_in_out(p) * 1.15).astype(np.float32)[..., None] / 255.0
        out = dst.astype(np.float32) * (1 - m) + self._paint() * m
        dst[:] = np.clip(out, 0, 255).astype(np.uint8)

    def reveal(self, dst, p):
        if p >= 1:
            return
        m = self._mask(self.strokes_b, ease_in_out(p) * 1.15).astype(np.float32)[..., None] / 255.0
        out = self._paint() * (1 - m) + dst.astype(np.float32) * m
        dst[:] = np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- doodles
def poly_progress(pts, p):
    """Return the first p fraction (by length) of a polyline."""
    pts = np.asarray(pts, np.float32)
    if p >= 1:
        return pts
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1] * clamp01(p)
    k = np.searchsorted(cum, L)
    if k <= 0:
        return pts[:1]
    if k >= len(pts):
        return pts
    f = (L - cum[k - 1]) / max(seg[k - 1], 1e-6)
    last = pts[k - 1] + (pts[k] - pts[k - 1]) * f
    return np.vstack([pts[:k], last[None]])


def draw_doodle(dst, pts, p, color=(255, 255, 255), width=7, shadow=(255, 120, 175), alpha=1.0, closed=False):
    if p <= 0 or alpha <= 0:
        return
    q = poly_progress(pts, p)
    if len(q) < 2:
        return
    q = (q * 4).astype(np.int32)
    if alpha < 1:
        layer = dst.copy()
    else:
        layer = dst
    if shadow is not None:
        cv2.polylines(layer, [q + np.array([0, 4 * 4])], closed and p >= 1, shadow, width + 6, cv2.LINE_AA, shift=2)
    cv2.polylines(layer, [q], closed and p >= 1, color, width, cv2.LINE_AA, shift=2)
    if alpha < 1:
        cv2.addWeighted(layer, alpha, dst, 1 - alpha, 0, dst=dst)


def heart_path(cx, cy, s, n=80):
    pts = []
    for i in range(n + 1):
        t = math.pi + i / n * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * s / 32, cy - y * s / 32))
    return pts


def star_path(cx, cy, r, n=5):
    pts = []
    for i in range(n * 2 + 1):
        rr = r if i % 2 == 0 else r * 0.45
        a = -math.pi / 2 + i * math.pi / n
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return pts


def squiggle_path(x0, x1, y, amp=10, waves=4, n=60):
    xs = np.linspace(x0, x1, n)
    ys = y + np.sin(np.linspace(0, waves * 2 * math.pi, n)) * amp
    return list(zip(xs, ys))


def arrow_path(x0, y0, x1, y1, bend=0.3, n=40):
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = -(y1 - y0), (x1 - x0)
    cx, cy = mx + nx * bend, my + ny * bend
    t = np.linspace(0, 1, n)
    xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
    return list(zip(xs, ys))


def arrow_head(x1, y1, xp, yp, size=26):
    a = math.atan2(y1 - yp, x1 - xp)
    l = (x1 - size * math.cos(a - 0.5), y1 - size * math.sin(a - 0.5))
    r = (x1 - size * math.cos(a + 0.5), y1 - size * math.sin(a + 0.5))
    return [l, (x1, y1), r]


def arch_path(cx, top, w, h, n=60):
    r = w / 2
    pts = [(cx - r, top + h)]
    for i in range(n + 1):
        a = math.pi + i / n * math.pi
        pts.append((cx + math.cos(a) * r, top + r + math.sin(a) * r))
    pts.append((cx + r, top + h))
    pts.append((cx - r, top + h))
    return pts


def ellipse_path(cx, cy, rx, ry, n=70, start=-math.pi / 2):
    return [(cx + math.cos(start + i / n * 2 * math.pi) * rx, cy + math.sin(start + i / n * 2 * math.pi) * ry) for i in range(n + 1)]


def rect_path(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2), (cx - w / 2, cy - h / 2)]
