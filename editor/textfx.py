"""Text rendering helpers (title + subtitles) with outlines, glow and shadows."""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_DIR = os.environ.get("FONT_DIR", os.path.join(os.path.dirname(__file__), "fonts"))

_font_cache = {}


def font(name, size, variation=None):
    key = (name, size, tuple(sorted(variation.items())) if variation else None)
    if key in _font_cache:
        return _font_cache[key]
    f = ImageFont.truetype(os.path.join(FONT_DIR, name), size, layout_engine=ImageFont.Layout.RAQM)
    if variation:
        axes = f.get_variation_axes()
        vals = []
        for ax in axes:
            nm = ax["name"].decode() if isinstance(ax["name"], bytes) else ax["name"]
            vals.append(variation.get(nm, ax["default"]))
        f.set_variation_by_axes(vals)
    _font_cache[key] = f
    return f


def FRAUNCES(size, wght=420):
    return font("Fraunces-Italic[SOFT,WONK,opsz,wght].ttf", size,
                {"Softness": 100, "Wonky": 1, "Optical size": 144, "Weight": wght})


def FREDOKA(size, wght=650):
    return font("Fredoka[wdth,wght].ttf", size, {"Width": 100, "Weight": wght})


def QUICKSAND(size, wght=500):
    return font("Quicksand[wght].ttf", size, {"Weight": wght})


def HAND(size):
    return font("Gaegu-Bold.ttf", size)


def render(text, f, fill, stroke=0, stroke_fill=None, glow=0, glow_color=None, glow_strength=1.0,
           shadow=None, pad=None, tracking=0):
    """Render text to a tight RGBA image.

    shadow: (dx, dy, blur, (r,g,b,a))
    glow: blur radius of a coloured halo behind the text.
    """
    if pad is None:
        pad = int(stroke + glow * 2.5 + (abs(shadow[0]) + abs(shadow[1]) + shadow[2] * 2 if shadow else 0) + 6)
    if tracking:
        widths = [f.getlength(ch) for ch in text]
        tw = int(sum(widths) + tracking * (len(text) - 1))
    else:
        tw = int(f.getlength(text))
    asc, desc = f.getmetrics()
    W = tw + pad * 2 + stroke * 2
    H = asc + desc + pad * 2 + stroke * 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def draw_text(img, color, sw=0, sf=None):
        d = ImageDraw.Draw(img)
        if tracking:
            x = pad + stroke
            for ch in text:
                d.text((x, pad + stroke), ch, font=f, fill=color, stroke_width=sw, stroke_fill=sf)
                x += f.getlength(ch) + tracking
        else:
            d.text((pad + stroke, pad + stroke), text, font=f, fill=color, stroke_width=sw, stroke_fill=sf)

    base_alpha = Image.new("L", (W, H), 0)
    da = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_text(da, (255, 255, 255, 255), stroke, (255, 255, 255, 255))
    base_alpha = da.split()[3]

    if shadow:
        dx, dy, bl, col = shadow
        a = np.asarray(base_alpha).astype(np.float32)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(a, M, (W, H))
        if bl > 0:
            a = cv2.GaussianBlur(a, (0, 0), bl)
        sh = Image.new("RGBA", (W, H), col[:3] + (0,))
        sh.putalpha(Image.fromarray(np.clip(a * (col[3] / 255.0), 0, 255).astype(np.uint8)))
        layer = Image.alpha_composite(layer, sh)
    if glow:
        a = np.asarray(base_alpha).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), glow) * glow_strength
        gl = Image.new("RGBA", (W, H), (glow_color or fill)[:3] + (0,))
        gl.putalpha(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
        layer = Image.alpha_composite(layer, gl)
    t = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_text(t, fill if len(fill) == 4 else fill + (255,), stroke, (stroke_fill if stroke_fill is None or len(stroke_fill) == 4 else stroke_fill + (255,)))
    layer = Image.alpha_composite(layer, t)
    bbox = layer.getbbox()
    if bbox:
        layer = layer.crop(bbox)
    return layer


def gradient_text(text, f, top, bottom, stroke=0, stroke_fill=None, **kw):
    """Text with vertical gradient fill."""
    solid = render(text, f, (255, 255, 255), stroke=0, pad=stroke + 8)
    w, h = solid.size
    g = np.linspace(0, 1, h)[:, None, None]
    top = np.array(top, np.float32)[None, None, :]
    bottom = np.array(bottom, np.float32)[None, None, :]
    rgb = top * (1 - g) + bottom * g
    rgb = np.repeat(rgb, w, axis=1)
    a = np.asarray(solid.split()[3])
    fillimg = Image.fromarray(np.dstack([rgb.astype(np.uint8), a]))
    return fillimg
