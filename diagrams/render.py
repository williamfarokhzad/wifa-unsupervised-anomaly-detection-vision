"""Regenerate the original experiment diagrams. Run from repository root; requires Pillow."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import W, H, experiment

OUT = Path(__file__).resolve().parent
BG = "#eef5f1"
INK = "#16343c"
SECONDARY = "#405960"
TEAL = "#006957"
AMBER = "#973f15"
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")

def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_ROOT / name) if (FONT_ROOT / name).exists() else name, size)

def label(draw, xy, value, size=28, *, fill=INK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)

def save(image, stem):
    image.save(OUT / f"{stem}.webp", "WEBP", quality=88, method=6)
    reduced = image.resize((600, round(image.height * 600 / image.width)), Image.Resampling.LANCZOS)
    reduced.save(OUT / f"{stem}-600.webp", "WEBP", quality=85, method=6)

def panels(model, cutoff, cases):
    image = Image.new("RGB", (800, 742), BG)
    draw = ImageDraw.Draw(image)
    label(draw, (35, 24), "What triggered a review?", 40, bold=True)
    label(draw, (35, 80), "Orange boxes mark tiles above the cutoff.", 30, fill=SECONDARY)
    titles = ("Clean part", "Dark dent", "Bright glint", "Lamp too bright")
    for i, (case, title) in enumerate(zip(cases.values(), titles)):
        x = 35 + i % 2 * 380
        y = 138 + i // 2 * 301
        label(draw, (x, y), title, 30, bold=True)
        photo = Image.frombytes("L", (W, H), bytes(p for row in case for p in row)).convert("RGB")
        photo = photo.resize((320, 240), Image.Resampling.NEAREST)
        image.paste(photo, (x, y + 43))
        _, flags = model.inspect(case, cutoff)
        for col, row in flags:
            xx, yy = x + col * 40, y + 43 + row * 40
            draw.rectangle((xx, yy, xx + 39, yy + 39), outline="#ffb15c", width=2 if len(flags) > 1 else 4)
    return image

def scores(model, cutoff, cases):
    image = Image.new("RGB", (800, 534), BG)
    draw = ImageDraw.Draw(image)
    label(draw, (35, 24), "One cutoff, four outcomes", 39, bold=True)
    label(draw, (35, 80), "The score has no units.", 31, fill=SECONDARY)
    x0, span = 240, 435
    for i, ((_, case), title) in enumerate(zip(cases.items(), ("Clean", "Dent", "Glint", "Lamp shift"))):
        y = 155 + i * 68
        score, flagged = model.inspect(case, cutoff)
        label(draw, (35, y + 19), title, 29, anchor="lm")
        right = x0 + span * score / 9
        draw.rounded_rectangle((x0, y, right, y + 38), radius=7, fill=AMBER if flagged else TEAL)
        label(draw, (right + 12, y + 19), f"{score:.2f}", 26, bold=True, anchor="lm")
    xx = x0 + span * cutoff / 9
    for y in range(142, 409, 15):
        draw.line((xx, y, xx, y + 9), fill="#243438", width=3)
    draw.line((x0, 427, x0 + span, 427), fill=INK, width=2)
    for val in (0, 3, 6, 9):
        x = x0 + span * val / 9
        draw.line((x, 427, x, 437), fill=INK, width=2)
        label(draw, (x, 444), str(val), 25, anchor="mt")
    label(draw, (35, 487), f"Dashed line: review cutoff {cutoff:.2f}", 30, fill=SECONDARY)
    return image

def main():
    data = experiment()
    save(panels(*data), "anomaly-tile-overlays")
    save(scores(*data), "anomaly-image-scores")

if __name__ == "__main__":
    main()
