"""Compare fresh synthetic images against a baseline fitted only on clean images."""
import argparse
from pathlib import Path
from random import Random

from anomaly import TileModel

W, H = 64, 48
CASES = {
    "clean": "new clean part",
    "dent": "dark 4 by 4 pixel dent",
    "glare": "bright 4 by 4 pixel glint on a clean part",
    "lighting": "clean part with a brighter lamp",
}

def plate(seed, *, offset=0, patch=None):
    rng = Random(seed)
    rows = [[102 + x // 4 + y // 8 + (15 if 14 <= x < 42 and 11 <= y < 35 else 0)
             + offset + rng.randint(-2, 2) for x in range(W)] for y in range(H)]
    if patch is not None:
        x0, y0, delta = patch
        for y in range(y0, y0 + 4):
            for x in range(x0, x0 + 4):
                rows[y][x] += delta
    return rows

def experiment():
    training = [plate(100 + i, offset=offset) for i, offset in enumerate((-3, -2, -1, 0, 1, 2, 3))]
    calibration = [plate(200 + i, offset=offset) for i, offset in enumerate((-3, -1, 1, 3))]
    model = TileModel.fit(training)
    cutoff = model.cutoff(calibration)
    cases = {
        "clean": plate(300, offset=1),
        "dent": plate(301, patch=(26, 18, -64)),
        "glare": plate(302, patch=(42, 26, 58)),
        "lighting": plate(303, offset=12),
    }
    return model, cutoff, cases

def save_pgm(path, image):
    path.write_bytes(f"P5\n{W} {H}\n255\n".encode("ascii") + bytes(p for row in image for p in row))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--pgm", type=Path, help="optional output image (binary grayscale PGM)")
    args = parser.parse_args(argv)
    model, cutoff, cases = experiment()
    image = cases[args.case]
    maximum, flagged = model.inspect(image, cutoff)
    verdict = "REVIEW" if flagged else "NO ALERT"
    print(f"{args.case}: {verdict}; max={maximum:.2f}; cutoff={cutoff:.2f}; flagged={len(flagged)}")
    if flagged:
        print(f"first flagged tile (column,row): {flagged[0]}")
    if args.pgm is not None:
        try:
            save_pgm(args.pgm, image)
        except OSError as exc:
            parser.error(f"could not write PGM: {exc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
