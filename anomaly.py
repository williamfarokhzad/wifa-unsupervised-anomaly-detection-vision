"""A small, dependency-free baseline for aligned grayscale inspection images."""
from dataclasses import dataclass
from statistics import median
import math

def gray_image(rows):
    if not isinstance(rows, (list, tuple)) or not rows or not isinstance(rows[0], (list, tuple)) or not rows[0]:
        raise ValueError("expected a nonempty rectangular grayscale image")
    w = len(rows[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != w for row in rows):
        raise ValueError("expected a rectangular grayscale image")
    if any(type(p) is not int or not 0 <= p <= 255 for row in rows for p in row):
        raise ValueError("pixels must be integers from 0 to 255")
    return w, len(rows)

def tile_means(rows, tile):
    w, h = gray_image(rows)
    if type(tile) is not int or tile < 1 or w % tile or h % tile:
        raise ValueError("tile must be a positive integer that divides both image dimensions")
    return tuple(sum(rows[y][x] for y in range(top, top + tile)
                                for x in range(left, left + tile)) / tile**2
                 for top in range(0, h, tile) for left in range(0, w, tile))

@dataclass(frozen=True)
class TileModel:
    shape: tuple
    tile: int
    centers: tuple
    scales: tuple

    @classmethod
    def fit(cls, clean_images, *, tile=8, scale_floor=2.0):
        if (isinstance(scale_floor, bool) or not isinstance(scale_floor, (int, float))
                or not math.isfinite(scale_floor) or scale_floor <= 0):
            raise ValueError("scale floor must be finite and positive")
        clean_images = tuple(clean_images)
        if len(clean_images) < 3:
            raise ValueError("provide at least three clean training images")
        shape = gray_image(clean_images[0])
        if any(gray_image(img) != shape for img in clean_images[1:]):
            raise ValueError("training images must share the same registered shape")
        features = [tile_means(img, tile) for img in clean_images]
        cols = tuple(zip(*features))
        centers = tuple(median(col) for col in cols)
        scales = tuple(max(scale_floor, median(abs(x - c) for x in col))
                       for c, col in zip(centers, cols))
        return cls(shape, tile, centers, scales)

    def score(self, image):
        if gray_image(image) != self.shape:
            raise ValueError("image shape differs from the trained grid")
        return tuple(abs(m - c) / s for m, c, s in
                     zip(tile_means(image, self.tile), self.centers, self.scales))

    def cutoff(self, clean_calibration, *, margin=1.0):
        if (isinstance(margin, bool) or not isinstance(margin, (int, float))
                or not math.isfinite(margin) or margin < 0):
            raise ValueError("margin must be finite and nonnegative")
        maxima = [max(self.score(image)) for image in clean_calibration]
        if not maxima:
            raise ValueError("provide at least one separate clean calibration image")
        return max(maxima) + margin

    def inspect(self, image, cutoff):
        if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                or not math.isfinite(cutoff) or cutoff < 0):
            raise ValueError("cutoff must be finite and nonnegative")
        scores = self.score(image)
        cols = self.shape[0] // self.tile
        flagged = tuple((i % cols, i // cols) for i, score in enumerate(scores) if score > cutoff)
        return max(scores), flagged
