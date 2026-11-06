import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from anomaly import TileModel, tile_means
from demo import experiment, main

def image(*levels):
    return [[level for level in levels] for _ in range(2)]

class TileTests(unittest.TestCase):
    def test_tile_averages_and_order(self):
        pixels = [[10, 20, 50, 50], [30, 40, 50, 50],
                  [60, 60, 100, 120], [60, 60, 140, 160]]
        self.assertEqual(tile_means(pixels, 2), (25, 50, 60, 130))

    def test_fit_uses_mad_and_floor(self):
        model = TileModel.fit([image(a, b) for a, b in [(0, 40), (10, 40), (20, 40)]], tile=1)
        self.assertEqual(model.centers, (10, 40, 10, 40))
        self.assertEqual(model.scales, (10, 2, 10, 2))
        self.assertEqual(model.score(image(30, 36)), (2, 2, 2, 2))

    def test_outlier_cannot_pull_median(self):
        model = TileModel.fit([image(v) for v in (45, 46, 47, 48, 250)], tile=1)
        self.assertEqual(model.centers, (47, 47))
        self.assertEqual(model.scales, (2, 2))

    def test_separate_calibration_and_strict_threshold(self):
        model = TileModel.fit([image(40, 40)] * 3, tile=1)
        cutoff = model.cutoff([image(40, 44), image(38, 42)])
        self.assertEqual(cutoff, 3)
        self.assertEqual(model.inspect(image(40, 46), cutoff), (3, ()))
        self.assertEqual(model.inspect(image(40, 33), cutoff), (3.5, ((1, 0), (1, 1))))

    def test_invalid_shapes_pixels_and_parameters(self):
        for bad in ([], [[1], [1, 2]], [[True]], [[256]], [[5.2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tile_means(bad, 1)
        for tile in (0, 3, True):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                tile_means(image(10, 20), tile)
        with self.assertRaises(ValueError):
            TileModel.fit([image(10)] * 2, tile=1)
        with self.assertRaises(ValueError):
            TileModel.fit([image(10), image(20, 30), image(20)], tile=1)
        with self.assertRaises(ValueError):
            TileModel.fit([image(10)] * 3, tile=1, scale_floor=float("inf"))
        model = TileModel.fit([image(10)] * 3, tile=1)
        with self.assertRaises(ValueError):
            model.score(image(10, 20))
        with self.assertRaises(ValueError):
            model.cutoff([])
        with self.assertRaises(ValueError):
            model.cutoff([image(10)], margin=-1)
        with self.assertRaises(ValueError):
            model.inspect(image(10), float("nan"))

    def test_demo_separates_controls_and_flags_nuisance(self):
        model, cutoff, cases = experiment()
        self.assertEqual(model.inspect(cases["clean"], cutoff)[1], ())
        self.assertEqual(model.inspect(cases["dent"], cutoff)[1], ((3, 2),))
        self.assertEqual(model.inspect(cases["glare"], cutoff)[1], ((5, 3),))
        self.assertEqual(len(model.inspect(cases["lighting"], cutoff)[1]), 48)

    def test_equal_and_opposite_changes_cancel_a_tile_mean(self):
        clean = [[100, 100], [100, 100]]
        model = TileModel.fit([clean] * 3, tile=2)
        self.assertEqual(model.inspect([[70, 130], [100, 100]], 1), (0, ()))

    def test_cli_writes_binary_pgm_and_does_not_call_review_a_defect(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            path = Path(folder) / "sample.pgm"
            self.assertEqual(main(["glare", "--pgm", str(path)]), 0)
            header = b"P5\n64 48\n255\n"
            data = path.read_bytes()
            self.assertEqual(data[:len(header)], header)
            self.assertEqual(len(data[len(header):]), 64 * 48)
            self.assertIn("glare: REVIEW", output.getvalue())

if __name__ == "__main__":
    unittest.main()
