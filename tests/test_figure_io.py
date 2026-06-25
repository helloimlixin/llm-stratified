import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TestFigureIO(unittest.TestCase):
    def test_save_figure_writes_pdf_companion_for_png(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from fiber.figure_io import companion_pdf_path, save_figure

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "diagnostic.png"
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            save_figure(fig, out_path, dpi=72)
            plt.close(fig)

            self.assertTrue(out_path.exists())
            self.assertTrue(companion_pdf_path(out_path).exists())


if __name__ == "__main__":
    unittest.main()
