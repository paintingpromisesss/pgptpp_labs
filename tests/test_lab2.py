import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).resolve().parents[1] / 'main'
DATA = BIN.parent / 'docs' / 'Бинарные данные ЛР2'

class SobelTests(unittest.TestCase):
    def run_case(self, filename, expected):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out.data'
            p = subprocess.run([str(BIN)], input=f'{DATA / filename}\n{out}\n', text=True, capture_output=True, timeout=20)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(out.exists(), p.stderr)
            data = out.read_bytes()
            self.assertEqual(data[:8], struct.pack('<ii', 3, 3))
            expected = bytes.fromhex(expected)
            self.assertEqual(len(data[8:]), len(expected))
            self.assertTrue(all(abs(a-b) <= 1 for a,b in zip(data[8:], expected)))

    def test_sobel_color_example(self):
        self.run_case('97.data', '14141400 0c0c0c00 11111100 13131300 15151500 18181800 39393900 14141400 3f3f3f00')

    def test_sobel_center_example(self):
        self.run_case('95.data', 'b5b5b500 ffffff00 b5b5b500 ffffff00 00000000 ffffff00 b5b5b500 ffffff00 b5b5b500')

    def test_invalid_output_reports_error(self):
        p = subprocess.run([str(BIN)], input=f'{DATA / "95.data"}\n/nonexistent-lab2-directory/out\n', text=True, capture_output=True, timeout=20)
        self.assertEqual(p.returncode, 0)
        self.assertIn('ERROR:', p.stderr)

if __name__ == '__main__': unittest.main()
