import struct
import random
from fractions import Fraction
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'main'

class ClassificationTests(unittest.TestCase):
    def classify(self, data, samples):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'in.data'
            dst = Path(tmp) / 'out.data'
            src.write_bytes(data)
            proc = subprocess.run([str(BIN)], input=f'{src}\n{dst}\n{samples}', text=True, capture_output=True, timeout=30)
            return proc, dst.read_bytes() if dst.exists() else b''

    def test_course_example(self):
        data = (ROOT / 'docs/Бинарные данные ЛР3/04.data').read_bytes()
        proc, result = self.classify(data, '2\n4 1 2 1 0 2 2 2 1\n4 0 0 0 1 1 1 2 0\n')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result, bytes.fromhex('03000000 03000000 A2DF4C01 F7C9FE00 9ED84501 B4E85301 99D14D01 92DD5601 A9E04C01 F7D1FA00 D4D0E900'), proc.stderr)

    def test_equal_distance_selects_first_class_and_preserves_rgb(self):
        data = struct.pack('<ii', 3, 1) + bytes([0,0,0,99, 2,0,0,88, 1,0,0,77])
        proc, result = self.classify(data, '2\n1 0 0\n1 1 0\n')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result, data[:8] + bytes([0,0,0,0, 2,0,0,1, 1,0,0,0]), proc.stderr)

    def test_fractional_means_tie_selects_first_class(self):
        data = struct.pack('<ii', 3, 1) + bytes([1,0,0,0, 2,0,0,0, 0,0,0,0])
        proc, result = self.classify(data, '2\n3 1 0 2 0 2 0\n3 1 0 1 0 2 0\n')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result[11], 0, proc.stderr)

    def test_second_course_example(self):
        data = (ROOT / 'docs/Бинарные данные ЛР3/09.data').read_bytes()
        samples = '5\n4 5 0 0 2 6 1 1 1\n6 2 0 7 1 1 0 1 2 6 0 4 0\n4 3 0 3 1 0 1 0 0\n4 0 3 6 2 5 2 7 2\n9 6 4 5 1 7 0 2 1 2 3 4 1 1 5 3 3 2 6\n'
        labels = [2,1,1,2,1,0,1,4,2,0,4,2,4,4,0,1,0,1,0,0,2,3,3,3,3,0,4,4,2,0,1,2,1,3,2,0,3,2,4,2,3,4,2,2,3,1,1,1,2,2,4,4,1,0,1,0,4,3,3,4,4,0,3,4]
        proc, result = self.classify(data, samples)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(result), len(data), proc.stderr)
        self.assertEqual(result[:8], data[:8])
        self.assertEqual(list(result[11::4]), labels)
        for i in range(8, len(data), 4):
            self.assertEqual(result[i:i+3], data[i:i+3])

    def test_invalid_sample_coordinate(self):
        data = struct.pack('<ii', 1, 1) + bytes([1,2,3,0])
        proc, result = self.classify(data, '1\n1 1 0\n')
        self.assertEqual(proc.returncode, 0)
        self.assertIn('ERROR:', proc.stderr)
        self.assertEqual(result, b'')

    def test_large_image_dimension(self):
        for w, h in ((65536, 1), (1, 65536)):
            with self.subTest(w=w, h=h):
                pixel = bytes([20, 30, 40, 99])
                data = struct.pack('<ii', w, h) + pixel * (w * h)
                proc, result = self.classify(data, '1\n1 0 0\n')
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(result, data[:8] + bytes([20, 30, 40, 0]) * (w * h), proc.stderr)

    def test_matches_exact_reference_with_32_classes(self):
        rng = random.Random(7)
        w, h = 17, 9
        pixels = [tuple(rng.randrange(256) for _ in range(4)) for _ in range(w*h)]
        data = struct.pack('<ii', w, h) + bytes(c for pixel in pixels for c in pixel)
        samples, centers = [], []
        for _ in range(32):
            indexes = [rng.randrange(w*h) for _ in range(rng.randrange(1, 8))]
            samples.append(str(len(indexes)) + ' ' + ' '.join(f'{i % w} {i // w}' for i in indexes))
            centers.append([Fraction(sum(pixels[i][c] for i in indexes), len(indexes)) for c in range(3)])
        proc, result = self.classify(data, '32\n' + '\n'.join(samples) + '\n')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(result), len(data), proc.stderr)
        for i, p in enumerate(pixels):
            distances = [sum((p[c] - center[c]) ** 2 for c in range(3)) for center in centers]
            self.assertEqual(result[8 + i*4:11 + i*4], bytes(p[:3]))
            self.assertEqual(result[11 + i*4], distances.index(min(distances)))

if __name__ == '__main__': unittest.main()
