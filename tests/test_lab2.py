import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).resolve().parents[1] / 'main'

class Lab2(unittest.TestCase):
    def run_image(self, w, h, pixels, radius, output=None):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'in.data'
            dst = Path(output) if output else Path(directory) / 'out.data'
            src.write_bytes(struct.pack('<ii', w, h) + bytes(pixels))
            result = subprocess.run([str(BIN)], input=f'{src}\n{dst}\n{radius}\n', text=True, capture_output=True, timeout=30)
            return result, dst.read_bytes() if dst.is_file() else b''

    def test_zero_radius_preserves_rgb(self):
        pixels = [25, 100, 240, 11, 255, 0, 8, 99]
        result, data = self.run_image(2, 1, pixels, 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data, struct.pack('<ii', 2, 1) + bytes([25, 100, 240, 0, 255, 0, 8, 0]), result.stderr)

    def test_assignment_example(self):
        pixels = bytes.fromhex('00000000 00000000 64646400 00000000 00000000')
        expected = bytes.fromhex('00000000 1B1B1B00 2D2D2D00 1B1B1B00 00000000')
        result, data = self.run_image(1, 5, pixels, 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(data), 28, result.stderr)
        self.assertIn('horizontal_ms = ', result.stderr)
        self.assertIn('vertical_ms = ', result.stderr)
        for actual, target in zip(data[8:], expected):
            self.assertLessEqual(abs(actual-target), 1)

    def test_three_lines_do_not_require_end_of_input(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory) / 'in.data', Path(directory) / 'out.data'
            src.write_bytes(struct.pack('<ii', 1, 1) + bytes([1, 2, 3, 0]))
            process = subprocess.Popen([str(BIN)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                process.stdin.write(f'{src}\n{dst}\n1\n')
                process.stdin.flush()
                process.wait(timeout=10)
                self.assertEqual(process.returncode, 0)
                self.assertTrue(dst.exists())
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_large_radius_and_single_pixel(self):
        result, data = self.run_image(1, 1, [123, 51, 237, 4], 1023)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(data), 12, result.stderr)
        for actual, target in zip(data[8:], [123, 51, 237, 0]):
            self.assertLessEqual(abs(actual - target), 1)

    def test_assignment_large_radius(self):
        pixels = bytes.fromhex('00000000 00000000 00000000 00000000 FFFFFF00 00000000 00000000 00000000 00000000')
        expected = bytes.fromhex('02020200 02020200 02020200 02020200 03030300 02020200 02020200 02020200 02020200')
        result, data = self.run_image(3, 3, pixels, 5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(data), 44, result.stderr)
        for actual, target in zip(data[8:], expected):
            self.assertLessEqual(abs(actual - target), 1)

    def test_invalid_output_path_reports_error(self):
        result, _ = self.run_image(1, 1, [1, 2, 3, 0], 1, '/nonexistent-lab2-directory/out')
        self.assertEqual(result.returncode, 0)
        self.assertIn('ERROR:', result.stderr)

    def test_missing_radius_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'in.data'
            src.write_bytes(struct.pack('<ii', 1, 1) + bytes([1, 2, 3, 0]))
            result = subprocess.run([str(BIN)], input=f'{src}\n{directory}/out\n', text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0)
            self.assertIn('ERROR:', result.stderr)

    def test_converter_round_trip(self):
        from PIL import Image
        converter = BIN.parent / 'converter.py'
        with tempfile.TemporaryDirectory() as directory:
            image, data, output = (Path(directory) / name for name in ('in.png', 'image.data', 'out.png'))
            Image.new('RGB', (2, 3), (17, 42, 128)).save(image)
            subprocess.run(['python3', str(converter), 'encode', str(image), str(data)], check=True)
            subprocess.run(['python3', str(converter), 'decode', str(data), str(output)], check=True)
            self.assertEqual(Image.open(output).getpixel((1, 2)), (17, 42, 128))

if __name__ == '__main__':
    unittest.main()
