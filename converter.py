#!/usr/bin/env python3
"""Convert common image formats to and from the laboratory RGBA format."""
import argparse
import struct
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('encode', 'decode'))
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
if args.mode == 'encode':
    image = Image.open(args.source).convert('RGBA')
    args.destination.write_bytes(struct.pack('<ii', *image.size) + image.tobytes())
else:
    data = args.source.read_bytes()
    if len(data) < 8:
        parser.error('invalid image header')
    width, height = struct.unpack('<ii', data[:8])
    if width <= 0 or height <= 0 or len(data) != 8 + width * height * 4:
        parser.error('invalid image dimensions or pixel data')
    Image.frombytes('RGBA', (width, height), data[8:]).convert('RGB').save(args.destination)
