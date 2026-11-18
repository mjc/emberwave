#!/usr/bin/env python3
"""A harder copy of a benchmark Y4M: the same pictures with fresh noise in
every frame, as a stand-in for the grain and the dithering of a game's frames.

    scripts/chrome_codecs/make_grainy.py IN.y4m OUT.y4m [LUMA_SIGMA]

The noise is Gaussian, seeded by the frame's number, with the standard
deviation LUMA_SIGMA (6 by default) on luma and half of it on chroma, added
to the 8-bit planes and clamped. A codec that predicts from the frame before
cannot predict it. Needs numpy.
"""
import sys

import numpy as np


def main():
    source, target = sys.argv[1], sys.argv[2]
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    with open(source, "rb") as src, open(target, "wb") as out:
        header = src.readline()
        out.write(header)
        fields = dict((f[0], f[1:]) for f in header.decode().split()[1:])
        width, height = int(fields["W"]), int(fields["H"])
        luma = width * height
        chroma = luma // 4
        index = 0
        while True:
            line = src.readline()
            if not line:
                break
            raw = src.read(luma + 2 * chroma)
            if len(raw) < luma + 2 * chroma:
                break
            rng = np.random.default_rng(index)
            planes = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            noise = np.empty_like(planes)
            noise[:luma] = rng.normal(0.0, sigma, luma)
            noise[luma:] = rng.normal(0.0, sigma / 2, 2 * chroma)
            out.write(line)
            out.write(np.clip(np.rint(planes + noise), 0, 255).astype(np.uint8).tobytes())
            index += 1
    print(f"{target}: {index} frames, sigma {sigma}")


if __name__ == "__main__":
    main()
