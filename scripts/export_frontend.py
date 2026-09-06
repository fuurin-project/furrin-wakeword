#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from oww_distill.frontend import extract_tflite_constants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fuurin_frontend.npz"))
    args = parser.parse_args()
    constants = extract_tflite_constants(args.model)
    constants.save(args.output)
    print(
        f"saved {args.output}: window_nonzero={(constants.window != 0).sum()}, "
        f"mel_nonzero={(constants.mel_weights != 0).sum()}"
    )


if __name__ == "__main__":
    main()
