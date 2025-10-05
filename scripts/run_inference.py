# See paper Appendix for the full prompt and parsing protocol.
# This is a skeleton; plug in your model provider.

import os, base64, csv, argparse
from pathlib import Path

DIMENSIONS = [str(i) for i in range(1,31)]  # placeholder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    imgs = sorted(Path(args.images).rglob("*.jpg"))
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID"] + [f"col_{i}" for i in range(1,31)] + ["Comments"])
        for img in imgs:
            w.writerow([img.stem] + ["Not applicable"]*30 + [""])

if __name__ == "__main__":
    main()
