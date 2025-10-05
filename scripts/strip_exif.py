import argparse
from pathlib import Path
from PIL import Image

def strip_one(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.mode in ("P","RGBA","LA"):
            im = im.convert("RGB")
        im.save(dst, format="JPEG", quality=92, optimize=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src = Path(a.src); dst = Path(a.dst)
    for p in src.rglob("*"):
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
            out = dst / p.relative_to(src)
            out = out.with_suffix(".jpg")
            strip_one(p, out)
            print("stripped", p, "->", out)

if __name__ == "__main__":
    main()
