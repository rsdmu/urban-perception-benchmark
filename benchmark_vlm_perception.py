#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependencies:
    pip install openai tqdm

Environment:
    OPENROUTER_API_KEY=<your key>

Example:
    python benchmark_vlm_perception.py \
        --image-dir ./100_images \
        --output ./vlm_outputs \
        --models openai-o4-mini claude-sonnet \
        --workers 6 --rpm 120 --shuffle --limit 100 --save-raw
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import mimetypes
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from threading import Lock, Semaphore
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(x: Iterable, **_: object) -> Iterable:  # type: ignore
        return x

try:
    from openai import OpenAI  # type: ignore
except ImportError as exc:
    raise SystemExit("pip install openai tqdm") from exc


###############################################################################
# Perception grid                                                             #
###############################################################################

@dataclass
class Dimension:
    name: str
    variables: List[str]
    multiple: bool = False  # True ↔ multiple-choice


GRID: List[Dimension] = [
    Dimension("Space Typology", [
        "Park", "Street", "Square", "Courtyard", "Garden", "Waterfront",
        "Public plaza", "Alley", "Playground", "Not applicable"], multiple=True),
    Dimension("Spatial Configuration", [
        "Open", "Enclosed", "Semi-enclosed", "Structured", "Organic", "Not applicable"]),
    Dimension("Size (visual estimate)", [
        "Small (<500 m²)", "Medium (500–2000 m²)", "Large (>2000 m²)", "Not applicable"]),
    Dimension("Lighting", [
        "Natural lighting", "Artificial lighting", "Well lit", "Poorly lit",
        "Shaded areas", "Not applicable"], multiple=True),
    Dimension("Maintenance", [
        "Clean", "Dirty", "Well maintained", "Neglected", "Recently renovated",
        "Not applicable"]),
    Dimension("Vegetation", [
        "Trees present", "Too much greenery", "Little greenery", "Grass present",
        "Bushes present", "Flower beds present", "No vegetation", "Not applicable"], multiple=True),
    Dimension("Paths", [
        "Paved paths present", "Unpaved paths present", "Wide paths present",
        "Narrow paths present", "Linear paths present", "Curved paths present",
        "Intersecting paths present", "Dead-end paths present", "Not applicable"], multiple=True),
    Dimension("Seating", [
        "Benches present", "Chairs present", "Picnic tables present",
        "Custom seats present", "Movable seats present", "No seating", "Not applicable"], multiple=True),
    Dimension("Built Environment", [
        "Modern buildings present", "Historic buildings present",
        "Residential buildings present", "Commercial buildings present",
        "Mixed-use buildings present", "Vacant lots present", "Not applicable"], multiple=True),
    Dimension("Signage", [
        "Informational signs present", "Decorative signs present",
        "Directional signs present", "Interactive signs present", "No signage",
        "Not applicable"], multiple=True),
    Dimension("Human Presence", [
        "Crowded (>50 people)", "Moderately populated (20–50 people)",
        "Sparsely populated (<20 people)", "Empty", "Not applicable"]),
    Dimension("Types of Activities", [
        "Recreational activities present", "Leisure activities present",
        "Commercial activities present", "Transportation activities present",
        "Cultural activities present", "Social activities present",
        "Sports activities present", "Religious activities present",
        "Not applicable"], multiple=True),
    Dimension("Accessibility Features", [
        "Ramps present", "Handrails present", "Tactile paving present",
        "Elevators present", "Wide entrances present", "Accessible restrooms present",
        "No accessibility features", "Not applicable"], multiple=True),
    Dimension("Visibility", [
        "Clear sight lines", "Obstructed views present", "Panoramic views present",
        "Hidden corners present", "Not applicable"], multiple=True),
    Dimension("Safety Measures", [
        "Surveillance cameras present", "Security personnel present",
        "Safety lighting", "Emergency exits present", "Safety signs present",
        "Fences present", "Walls present", "Not applicable"], multiple=True),
    Dimension("Barriers", [
        "Physical barriers present (fences, walls)",
        "Natural barriers present (rivers, hills)", "No barriers", "Not applicable"]),
    Dimension("Aesthetic Elements", [
        "Bright colours present", "Dark colours present",
        "Monochrome elements present", "Murals present", "Sculptures present",
        "Street art present", "Water features present", "No decorative elements",
        "Not applicable"], multiple=True),
    Dimension("Architectural Style", [
        "Traditional buildings present", "Contemporary buildings present",
        "Eclectic buildings present", "Vernacular buildings present",
        "Post-modern buildings present", "Brutalist buildings present",
        "Not applicable"], multiple=True),
    Dimension("Gathering Points", [
        "Central gathering point present", "Edge gathering points present",
        "Gathering points near monuments present", "Informal gathering points present",
        "No gathering points", "Not applicable"]),
    Dimension("Demographic Diversity", [
        "Varied age groups present", "Ethnic diversity present",
        "Gender diversity present", "Not applicable"], multiple=True),
    Dimension("Design", [
        "Wheelchair-accessible features present", "Braille signage present",
        "Multilingual signs present", "Gender-neutral restrooms present",
        "Adapted play equipment present", "No design features", "Not applicable"], multiple=True),
    Dimension("Weather Conditions", [
        "Sunny", "Rainy", "Snowy", "Cloudy", "Windy", "Foggy", "Not applicable"]),
    Dimension("Temperature Range", [
        "Hot (>30 °C)", "Warm (20–30 °C)", "Cool (10–20 °C)", "Cold (<10 °C)",
        "Not applicable"]),
    Dimension("Noise Levels", [
        "Quiet", "Moderate", "Loud", "Traffic noise present",
        "Construction noise present", "Natural sounds present", "Not applicable"], multiple=True),
    Dimension("Temporal Aspects", [
        "Daytime", "Night", "Weekday", "Weekend", "Seasonal variations",
        "Not applicable"]),
    Dimension("Public Amenities", [
        "Restrooms present", "Water fountains present", "Information kiosks present",
        "Trash bins present", "Play areas present", "Fitness equipment present",
        "Not applicable"], multiple=True),
    Dimension("Economic Activities", [
        "Street vendors present", "Markets present", "Shops present",
        "Cafés present", "No commercial activities", "Not applicable"], multiple=True),
    Dimension("Transport Connectivity", [
        "Public transport access present", "Bicycle lanes present",
        "Pedestrian paths present", "Parking spaces present", "Carpool points present",
        "Not applicable"], multiple=True),
    Dimension("Cultural Elements", [
        "Historic monuments present", "Monuments present",
        "Culturally significant features present", "Public art installations present",
        "Not applicable"], multiple=True),
    Dimension("Sustainability", [
        "Recycling bins present", "Green building features present",
        "Use of renewable energy present (e.g., solar panels)",
        "Water conservation measures present", "Not applicable"], multiple=True),
    Dimension("Overall Impression", [
        "Inviting", "Accessible", "Comfortable", "Inclusive", "Safe and secure",
        "Diverse"], multiple=False),
]

N_DIMS: int = len(GRID)

CSV_HEADERS_MODEL: List[str] = [d.name for d in GRID]
CSV_HEADERS: List[str] = ["Image_ID"] + CSV_HEADERS_MODEL + ["Comments"]


###############################################################################
# Prompt builder                                                              #
###############################################################################

def build_system_prompt() -> str:
    """
    We expect the *model* to return exactly N_DIMS + 1 fields:
    the N_DIMS dimension values, followed by a final "Comments" field.
    We (the script) will insert Image_ID as a separate CSV column.
    """
    lines: List[str] = [
        "You are an expert urban‑perception assessor.",
        f"Return **only** a single CSV line (no header) with exactly "
        f"{N_DIMS + 1} comma‑separated fields:",
        f"  • {N_DIMS} fields for the dimensions listed below, in order,",
        "  • then one final field named Comments.",
        "Do **not** include an Image_ID and do **not** include any text "
        "outside the single CSV line.",
        "",
        "Formatting rules:",
        "- For multiple‑choice dimensions, join selected options with "
        "semicolons, e.g., `OptionA;OptionB` (no spaces around semicolons).",
        "- If a dimension is unclear or not present, write exactly "
        "`Not applicable`.",
        "- Do not add quotes, code fences, or prose.",
        "",
        "Column order and allowed values:",
    ]
    for i, dim in enumerate(GRID, 1):
        flag = " (multiple)" if dim.multiple else " (single)"
        lines.append(f"{i}. {dim.name}{flag}: " + "; ".join(dim.variables))
    lines.append("\nReturn just the CSV line – nothing else.")
    return "\n".join(lines)


SYSTEM_PROMPT: str = build_system_prompt()


###############################################################################
# Utilities                                                                   #
###############################################################################

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def to_data_uri(path: Path) -> str:
    """Convert a local image file to a base64 data URI suitable for OpenRouter."""
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _find_px_subfolders(root: Path) -> List[Path]:
    """Return [p1, p2, ... p10] if present (sorted), otherwise []."""
    folders: List[Path] = []
    for n in range(1, 11):
        p = root / f"p{n}"
        if p.exists() and p.is_dir():
            folders.append(p)
    return folders


def discover_images(root: Path) -> List[Path]:
    """
    Discover images under `root`.

    Accepted layouts:
        A) Flat directory of images
        B) Nested p1…p10 subfolders

    Returns a sorted list of image Paths. Logs a warning if not ~100 images.
    """
    if not root.exists():
        raise FileNotFoundError(root)

    images: List[Path] = []
    p_folders = _find_px_subfolders(root)
    if p_folders:
        for folder in p_folders:
            images.extend(p for p in folder.iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        images.extend(p for p in root.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    if not images:
        raise FileNotFoundError(f"No images found under {root}")

    images = sorted(images, key=lambda p: p.name)
    if len(images) != 100:
        logging.warning("Expected ~100 images, found %d under %s",
                        len(images), root)
    return images


def get_client(api_key: str) -> "OpenAI":  # type: ignore[name-defined]
    """
    Create a client against OpenRouter's OpenAI‑compatible API.
    """
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


MODEL_MAP: Dict[str, str] = {
    "openai-o4-mini": "openai/o4-mini-high",
    "claude-sonnet": "anthropic/claude-3.7-sonnet:thinking",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "grok-2-vision": "x-ai/grok-2-vision-1212",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "qwen2.5-vl": "qwen/qwen2.5-vl-72b-instruct",
    "gpt-4.1": "openai/gpt-4.1",
}


###############################################################################
# Message builder & API call                                                  #
###############################################################################

def build_messages(image_path: Path) -> List[Dict[str, object]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Assess this image using the schema."},
            {"type": "image_url", "image_url": {"url": to_data_uri(image_path)}},
        ]},
    ]


class RateLimiter:
    """
    Simple token-based limiter for Requests Per Minute (RPM).
    Uses a semaphore of size `burst` and sleeps to refill.
    """

    def __init__(self, rpm: int | None):
        self.rpm = rpm
        self._lock = Lock()
        if rpm and rpm > 0:
            # Burst size = max concurrent allowed within ~1 sec window.
            self._burst = max(1, min(rpm // 10, rpm))
            self._sem = Semaphore(self._burst)
            self._interval = 60.0 / max(rpm, 1)
        else:
            self._sem = None  # type: ignore[assignment]
            self._interval = 0.0

    def acquire(self) -> None:
        if self._sem is None:
            return
        self._sem.acquire()

    def release(self) -> None:
        if self._sem is None:
            return
        # Refill with spacing to target RPM
        time.sleep(self._interval)
        self._sem.release()


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    # Remove opening fence (optionally with language)
    lines = s.splitlines()
    if not lines:
        return s
    if lines[0].startswith("```"):
        lines = lines[1:]
    # Remove trailing fence if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def call_model(
    client: "OpenAI",
    model_id: str,
    image: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    rate_limiter: RateLimiter | None,
    temperature: float | None = None,
) -> Tuple[str, str]:
    """
    Returns (clean_text, raw_text). clean_text is stripped of code fences.
    """
    last_err: Exception | None = None
    raw = ""
    for attempt in range(retries):
        try:
            if rate_limiter:
                rate_limiter.acquire()
            # Per‑request timeout via with_options for broad compatibility
            req = client.with_options(timeout=timeout).chat.completions.create(
                model=model_id,
                messages=build_messages(image),
                temperature=temperature,
            )
            raw = (req.choices[0].message.content or "").strip()
            clean = _strip_code_fences(raw)
            logging.debug("%s -> %s", model_id, clean[:160].replace("\n", " "))
            return clean, raw
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = (backoff ** attempt) + random.uniform(0, 0.5)
            logging.warning(
                "Retry %d/%d for %s on %s – %s",
                attempt + 1, retries, model_id, image.name, exc,
            )
            time.sleep(wait)
        finally:
            if rate_limiter:
                rate_limiter.release()
    err_msg = f"ERROR: {last_err}"
    logging.error("Giving up on %s (%s): %s", model_id, image.name, last_err)
    return err_msg, raw


###############################################################################
# CSV parsing & writing                                                       #
###############################################################################

_IMAGE_ID_LIKE = re.compile(r"^(?:image)?\s*[\w\-\.\(\)]+$", re.IGNORECASE)


def _cells_from_csv_text(text: str) -> List[str]:
    """
    Try strict CSV parsing first; if it fails, fall back to simple split.
    """
    try:
        reader = csv.reader(StringIO(text))
        cells = next(reader)
        return [c.strip() for c in cells]
    except Exception:  # noqa: BLE001
        return [c.strip() for c in text.split(",")]


def _normalize_cells(
    image_id: str,
    cells: List[str],
    *,
    n_dims: int,
) -> List[str]:
    """
    Normalize model output into exactly n_dims + 1 cells (dims + Comments).

    - Drop leading Image_ID if the model included it.
    - If too few cells, right‑pad with "".
    - If too many cells, merge the extras into Comments.
    """
    # Drop any empty trailing lines from code-fence parsing
    while cells and cells[-1] == "":
        # Keep a single empty last cell if it's meant to be Comments
        # We'll re-pad below as needed.
        cells.pop()

    # Heuristic: if the first cell equals the known image_id or looks like one,
    # drop it. (Some models ignore the instruction and prepend it anyway.)
    if cells and (cells[0] == image_id or _IMAGE_ID_LIKE.match(cells[0])):
        # Only drop if overall length suggests an extra column
        if len(cells) in (n_dims + 2, n_dims + 3, n_dims + 10):
            cells = cells[1:]

    expected = n_dims + 1  # dims + Comments
    if len(cells) < expected:
        cells = cells + [""] * (expected - len(cells))
    elif len(cells) > expected:
        # Keep first n_dims as dimensions; merge the rest into Comments
        dims, rest = cells[:n_dims], cells[n_dims:]
        merged_comment = ",".join(x for x in rest if x)
        cells = dims + [merged_comment]
    return cells


def parse_row(image_id: str, response: str) -> Dict[str, str]:
    """
    Build a row dict for the global CSV schema.

    If response starts with "ERROR:", put it into Comments.
    Otherwise, ensure we end with exactly N_DIMS + 1 cells (dims + Comments).
    """
    row: Dict[str, str] = {h: "" for h in CSV_HEADERS}
    row["Image_ID"] = image_id

    if not response:
        return row

    resp = response.strip()
    if resp.upper().startswith("ERROR:"):
        row["Comments"] = resp
        return row

    cells = _cells_from_csv_text(resp)
    cells = _normalize_cells(image_id, cells, n_dims=N_DIMS)

    # Assign: first N_DIMS -> model columns; last -> Comments
    for header, value in zip(CSV_HEADERS[1:], cells):
        row[header] = value.strip()
    return row


def write_csv(rows: List[Dict[str, str]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


###############################################################################
# Orchestration                                                               #
###############################################################################

def run_for_model(
    key: str,
    model_id: str,
    images: List[Path],
    client: "OpenAI",
    out_dir: Path,
    *,
    workers: int,
    timeout: float,
    retries: int,
    backoff: float,
    rpm: int | None,
    save_raw: bool,
    temperature: float | None,
    skip_existing: bool,
) -> None:
    """
    Execute the benchmark for a single logical model key.
    """
    logging.info("Starting %s (%s images)", key, len(images))

    model_out_dir = out_dir
    model_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = model_out_dir / f"vlm_{key}_responses.csv"
    jsonl_path = model_out_dir / f"vlm_{key}_raw.jsonl"

    if skip_existing and csv_path.exists():
        logging.info("Skipping %s because %s already exists", key, csv_path)
        return

    rows: List[Dict[str, str]] = []
    rl = RateLimiter(rpm) if rpm else None

    # For JSONL raw capture
    jsonl_fp = jsonl_path.open("w", encoding="utf-8") if save_raw else None

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    call_model,
                    client,
                    model_id,
                    img,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                    rate_limiter=rl,
                    temperature=temperature,
                ): img
                for img in images
            }

            for fut in tqdm(as_completed(futures), total=len(futures), desc=key):
                img = futures[fut]
                clean, raw = fut.result()
                if jsonl_fp is not None:
                    json_line = {
                        "model_key": key,
                        "model_id": model_id,
                        "image_id": img.stem,
                        "raw": raw,
                        "clean": clean,
                        "ts": time.time(),
                    }
                    jsonl_fp.write(json.dumps(json_line, ensure_ascii=False) + "\n")

                rows.append(parse_row(img.stem, clean))

        rows.sort(key=lambda r: r["Image_ID"])
        write_csv(rows, csv_path)
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    logging.info("%s finished → %s", key, csv_path)


###############################################################################
# CLI                                                                         #
###############################################################################

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark VLMs on an urban‑perception schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--image-dir", type=Path, default=Path("./100_images"),
                    help="Directory with images (flat or p1…p10 subfolders)")
    ap.add_argument("--output", type=Path, default=Path("vlm_outputs"),
                    help="Directory to store outputs")
    ap.add_argument("--models", nargs="*", choices=list(MODEL_MAP.keys()),
                    default=list(MODEL_MAP.keys()),
                    help="Logical model keys to run")
    ap.add_argument("--api-key", help="OpenRouter API key (or env OPENROUTER_API_KEY)")
    ap.add_argument("--workers", type=int,
                    default=min(8, (os.cpu_count() or 4)),
                    help="Max worker threads per model")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="Per‑request timeout (seconds)")
    ap.add_argument("--retries", type=int, default=3,
                    help="Retries per request")
    ap.add_argument("--backoff", type=float, default=2.0,
                    help="Exponential backoff base")
    ap.add_argument("--rpm", type=int, default=None,
                    help="Requests per minute limit across threads (approximate)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Optional temperature to pass to models")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle image order before sending")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of images processed")
    ap.add_argument("--seed", type=int, default=2025,
                    help="Random seed (for --shuffle)")
    ap.add_argument("--save-raw", action="store_true",
                    help="Save raw JSONL responses alongside CSV")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a model if its CSV already exists")
    ap.add_argument("--log", default="INFO",
                    help="Logging level (e.g., DEBUG, INFO, WARNING)")
    ap.add_argument("--list-models", action="store_true",
                    help="Print available model keys and exit")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_models:
        print("Available models:")
        for k, v in MODEL_MAP.items():
            print(f"  {k:18s} -> {v}")
        return

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        logging.critical("No API key provided or found in OPENROUTER_API_KEY")
        sys.exit(1)

    try:
        client = get_client(key)
    except Exception as exc:  # noqa: BLE001
        logging.critical("Failed to initialize client: %s", exc)
        sys.exit(1)

    try:
        images = discover_images(args.image_dir)
    except Exception as exc:  # noqa: BLE001
        logging.critical("Image discovery failed: %s", exc)
        sys.exit(1)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(images)

    if args.limit is not None:
        images = images[: max(0, args.limit)]

    # Sanity log
    logging.info("Dimensions in grid: %d", N_DIMS)
    logging.info("CSV schema columns: %d (Image_ID + %d dims + Comments)",
                 len(CSV_HEADERS), N_DIMS)

    for m in args.models:
        try:
            run_for_model(
                m,
                MODEL_MAP[m],
                images,
                client,
                args.output,
                workers=max(1, int(args.workers)),
                timeout=float(args.timeout),
                retries=int(args.retries),
                backoff=float(args.backoff),
                rpm=args.rpm,
                save_raw=bool(args.save_raw),
                temperature=args.temperature,
                skip_existing=bool(args.skip_existing),
            )
        except KeyboardInterrupt:
            logging.warning("Interrupted by user")
            break
        except Exception as exc:  # noqa: BLE001
            logging.exception("Model %s failed: %s", m, exc)

    logging.info("All done – outputs in %s", args.output.resolve())


if __name__ == "__main__":
    main()
