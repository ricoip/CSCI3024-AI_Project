"""Detect pet heads in the supplied pool and build an auditable crop manifest."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter_kit"))
from catdog_eval.data import DEFAULT_DATA_ROOT, Pool, crop, pool_dir
from catdog_eval.manifest import check_manifest


FIELDS = ("pool_id", "x0", "y0", "size", "angle", "label")
CLASSES = ("cat", "dog")
DETECTION_FLOOR = 0.20


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def device_name(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def model_boxes(result):
    return result.boxes.data[:, :6].cpu().tolist()


def infer(models, images, classes, image_size, device):
    body, head = models
    options = dict(imgsz=image_size, conf=DETECTION_FLOOR, max_det=12,
                   device=device, verbose=False, rect=False)
    # The body detector establishes species. A head detector trained on the
    # Oxford-IIIT Pet boxes is more suitable for the crop geometry.
    bodies = [model_boxes(result) for result in body.predict(images, classes=classes, **options)]
    candidates = [i for i, boxes in enumerate(bodies) if boxes]
    heads = [[] for _ in images]
    if candidates:
        results = head.predict([images[i] for i in candidates], **options)
        for index, result in zip(candidates, results, strict=True):
            heads[index] = model_boxes(result)
    class_map = {category: label for label, category in enumerate(classes)}
    for boxes in bodies:
        for box in boxes:
            box[5] = class_map[int(box[5])]
    return zip(bodies, heads, strict=True)


def scan(args, pool):
    # Keep dependency configuration and all downloaded weights out of the submission.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "work" / "ultralytics"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import YOLO, __version__ as ultralytics_version

    device = device_name(args.device)
    ids = list(pool.index)
    end = len(ids) if args.limit is None else min(len(ids), args.limit)
    if end <= 0:
        raise ValueError("scan must include at least one image")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    config = json.dumps({
        "body_sha256": file_digest(args.body), "head_sha256": file_digest(args.head),
        "pool_index_sha256": file_digest(pool.folder / "index.csv"),
        "imgsz": args.imgsz, "detection_floor": DETECTION_FLOOR,
        "batch_size": args.batch_size, "device": device,
        "ultralytics_version": ultralytics_version,
    }, sort_keys=True)
    with sqlite3.connect(args.cache, timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS settings (config TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS detections (idx INTEGER PRIMARY KEY, "
                   "pool_id TEXT NOT NULL, bodies TEXT NOT NULL, heads TEXT NOT NULL)")
        settings = db.execute("SELECT config FROM settings").fetchone()
        if settings is None:
            db.execute("INSERT INTO settings VALUES (?)", (config,))
            db.commit()
        elif settings[0] != config:
            raise ValueError("scan settings changed, use a new --cache path")
        start = db.execute("SELECT COALESCE(MAX(idx) + 1, 0) FROM detections").fetchone()[0]
        if start >= end:
            print(f"Already scanned {start:,} images in {args.cache}", flush=True)
            return

        body, head = YOLO(str(args.body)), YOLO(str(args.head))
        classes = [next(i for i, name in body.names.items() if name == animal) for animal in CLASSES]
        print(f"Scanning {start:,}–{end:,} of {len(ids):,} pool images on {device}", flush=True)
        began = time.monotonic()
        next_log = start + 512
        for first in range(start, end, args.batch_size):
            batch_ids = ids[first:min(first + args.batch_size, end)]
            images = [pool.image(pool_id) for pool_id in batch_ids]
            boxes = infer((body, head), images, classes, args.imgsz, device)
            db.executemany("INSERT INTO detections VALUES (?, ?, ?, ?)", [
                (first + j, pool_id, json.dumps(bodies), json.dumps(heads))
                for j, (pool_id, (bodies, heads)) in enumerate(zip(batch_ids, boxes, strict=True))
            ])
            db.commit()
            seen = first + len(batch_ids)
            if seen >= next_log or seen == end:
                speed = (seen - start) / (time.monotonic() - began)
                print(f"{seen:,}/{end:,} images | {speed:.1f} images/s", flush=True)
                next_log = seen + 512


def overlap_fraction(head, body):
    """Fraction of the detected head inside the detected animal."""
    left, top = max(head[0], body[0]), max(head[1], body[1])
    right, bottom = min(head[2], body[2]), min(head[3], body[3])
    area = max(0, right - left) * max(0, bottom - top)
    return area / max(1, (head[2] - head[0]) * (head[3] - head[1]))


def square_for_head(head, width, height, fill):
    left, top, right, bottom = head[:4]
    w, h = right - left, bottom - top
    if w < 32 or h <= 0 or not 0.55 <= w / h <= 1.65:
        return None
    side = math.ceil(max(w, h) / fill)
    if side < 40 or side > min(width, height):
        return None
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x = min(max(0, round(cx - side / 2)), width - side)
    y = min(max(0, round(cy - side / 2)), height - side)
    if max(abs(cx - (x + side / 2)), abs(cy - (y + side / 2))) > side * 0.08:
        return None
    return x, y, side


def difference_hash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = np.packbits((pixels[:, 1:] > pixels[:, :-1]).ravel())
    return int.from_bytes(bits.tobytes(), "big")


class DuplicateFilter:
    """Find hashes within Hamming distance 3 using four 16-bit buckets."""

    def __init__(self):
        self.buckets = defaultdict(set)

    def add(self, value):
        keys = [(i, (value >> (16 * i)) & 0xFFFF) for i in range(4)]
        near = set().union(*(self.buckets[key] for key in keys))
        if any((value ^ previous).bit_count() <= 3 for previous in near):
            return False
        for key in keys:
            self.buckets[key].add(value)
        return True


def build(args, pool):
    counts = Counter()
    candidates = []
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    with sqlite3.connect(args.cache) as db:
        config = json.loads(db.execute("SELECT config FROM settings").fetchone()[0])
        if config["pool_index_sha256"] != file_digest(pool.folder / "index.csv"):
            raise ValueError("detection cache belongs to a different pool index")
        limit = -1 if args.limit is None else args.limit
        records = db.execute("SELECT pool_id, bodies, heads FROM detections ORDER BY idx LIMIT ?", (limit,))
        for pool_id, raw_bodies, raw_heads in records:
            counts["scanned"] += 1
            bodies = json.loads(raw_bodies)
            if not bodies or max(body[4] for body in bodies) < args.body_conf:
                continue
            counts["body_detection"] += 1
            heads = [head for head in json.loads(raw_heads) if head[4] >= args.second_head_conf]
            if len(heads) != 1 or heads[0][4] < args.head_conf:
                continue
            counts["single_head"] += 1
            head = heads[0]
            matches = [body for body in bodies
                       if body[4] >= args.body_conf and overlap_fraction(head, body) >= 0.75]
            if not matches or len({match[5] for match in matches}) > 1:
                continue
            counts["species_matched"] += 1
            body = max(matches, key=lambda candidate: candidate[4])
            square = square_for_head(head, *pool.size(pool_id), args.fill)
            if square is None:
                continue
            counts["valid_geometry"] += 1
            x, y, size = square
            row = dict(zip(FIELDS, (pool_id, x, y, size, 0, int(body[5])), strict=True))
            candidates.append((head[4] * body[4], row))

    counts["photo_screened"] = len(candidates)
    if not args.without_photo_filter:
        candidates = screen_photos(args, pool, candidates)
    counts["photo_accepted"] = len(candidates)
    if args.exclude is not None:
        with args.exclude.open(newline="", encoding="utf-8") as handle:
            excluded = {row["pool_id"] for row in csv.DictReader(handle)}
        candidates = [candidate for candidate in candidates if candidate[1]["pool_id"] not in excluded]
    counts["after_review_exclusions"] = len(candidates)
    kept = []
    unique = DuplicateFilter()
    for _, row in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1]["pool_id"])):
        tile = crop(pool.image(row["pool_id"]), row["x0"], row["y0"], row["size"], row["angle"])
        if unique.add(difference_hash(tile)):
            kept.append(row)
    counts["after_dedup"] = len(kept)
    per_class = Counter(CLASSES[row["label"]] for row in kept)
    if not all(per_class[animal] for animal in CLASSES):
        raise ValueError(f"both classes must be present, found {dict(per_class)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(kept, key=lambda row: row["pool_id"]))
    result = check_manifest(temporary, pool)
    if result["errors"]:
        raise ValueError(result["errors"])
    temporary.replace(args.out)
    stats = {
        "stage_counts": dict(counts), "class_counts": dict(per_class), "scan": config,
        "filters": {key: getattr(args, key) for key in
                    ("body_conf", "head_conf", "second_head_conf", "fill")},
        "manifest_sha256": file_digest(args.out), "warnings": result["warnings"],
    }
    if not args.without_photo_filter:
        stats["photo_filter"] = {"sha256": file_digest(args.photo_weights),
                                 "source_margin": args.source_margin,
                                 "crop_margin": args.crop_margin,
                                 "species_margin": args.species_margin}
    if args.exclude is not None:
        stats["exclusions_sha256"] = file_digest(args.exclude)
    write_json(args.stats, stats)
    print(json.dumps(stats, indent=2), flush=True)


def screen_photos(args, pool, candidates):
    from photo_filter import PhotoFilter

    device = device_name(args.device)
    detector = PhotoFilter(args.photo_weights, device)
    thresholds = (args.source_margin, args.crop_margin, args.species_margin)
    keep = []
    scores_log = []
    print(f"Screening {len(candidates):,} candidate photos on {device}", flush=True)
    for start in range(0, len(candidates), args.photo_batch_size):
        batch = candidates[start:start + args.photo_batch_size]
        rows = [row for _, row in batch]
        originals = [pool.image(row["pool_id"]) for row in rows]
        crops = [crop(image, row["x0"], row["y0"], row["size"], row["angle"], res=128)
                 for image, row in zip(originals, rows, strict=True)]
        margins = detector.margins(originals, crops, [row["label"] for row in rows])
        if args.photo_scores is not None:
            scores_log.extend({**row, **dict(zip(("source_margin", "crop_margin", "species_margin"),
                                                scores, strict=True))}
                              for row, scores in zip(rows, margins, strict=True))
        keep.extend(candidate for candidate, scores in zip(batch, margins, strict=True)
                    if all(score >= threshold for score, threshold in zip(scores, thresholds, strict=True)))
        if start % (args.photo_batch_size * 16) == 0:
            print(f"Screened {start + len(batch):,}/{len(candidates):,}, kept {len(keep):,}", flush=True)
    if args.photo_scores is not None:
        args.photo_scores.parent.mkdir(parents=True, exist_ok=True)
        with args.photo_scores.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS +
                                    ("source_margin", "crop_margin", "species_margin"))
            writer.writeheader()
            writer.writerows(scores_log)
    return keep


def review(args, pool):
    result = check_manifest(args.manifest, pool)
    if result["errors"]:
        raise ValueError(result["errors"])
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for label, animal in enumerate(CLASSES):
        candidates = [row for row in result["rows"] if row["label"] == label]
        selected = rng.sample(candidates, min(args.per_class, len(candidates)))
        if not selected:
            continue
        cols, cell_w, cell_h = 5, 210, 148
        sheet = Image.new("RGB", (cols * cell_w, math.ceil(len(selected) / cols) * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            original = pool.image(row["pool_id"])
            thumb = original.copy()
            thumb.thumbnail((112, 112))
            x, y = (i % cols) * cell_w, (i // cols) * cell_h
            ratio_x, ratio_y = thumb.width / original.width, thumb.height / original.height
            left, top, size = row["x0"], row["y0"], row["size"]
            sheet.paste(thumb, (x, y))
            draw.rectangle((x + left * ratio_x, y + top * ratio_y,
                            x + (left + size) * ratio_x, y + (top + size) * ratio_y),
                           outline="#00ed80", width=2)
            tile = crop(original, left, top, size, row["angle"])
            sheet.paste(tile.resize((88, 88), Image.Resampling.NEAREST), (x + 116, y))
            draw.text((x, y + 115), f"{row['pool_id']}   {animal}   ({left}, {top}, {size})", fill="black")
        sheet.save(args.out / f"{animal}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan", help="cache detections, resuming an interrupted scan")
    scan_parser.add_argument("--cache", type=Path, default=Path("work/detections.sqlite"))
    scan_parser.add_argument("--head", type=Path, default=Path("data/pet_head_yolo26n.pt"))
    scan_parser.add_argument("--body", type=Path, default=Path("data/yolo26n.pt"))
    scan_parser.add_argument("--limit", type=int)
    scan_parser.add_argument("--imgsz", type=int, default=256)
    scan_parser.add_argument("--batch-size", type=int, default=32)
    scan_parser.add_argument("--device", default="auto")
    scan_parser.add_argument("--threads", type=int, default=4)
    build_parser = commands.add_parser("build", help="filter detections and validate the manifest")
    build_parser.add_argument("--cache", type=Path, default=Path("work/detections.sqlite"))
    build_parser.add_argument("--limit", type=int, help="use a prefix of the cached scan for the pilot")
    build_parser.add_argument("--out", type=Path, default=Path("submission/manifest.csv"))
    build_parser.add_argument("--stats", type=Path, default=Path("results/data.json"))
    build_parser.add_argument("--exclude", type=Path, help="CSV of pool IDs excluded during visual review")
    build_parser.add_argument("--body-conf", type=float, default=0.45)
    build_parser.add_argument("--head-conf", type=float, default=0.45)
    build_parser.add_argument("--second-head-conf", type=float, default=0.30)
    build_parser.add_argument("--fill", type=float, default=0.70)
    build_parser.add_argument("--without-photo-filter", action="store_true", help="detector-only pilot")
    build_parser.add_argument("--photo-weights", type=Path, default=Path("data/ViT-B-32.pt"))
    build_parser.add_argument("--photo-batch-size", type=int, default=32)
    build_parser.add_argument("--photo-scores", type=Path)
    build_parser.add_argument("--source-margin", type=float, default=0.010)
    build_parser.add_argument("--crop-margin", type=float, default=0.000)
    build_parser.add_argument("--species-margin", type=float, default=0.010)
    build_parser.add_argument("--device", default="auto")
    review_parser = commands.add_parser("review", help="show a random sample alongside the source images")
    review_parser.add_argument("--manifest", type=Path, default=Path("submission/manifest.csv"))
    review_parser.add_argument("--out", type=Path, default=Path("work/review"))
    review_parser.add_argument("--seed", type=int, default=3024)
    review_parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()
    pool = Pool(pool_dir(args.data_root))
    if args.command == "scan":
        torch.set_num_threads(args.threads)
        scan(args, pool)
    elif args.command == "build":
        build(args, pool)
    else:
        review(args, pool)


if __name__ == "__main__":
    main()
