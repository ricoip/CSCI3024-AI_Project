"""Replay the saved curation run and export individual annotated examples.

Run with the project's ptrc environment. The PNGs and values.json are written
beside this script. Amber marks a body box, blue a head box, and green a crop.
"""

import csv
import json
import os
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

from PIL import ImageDraw

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT))
from curate import (CLASSES, DuplicateFilter, Pool, crop, difference_hash,
                    file_digest, overlap_fraction, pool_dir, square_for_head)


# Stage name, rejected pool ID, retained pool ID.
EXAMPLES = (
    ("scan", "s000008", "s001408"),
    ("body_confidence", "s000132", "s000174"),
    ("single_head", "s002931", "s000058"),
    ("species_match", "s002792", "s001310"),
    ("crop_geometry", "s047880", "s000402"),
    ("clip", "s001264", "s000159"),
    ("visual_review", "s021331", "s000760"),
    ("deduplication", "s006928", "s020610"),
)
IDS = {pool_id for _, rejected, kept in EXAMPLES for pool_id in (rejected, kept)}
MARGINS = ("source_margin", "crop_margin", "species_margin")
COLORS = {"body": "#ffc05a", "head": "#61d4ff", "crop": "#69efab"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["pool_id"]: row for row in csv.DictReader(handle)}


def replay(pool, stats):
    filters = stats["filters"]
    margins = stats["photo_filter"]
    photo_scores = read_rows(ROOT / "work/photo-scores.csv")
    excluded = read_rows(ROOT / "configs/exclusions.csv")
    counts = Counter()
    samples = {}
    candidates = []
    cache = (ROOT / "work/detections.sqlite").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(cache, uri=True)) as db:
        records = db.execute("SELECT pool_id, bodies, heads FROM detections ORDER BY idx")
        for pool_id, raw_bodies, raw_heads in records:
            counts["scanned"] += 1
            bodies = json.loads(raw_bodies)
            body = max(bodies, key=lambda box: box[4], default=None)
            heads = [box for box in json.loads(raw_heads) if box[4] >= filters["second_head_conf"]]
            sample = {"pool_id": pool_id, "body": body, "heads": heads}
            if pool_id in IDS:
                samples[pool_id] = sample
            if body is None:
                sample["rejected_at"] = "scan"
                continue
            if body[4] < filters["body_conf"]:
                sample["rejected_at"] = "body_confidence"
                continue
            counts["body_detection"] += 1
            if len(heads) != 1 or heads[0][4] < filters["head_conf"]:
                sample["rejected_at"] = "single_head"
                continue
            counts["single_head"] += 1
            head = heads[0]
            qualifying = [box for box in bodies if box[4] >= filters["body_conf"]]
            best_overlap = max(qualifying, key=lambda box: overlap_fraction(head, box))
            sample.update(body=best_overlap, overlap=overlap_fraction(head, best_overlap))
            matches = [box for box in qualifying if overlap_fraction(head, box) >= 0.75]
            if not matches or len({box[5] for box in matches}) > 1:
                sample["rejected_at"] = "species_match"
                continue
            counts["species_matched"] += 1
            body = max(matches, key=lambda box: box[4])
            head_width, head_height = head[2] - head[0], head[3] - head[1]
            sample.update(body=body, overlap=overlap_fraction(head, body),
                          head_size_px=(head_width, head_height))
            square = square_for_head(head, *pool.size(pool_id), filters["fill"])
            if square is None:
                sample["rejected_at"] = "crop_geometry"
                continue
            counts["valid_geometry"] += 1
            counts["photo_screened"] += 1
            x, y, side = square
            offset = max(abs((head[0] + head[2]) / 2 - (x + side / 2)),
                         abs((head[1] + head[3]) / 2 - (y + side / 2))) / side
            label = int(body[5])
            scores = photo_scores[pool_id]
            if tuple(int(scores[key]) for key in ("x0", "y0", "size", "label")) != (*square, label):
                raise ValueError(f"Saved CLIP scores do not match the crop for {pool_id}")
            scores = {key: float(scores[key]) for key in MARGINS}
            sample.update(square=square, center_offset=offset, label=CLASSES[label], margins=scores)
            if any(scores[key] < margins[key] for key in MARGINS):
                sample["rejected_at"] = "clip"
                continue
            counts["photo_accepted"] += 1
            if pool_id in excluded:
                sample.update(rejected_at="visual_review", reason=excluded[pool_id]["reason"])
                continue
            counts["after_review_exclusions"] += 1
            sample["joint_confidence"] = body[4] * head[4]
            candidates.append(sample)

    duplicate_filter = DuplicateFilter()
    kept_ids = set()
    for sample in sorted(candidates, key=lambda row: (-row["joint_confidence"], row["pool_id"])):
        pool_id = sample["pool_id"]
        fingerprint = difference_hash(crop(pool.image(pool_id), *sample["square"], 0))
        sample["hash"] = f"{fingerprint:016x}"
        if duplicate_filter.add(fingerprint):
            counts["after_dedup"] += 1
            kept_ids.add(pool_id)
        else:
            sample["rejected_at"] = "deduplication"

    manifest = ROOT / "submission/manifest.csv"
    if (counts != stats["stage_counts"] or file_digest(manifest) != stats["manifest_sha256"]
            or kept_ids != set(read_rows(manifest))):
        raise ValueError("The replay does not match the saved curation run")
    return samples


def draw_example(pool, sample, stage, filename):
    image = pool.image(sample["pool_id"])
    draw = ImageDraw.Draw(image)
    stroke = max(2, round(min(image.size) / 150))
    if sample["body"] is not None:
        draw.rectangle(sample["body"][:4], outline=COLORS["body"], width=stroke)
    if stage not in ("scan", "body_confidence"):
        for head in sample["heads"]:
            draw.rectangle(head[:4], outline=COLORS["head"], width=stroke)
    if stage in ("crop_geometry", "clip", "visual_review", "deduplication") and "square" in sample:
        x, y, side = sample["square"]
        draw.rectangle((x, y, x + side - 1, y + side - 1), outline=COLORS["crop"], width=stroke)
    image.save(OUTPUT / filename)


def main():
    pool = Pool(pool_dir(os.environ.get("CATDOG_DATA", ROOT / "data")))
    stats = json.loads((ROOT / "results/data.json").read_text(encoding="utf-8"))
    samples = replay(pool, stats)
    rows = []
    for index, (stage, rejected_id, kept_id) in enumerate(EXAMPLES, 1):
        rejected = samples[rejected_id]
        kept = samples[kept_id]
        if rejected.get("rejected_at") != stage or "rejected_at" in kept:
            raise ValueError(f"The saved examples for {stage} no longer match the run")
        row = {"step": index, "stage": stage, "filtered": rejected, "kept": kept}
        if stage == "deduplication":
            distance = (int(rejected["hash"], 16) ^ int(kept["hash"], 16)).bit_count()
            if distance > 3 or kept["joint_confidence"] <= rejected["joint_confidence"]:
                raise ValueError("The saved deduplication pair no longer matches the rule")
            row["hash_distance_bits"] = distance
        rows.append(row)

    for row in rows:
        for outcome in ("filtered", "kept"):
            sample = row[outcome]
            filename = f"{row['step']:02d}-{outcome}-{sample['pool_id']}.png"
            draw_example(pool, sample, row["stage"], filename)
            sample.update(image=filename, source_size_px=pool.size(sample["pool_id"]))

    values = {
        "box_colors": COLORS,
        "detection_floor": stats["scan"]["detection_floor"],
        "detection_and_geometry_filters": stats["filters"],
        "clip_thresholds": {key: stats["photo_filter"][key] for key in MARGINS},
        "examples": rows,
    }
    (OUTPUT / "values.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {2 * len(rows)} individual images and values.json to {OUTPUT}")


if __name__ == "__main__":
    main()
