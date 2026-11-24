"""Generate a local, reproducible preview from the graded checkpoint."""

import argparse
import hashlib
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from submission.generate import generate, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("cat", "dog", "both"), default="both")
    parser.add_argument("--count", type=int, default=4, help="number of samples per requested class")
    parser.add_argument("--seed", type=int, default=3024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("work/demo.png"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    torch.set_num_threads(6)
    animals = ("cat", "dog")
    classes = (0, 1) if args.label == "both" else (animals.index(args.label),)
    labels = torch.tensor(classes, device=args.device).repeat_interleave(args.count)
    model = load_model(args.device)
    images = generate(model, labels, args.seed).cpu()
    pixels = (images.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()
    cols = min(args.count, 8)
    rows = math.ceil(args.count / cols)
    tile, spacing = 128, 8
    width = cols * (tile + spacing) + spacing
    section_height = rows * (tile + spacing) + 24
    canvas = Image.new("RGB", (width, len(classes) * section_height + spacing), "white")
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(classes):
        top = i * section_height
        draw.text((spacing, top + 5), f"{animals[label].capitalize()}  |  seed {args.seed}", fill="black")
        for j in range(args.count):
            image = Image.fromarray(pixels[i * args.count + j])
            canvas.paste(image.resize((tile, tile), Image.Resampling.NEAREST),
                         (spacing + (j % cols) * (tile + spacing), top + 24 + (j // cols) * (tile + spacing)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"{args.out} | image pixels SHA-256 {hashlib.sha256(pixels.tobytes()).hexdigest()}")


if __name__ == "__main__":
    main()
