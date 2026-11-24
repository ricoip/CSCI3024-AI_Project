"""Compare diffusion checkpoints and guidance using aggregate dev metrics."""

import argparse
import sys
from pathlib import Path

import torch

from curate import device_name, file_digest, write_json
from submission import generate

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter_kit"))
from evaluate import check_training_data, compute_metrics, default_reference, sample_submission, save_grid
from catdog_eval.data import DEFAULT_DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--guidance", type=float, nargs="+", default=(1.0, 1.5, 2.5))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--n-per-class", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = device_name(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    reference = default_reference(args.data_root)
    train_report, training = check_training_data(Path("submission"), args.data_root)
    if training is None or train_report["errors"]:
        raise ValueError("comparison requires a rebuildable training manifest")
    manifest_sha = file_digest("submission/manifest.csv")
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config, step = checkpoint["config"], checkpoint["step"]
        if config["manifest_sha256"] != manifest_sha:
            raise ValueError(f"manifest does not match {path}")
        model = generate.Diffusion(config["width"], args.steps).eval()
        model.load_state_dict(checkpoint["state"])
        model.to(device)
        for scale in args.guidance:
            model.guidance_scale = scale
            stem = f"trial-{step:06d}-ddim{args.steps}-cfg{scale:g}"
            print(f"Running {stem} ({args.n_per_class} per class, sampling on {device})", flush=True)
            images, labels, elapsed = sample_submission(
                generate, model, args.n_per_class, args.seed, args.batch_size, device
            )
            metrics = compute_metrics(images, labels, reference, "cpu", args.seed, training)
            save_grid(images, labels, args.out / f"{stem}.png")
            write_json(args.out / f"{stem}.json", {
                "checkpoint_step": step, "ddim_steps": args.steps, "guidance_scale": scale,
                "n_per_class": args.n_per_class, "seed": args.seed,
                "sampling_device": device, "metric_device": "cpu", "metrics": metrics,
                "sampling_seconds_per_1000": elapsed / len(images) * 1000,
            })
            print(f"{stem} | Accuracy {metrics['accuracy']:.3f} | FID {metrics['fid']:.2f} | "
                  f"Recall {metrics['recall']:.3f} | Memorization {metrics['memorization_rate']:.3%}",
                  flush=True)


if __name__ == "__main__":
    main()
