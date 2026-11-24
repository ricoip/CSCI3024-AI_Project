"""Train the diffusion model from scratch using exactly the manifest's crops."""

import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from curate import device_name, file_digest, write_json
from submission.generate import Diffusion, NOISE_STEPS, NULL_LABEL, noisy_with_velocity

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter_kit"))
from catdog_eval.data import DEFAULT_DATA_ROOT, Pool, pool_dir, rebuild
from catdog_eval.manifest import check_manifest


def save_checkpoint(path, content):
    temp = path.with_suffix(".pt.tmp")
    torch.save(content, temp)
    temp.replace(path)


def loss_for_batch(model, clean, labels, noise, steps):
    alpha = model.alpha_bar[steps, None, None, None]
    noisy, velocity = noisy_with_velocity(clean, noise, alpha)
    return F.mse_loss(model(noisy, steps, labels), velocity)


@torch.no_grad()
def update_ema(ema, model, decay):
    for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
        average.lerp_(parameter.detach(), 1 - decay)


def model_config(args):
    return {key: getattr(args, key) for key in
            ("width", "batch_size", "lr", "label_dropout", "ema_decay", "seed", "warmup")}


def train(args):
    device = torch.device(device_name(args.device))
    if args.width <= 0 or args.width % 8 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("width must be a positive multiple of 8 and batch size must be even")
    if min(args.steps, args.snapshot_every, args.log_every, args.threads) <= 0:
        raise ValueError("steps, snapshot interval, log interval and threads must be positive")
    if not 0 < args.label_dropout < 1:
        raise ValueError("use a positive label-drop probability below 1 for CFG")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1)
    pool = Pool(pool_dir(args.data_root))
    checked = check_manifest(args.manifest, pool)
    if checked["errors"] or not checked["stats"]["cat"] or not checked["stats"]["dog"]:
        raise ValueError(f"training requires a valid manifest with both classes: {checked}")
    manifest_sha = file_digest(args.manifest)
    images, labels = rebuild(checked["rows"], pool)
    by_class = [torch.where(labels == label)[0] for label in (0, 1)]
    images = F.pixel_unshuffle(images.mul_(2).sub_(1), 2).to(device)
    labels = labels.to(device)

    model = Diffusion(width=args.width).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    last = args.run_dir / "resume.pt"
    log_path = args.run_dir / "loss.csv"
    config = dict(model_config(args), manifest_sha256=manifest_sha, device=str(device))
    start = 0
    elapsed_before_resume = 0.0
    if args.resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("resume settings or training manifest differ from this run")
        start = checkpoint["step"]
        elapsed_before_resume = checkpoint["training_seconds"]
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.set_state(checkpoint["rng"])
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            log_path.write_text("\n".join(lines[:1] + [line for line in lines[1:]
                                                      if int(line.split(",")[0]) <= start]) + "\n")
    elif last.exists() or log_path.exists():
        raise FileExistsError(f"{args.run_dir} already exists, use --resume or a new --run-dir")
    if args.steps <= start:
        raise ValueError(f"--steps must exceed last saved step {start}")

    print(f"Training {len(images):,} manifest crops ({', '.join(str(len(x)) for x in by_class)}) "
          f"on {device}, steps {start + 1:,}–{args.steps:,}", flush=True)
    print(f"Denoiser parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    write_json(args.run_dir / "config.json", dict(config, steps=args.steps,
                                                 class_counts=[len(x) for x in by_class]))
    began = time.monotonic()
    running_loss = torch.zeros((), device=device)
    since_log = 0
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(("step", "mean_loss", "learning_rate", "training_seconds"))
        for step in range(start + 1, args.steps + 1):
            # Balanced draws with replacement expose both labels equally without
            # adding anything to the manifest or changing the crop pixels.
            indices = torch.cat([group[torch.randint(len(group), (args.batch_size // 2,), generator=rng)]
                                 for group in by_class]).to(device)
            clean, classes = images[indices], labels[indices]
            times = torch.randint(1, NOISE_STEPS + 1, (args.batch_size,), generator=rng).to(device)
            noise = torch.randn(clean.shape, generator=rng).to(device)
            dropped = (torch.rand(args.batch_size, generator=rng) < args.label_dropout).to(device)
            classes = classes.masked_fill(dropped, NULL_LABEL)
            lr = args.lr * min(1, step / max(1, args.warmup))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, clean, classes, noise, times)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, model, min(args.ema_decay, (step + 1) / (step + 10)))
            running_loss.add_(loss.detach())
            since_log += 1
            save = step % args.snapshot_every == 0 or step == args.steps
            if step % args.log_every == 0 or save:
                value = running_loss.item() / since_log
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                seconds = elapsed_before_resume + time.monotonic() - began
                writer.writerow((step, f"{value:.8f}", f"{lr:.8g}", f"{seconds:.2f}"))
                handle.flush()
                print(f"Step {step:6,d} | v-MSE {value:.5f} | {seconds / 60:.1f} min", flush=True)
                running_loss.zero_()
                since_log = 0
            if save:
                cpu_ema = {key: tensor.cpu() for key, tensor in ema.state_dict().items()}
                metadata = {"step": step, "training_seconds": seconds, "config": config}
                save_checkpoint(args.run_dir / f"ema-{step:06d}.pt", dict(metadata, state=cpu_ema))
                save_checkpoint(last, dict(metadata, ema=cpu_ema, model=model.state_dict(),
                                           optimizer=optimizer.state_dict(), rng=rng.get_state()))


def export(args):
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest = args.submission / "manifest.csv"
    if file_digest(manifest) != trained["config"]["manifest_sha256"]:
        raise ValueError("checkpoint was trained on a different manifest")
    if not 1 <= args.sampling_steps <= NOISE_STEPS or not math.isfinite(args.guidance_scale):
        raise ValueError("invalid DDIM sampling settings")
    save_checkpoint(args.submission / "weights.pt", {
        "width": trained["config"]["width"], "sampling_steps": args.sampling_steps,
        "guidance_scale": args.guidance_scale, "state": trained["state"],
        "training_step": trained["step"], "training_seconds": trained["training_seconds"],
        "manifest_sha256": trained["config"]["manifest_sha256"],
    })
    print(f"Exported step {trained['step']:,}, DDIM {args.sampling_steps}, CFG {args.guidance_scale:g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("fit")
    training.add_argument("--manifest", type=Path, default=Path("submission/manifest.csv"))
    training.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    training.add_argument("--run-dir", type=Path, default=Path("work/train"))
    training.add_argument("--steps", type=int, default=16000)
    training.add_argument("--snapshot-every", type=int, default=2000)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--width", type=int, default=64)
    training.add_argument("--lr", type=float, default=0.0003)
    training.add_argument("--warmup", type=int, default=250)
    training.add_argument("--ema-decay", type=float, default=0.999)
    training.add_argument("--label-dropout", type=float, default=0.15)
    training.add_argument("--seed", type=int, default=3024)
    training.add_argument("--device", default="auto")
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--resume", action="store_true")
    exporting = commands.add_parser("export")
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--submission", type=Path, default=Path("submission"))
    exporting.add_argument("--sampling-steps", type=int, default=32)
    exporting.add_argument("--guidance-scale", type=float, default=1.5)
    args = parser.parse_args()
    if args.command == "fit":
        train(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
