# Project

Class-conditional generation of 32×32 cat and dog heads (course project). The
training set is curated for this project from the supplied LAION pool and described by
`submission/manifest.csv`. The official evaluator is `starter_kit/evaluate.py` and the
submission interface lives in `submission/generate.py`.

Rico chose a **class-conditional diffusion model** and asked to investigate
**classifier-free guidance (CFG)**. Train a single denoiser with randomly dropped
class conditioning. Compare conditional-only sampling with modest CFG scales using
aggregate dev metrics and sample grids. See [Ho and Salimans](https://arxiv.org/abs/2207.12598).
Use the project handout as the source of truth for assignment rules.

## Hard constraints (never violate)

- Training images come only from the raw pool in `data/student/`, and every one is a row of
  a manifest (`pool_id,x0,y0,size,angle,label`, cropped with `catdog_eval.data.crop`).
- Use `data/dev/` for visual inspection and aggregate validation only. Do not use
  dev images, transformed versions, per-image features, or labels to train any model
  or to select pool images. Do not train on any other dataset or images made or
  edited by generative models.
- Pretrained models (detectors, CLIP, ...) only for curating data. The generative model is
  trained from scratch; no pretrained weights, no downloads at generation time.
- Use the evaluation classifier and Inception network only through the official
  evaluator. Never use either to curate data, train, or generate.
- `generate(model, labels, seed)` must be deterministic in `(labels, seed)` and return
  `[N, 3, 32, 32]` floats in `[0, 1]`. Labels are 0 = cat and 1 = dog.
  Use a local seeded random generator and per-image operations so other batch
  members cannot change an image. Generate novel samples without returning copies
  or edits of training images.
- Budget: <= 20M parameters, submission directory <= 100 MB, memorization rate <= 5%.
- Do not modify anything under `starter_kit/`.
- The student personally labels the 200 official audit crops and records three
  independent checks as they happen in `AGENT_LOG.md`, including at least one data
  check. The student creates the presentation slides and delivers the demo.

## Shell and Python environment

Always use Zsh. A Conda environment `ptrc` is ready for you — no venv or uv needed.

```
OtherShell> zsh

(d) % conda activate ptrc
(ptrc) % <type command here>
````

This environment contains Python 3.14.10. **PyTorch recommends using this 
version of Python.** See it for yourself:

```
% curl https://pytorch.org/get-started/locally/
[…]

<h3>Python</h3>

<p>It is recommended that you use Python 3.10 - 3.14. You can install Python 
either through Homebrew or the Python website.</p>

[…]
```

If you really need an older version of Python due to dependencies, conduct 
necessary research about it, then raise it up to my attention. I have the final 
say on which Python version to use.
 
## Data specification

Use real, 32×32 RGB photos of exactly one centered, roughly upright cat or dog
head, including ears. The head should fill approximately 60–80% of the crop side
and measure at least 32 pixels wide in the pool image. Exclude illustrations,
renders, toys and prints. Aim for at least 500 usable crops per class. The official
crop rule uses square crops with optional rotation and Lanczos resizing.

## Layout

- `data/student/` contains the unlabelled raw pool.
- `data/dev/` contains curated examples for inspection and aggregate validation.
- `starter_kit/` contains the course evaluator, crop code and audit tool.
- `curate.py`, `photo_filter.py` and `configs/exclusions.csv` create the manifest.
- `train.py` fits and exports the denoiser. `submission/` is the graded interface, manifest and weights.
- `compare.py` runs aggregate dev comparisons. `results/` records measurements and data counts.
- `work/` stores local caches, intermediate checkpoints and review images.
- `audit/`, `DATA_CARD.md`, `AGENT_LOG.md` and `report.pdf` support the student handoff.


## Commands

Run from the project root in Zsh, using `ptrc` and the dependency paths in `README.md`.

- Data: `python curate.py scan` then `python curate.py build --source-margin=-0.022 --crop-margin=-0.031 --species-margin=0.015 --exclude configs/exclusions.csv`
- Audit: `python starter_kit/audit.py --manifest submission/manifest.csv --student-id "$student_id"`
- Train: `python train.py fit --steps 24000 --snapshot-every 2000 --run-dir work/train`
- Tests: `python -m pytest -q -p no:cacheprovider tests starter_kit/tests`
- Evaluate: `python starter_kit/evaluate.py --submission submission --device cpu --n-per-class 1000 --grading --out results/official.json`

## Conventions

<!-- Code style, how experiments are named and logged, where results go,
     commit message format, what must be true before committing (tests pass, ...). -->

### Code style

Use explicit tensor shapes and fixed seeds for reproducible experiments. Store training resumes and
detector caches in `work/`, and publish aggregate numbers and run settings in `results/`.

### Commits

Every commit requires two additional steps:

1. Your commit message MUST NOT end with `Co-authored-by: Codex` or similar attributions. 
   Generative AI usage is declared in the author line (see step 2).  
   Instead, end commits with `Co-authored-by: $PTRC_USER_ATTR`. Let the shell unwrap the 
   environment variable.

2. Your `git commit` command should have an additional `--author` argument, set to 
   `$PTRC_GENAI_ATTR`. DO NOT modify Git's config for `user.name`, `user.email` or 
   `user.signingkey`.

A correct commit looks like this:
```
% git cat-file commit HEAD
tree [hash]
author [$PTRC_GENAI_ATTR] [timestamp]    # set with `git commit --author`
committer [$PTRC_USER_ATTR] [timestamp]  # alreday set with config `user.*`
gpgsig -----BEGIN PGP SIGNATURE-----

[PTRC_USER's signature]                  # just commit as normal, signature will
                                         # be automatically applied

 -----END PGP SIGNATURE-----

[>>> YOUR MESSAGE HERE! <<<]

Co-authored-by: [$PTRC_USER_ATTR]        # set by your commit message
```

## Verification checklist for every change

- Run the crop, overfit, shape, value range, seed and batch-independence tests.
- Run `check_manifest`; review randomly selected crop/source pairs and record exclusions.
- Check the loss and sample grids. Run the official protocol at 1,000 generated images per class.
- Match the checkpoint manifest hash, evaluation stats, and audit rows to the final manifest.

## Known pitfalls in this project

- Detectors also recognize heads printed on products. Check both source and crop against photo prompts,
  then review the remaining crops. The filter does not prove compliance on an individual image.
- A mixed Conda and wheel PyTorch install can produce a dynamic-library conflict. Keep curation
  dependencies in the project-local overlay described in the README.
- The starter kit extracts float64 Inception features. Use its CPU device for the official evaluator.
- Local development metrics use the dev set. Preserve the hidden grading protocol and keep all pool
  images, intermediate checkpoints, and tool dependencies out of the submission archive.
