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

<!-- Where things live: data pipeline, manifests, models, training
     scripts, configs, experiment logs, report. -->

`AGENTS.md`: You're here!  
- `data/student/` contains the unlabelled raw pool.
- `data/dev/` contains curated examples for inspection and aggregate validation.
(`eval_results.json`: Output of `evaluate.py`)  
(`eval_results.png`: Output of `evaluate.py`)  
`starter_kit/`: The toolkit for grading our work.  
`starter_kit/submission_template/`: Template of the `submission` folder.  
`submission/`: A soon-to-be-filled-in version of the submission template.


## Commands

<!-- Exact commands, e.g.
- Tests:     `pytest -q`
- Data:      `python data/build_manifest.py --out submission/manifest.csv`
- Audit:     `python starter_kit/audit.py --manifest submission/manifest.csv --student-id ...`
- Train:     `python train.py --manifest submission/manifest.csv --config configs/cgan.yaml`
- Evaluate:  `python starter_kit/evaluate.py --submission submission --n-per-class 250`
-->

## Conventions

<!-- Code style, how experiments are named and logged, where results go,
     commit message format, what must be true before committing (tests pass, ...). -->

### Code style



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

<!-- What the agent must check before claiming something works, e.g.
- unit tests pass (including: my crop code == catdog_eval.data.crop)
- `check_manifest` reports no errors
- a grid of 100 random crops from a new manifest was saved and looked at
- a short training run shows the losses behaving and a sample grid was looked at
-->`

## Known pitfalls in this project

<!-- Things that went wrong before and how to avoid them. -->
