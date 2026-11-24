# Project

Class-conditional generation of 32×32 cat and dog heads (course project). The
training set is curated by the TAs from a raw pool of LAION web images and described by
`submission/manifest.csv`. Graded by `starter_kit/evaluate.py`; the interface is fixed
(see `submission/generate.py`).

## Hard constraints (never violate)

- Training images come only from the raw pool in `data/student/`, and every one is a row of
  a manifest (`pool_id,x0,y0,size,angle,label`, cropped with `catdog_eval.data.crop`).
- Never train on `data/dev/` or anything derived from it; never use other datasets or images
  made or edited by generative models.
- Pretrained models (detectors, CLIP, ...) only for curating data. The generative model is
  trained from scratch; no pretrained weights, no downloads at generation time.
- Never use the evaluation classifier or the evaluator's Inception network in training or
  generation.
- `generate(model, labels, seed)` must be deterministic in `(labels, seed)` and return
  `[N, 3, 32, 32]` floats in `[0, 1]`; labels 0 = cat, 1 = dog.
- Budget: <= 20M parameters, submission directory <= 100 MB, memorization rate <= 5%.
- Do not modify anything under `starter_kit/`.

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

<!-- Copy the specification from the handout, and add the decisions you made to meet it
     (detector, thresholds, crop margin, filters), with the reason for each. -->

## Layout

<!-- Where things live: data pipeline, manifests, models, training
     scripts, configs, experiment logs, report. -->

`AGENTS.md`: You're here!  
`data/student/`: Training data.  
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
