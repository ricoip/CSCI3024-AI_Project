# Cat and dog head generation

Class-conditional diffusion with velocity prediction, a cosine noise schedule,
DDIM sampling, and classifier-free guidance (CFG). Label `0` requests a cat and
label `1` a dog. The grader uses `submission/generate.py`, `manifest.csv`, and
`weights.pt`. The denoiser starts from fresh weights and sampling starts from
seeded noise.

## Run the demo

From the project root in Zsh, activate `ptrc` and run the local sampler. The
pretrained curation models and course data are unnecessary for this command.

```zsh
conda activate ptrc
python demo.py --label both --count 4 --seed 3024 --out work/demo-3024.png
```

For a replay check, save a second grid with the same seed and compare the pixel
SHA-256 printed by the two runs. Change the seed and compare again. `--label cat`
and `--label dog` request a single class.

## Reproduce the manifest

The experiment used Python 3.14.10, PyTorch 2.15.0, TorchVision 0.29.2, and the
starter kit's dependencies. Put the unchanged kit and class data in
`starter_kit/`, `data/student/`, and `data/dev/` as described in the starter-kit
README. Set `CATDOG_DATA` if the last two directories live under a different
parent. The evaluator needs the published Inception weights in the PyTorch hub
cache, documented in the starter kit.

Install the following curation and report dependencies in an overlay to keep
Ultralytics from pulling in a separate PyTorch build:

```zsh
conda activate ptrc
python -m pip install --no-deps --only-binary=:all: --target data/ptrc-deps \
  'ultralytics==8.4.218' 'opencv-python-headless==5.1.0.95' \
  'ultralytics-thop==2.2.5' 'polars==2.1.0' 'polars-runtime-32==2.1.0' \
  'reportlab==5.1.1' 'pypdf==7.1.1'
python -m pip install --no-deps --only-binary=:all: --target data/clip-deps \
  'ftfy==6.3.1' 'regex==2026.11.23'
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
  -o data/yolo26n.pt
curl -fL https://raw.githubusercontent.com/idobd3006/digital-image-processing-project/b0e39199c9cb55f264a65dcda570330f8a82693a/runs/detect/train_head/weights/best.pt \
  -o data/pet_head_yolo26n.pt
curl -fL https://codeload.github.com/openai/CLIP/zip/a1d071733d7111c9c014f024669f959182114e33 \
  -o data/openai-clip-source.zip
unzip -nq data/openai-clip-source.zip -d data
curl -fL https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
  -o data/ViT-B-32.pt
python - <<'PY'
from pathlib import Path
f = Path('data/CLIP-a1d071733d7111c9c014f024669f959182114e33/clip/clip.py')
code = f.read_text().replace('from pkg_resources import packaging', 'from packaging import version')
f.write_text(code.replace('packaging.version.parse', 'version.parse'))
PY
export PYTHONPATH="$PWD/data/ptrc-deps:$PWD/data/clip-deps:$PWD/data/CLIP-a1d071733d7111c9c014f024669f959182114e33"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The curation weights' SHA-256 digests are in `results/data.json`. The official
COCO detector locates cats and dogs, and the Oxford-IIIT Pet detector localizes
their heads. CLIP filters the source images and proposed crops. These models are
used locally for curation only. [YOLO26](https://docs.ultralytics.com/models/yolo26/) ·
[Pet head checkpoint](https://github.com/idobd3006/digital-image-processing-project/tree/b0e39199c9cb55f264a65dcda570330f8a82693a) ·
[OpenAI CLIP](https://github.com/openai/CLIP)

```zsh
python curate.py scan --limit 2048 --device mps
python curate.py scan --device mps
python curate.py build --device mps --source-margin=-0.022 --crop-margin=-0.031 \
  --species-margin=0.015 --exclude configs/exclusions.csv
python curate.py review --out work/review --per-class 50
```

Use `--device cpu` and a fresh cache path if MPS is unavailable. Scans resume
from the last committed batch in `work/detections.sqlite`. Detector results near
thresholds can vary across devices. The archived manifest defines the exact
training crops. The following commands reproduce the detector-only and strict
photo-filter pilot counts:

```zsh
python curate.py build --limit 2048 --without-photo-filter \
  --out work/pilot/detector.csv --stats work/pilot/detector.json
python curate.py build --limit 2048 --device mps \
  --out work/pilot/strict.csv --stats work/pilot/strict.json
```

## Reproduce training and evaluation

```zsh
python -m pytest -q -p no:cacheprovider tests starter_kit/tests
python train.py fit --steps 24000 --snapshot-every 2000 --device mps --run-dir work/train
```

`train.py fit` draws balanced batches from official reconstructions of the
manifest. The default run uses width 64, seed 3024, batch size 64, learning rate
0.0003, and 15% class dropout. For a resumed run, pass the same arguments and
`--resume`, with a total step count above the last saved step. Save `work/train`
outside an existing run if you want to rerun from scratch.

Use `compare.py` to reproduce a row in `results/trial-*.json`. Pass the row's
EMA checkpoint step, guidance scale, DDIM steps, and per-class sample count.
The official metric code runs on CPU and takes aggregate dev measurements.

```zsh
python compare.py --checkpoint work/train/ema-012000.pt --guidance 1.5 \
  --steps 32 --n-per-class 160 --seed 0 --device mps
```

Export the checkpoint listed in `results/selection.json` and run
the unmodified evaluator at the official sample count. This is the source of
the metrics, resource usage, sampling speed, and sample grid in the report.

```zsh
export OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 OPENBLAS_NUM_THREADS=6
python train.py export \
  --checkpoint "$(python -c 'import json; print(json.load(open("results/selection.json"))["checkpoint"])')" \
  --guidance-scale "$(python -c 'import json; print(json.load(open("results/selection.json"))["guidance_scale"])')" \
  --sampling-steps "$(python -c 'import json; print(json.load(open("results/selection.json"))["ddim_steps"])')"
python starter_kit/evaluate.py --submission submission --device cpu \
  --n-per-class 1000 --seed 0 --grading --out results/official.json
```

The hidden grade uses a separate held-out reference and seed. CFG uses the
denoiser's trained null label and requires no extra classifier. The sampler uses
DDIM and produces deterministic instances from seeded CPU noise.
[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) ·
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

## Audit and submit

The official CSV and four audit sheets are in `audit/`. For each index, enter
`good` or `bad` in `verdict` and give a reason for every `bad`. Use the course
specification to judge photo content, the species, one head, upright pose,
centering, head fill and source resolution. Only redraw the sheets if needed,
since the audit command resets the verdicts. Set `student_id` locally in Zsh.

```zsh
read 'student_id?Student ID: '
python starter_kit/audit.py --manifest submission/manifest.csv --student-id "$student_id"
```

Record your independent checks in `AGENT_LOG.md`. After labeling and preparing
your slides, regenerate the report and project archive:

```zsh
python deliver.py --student-id "$student_id" --name "Your Name" --slides /path/to/your/slides.pptx
```

This validates the audit rows, estimates data precision with a 95% Wilson
interval, and writes `DATA_CARD.md`, `report.pdf`, and `project.zip`. Upload the
report and zip separately to UMMoodle. The archive excludes pool and dev images,
review sheets, caches, environments, and intermediate checkpoints.
