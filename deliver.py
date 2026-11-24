"""Build the data card and report, and package the project after the student audit."""

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from curate import file_digest
from audit import sample_rows
from catdog_eval.data import default_pool
from catdog_eval.manifest import check_manifest


STAGES = (
    ("scanned", "Pool images read"),
    ("body_detection", "Confident cat or dog detection"),
    ("single_head", "One usable head detection"),
    ("species_matched", "Head matched to species box"),
    ("valid_geometry", "Valid square and head size"),
    ("photo_accepted", "Photo and species screen passed"),
    ("after_review_exclusions", "After visual-review exclusions"),
    ("after_dedup", "After near-duplicate removal"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wilson(successes, total):
    z = 1.95996398454005
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return centre - margin, centre + margin


def audit_summary(student_id):
    with Path("audit/audit.csv").open(newline="", encoding="utf-8") as handle:
        recorded = list(csv.DictReader(handle))
    manifest = check_manifest("submission/manifest.csv", default_pool())
    if manifest["errors"]:
        raise ValueError(manifest["errors"])
    rows = sample_rows(manifest["rows"], student_id, 200)
    if len(recorded) != 200 or len(rows) != 200:
        raise ValueError("expected the official 200-row audit")
    good, bad = 0, 0
    for i, (actual, expected) in enumerate(zip(recorded, rows, strict=True)):
        if (int(actual["index"]) != i or actual["pool_id"] != expected["pool_id"]
                or any(float(actual[key]) != expected[key] for key in ("x0", "y0", "size", "angle", "label"))):
            raise ValueError(f"audit row {i} differs from the official draw for this manifest and student ID")
        verdict = actual["verdict"].strip()
        good += verdict == "good"
        bad += verdict == "bad" and bool(actual["reason"].strip())
    if good + bad < 200:
        return (f"{good + bad}/200 audit rows have valid student verdicts. The precision estimate and "
                "confidence interval are pending the completed manual audit."), False
    low, high = wilson(good, 200)
    return (f"{good}/200 crops met the specification ({good / 200:.1%}). The 95% Wilson interval "
            f"is {low:.1%} to {high:.1%}. The other {bad} rows have a recorded failure reason."), True


def data_card(data, audit):
    counts = data["stage_counts"]
    cats, dogs = data["class_counts"]["cat"], data["class_counts"]["dog"]
    stages = "\n".join(f"| {label} | {counts[key]:,} |" for key, label in STAGES)
    Path("DATA_CARD.md").write_text(f"""# Cat and dog head training data

The final manifest contains **{cats:,} cat** and **{dogs:,} dog** crops. Pixel data comes
from the course-supplied raw pool. Each row specifies a square and class, and the
starter kit reconstructs it at 32×32 RGB with Lanczos resizing.

| Stage | Images |
| --- | ---: |
{stages}

The COCO YOLO26n detector proposes species and an Oxford-IIIT Pet YOLO26n checkpoint
locates the head. We require one head scoring at least 0.45, with a second head at
0.30 enough to reject the source. A species box must score at least 0.45 and contain
at least 75% of the detected head. We target 70% fill along its longer dimension,
require at least 32 source pixels across the head, and reject substantial centering
shifts at image boundaries. All final angles are zero degrees.

CLIP ViT-B/32 compares photo prompts with product, illustration, toy and collage
prompts on the whole source and crop. Negative thresholds account for how often
the competing descriptions mention a real animal. A crop must also score higher
for its assigned species. The defaults for the experiment are recorded in
`results/data.json`. We removed clear failures found during the pilot review and
kept the higher joint-confidence crop when 64-bit difference hashes differed by
at most three bits.

## Student audit

{audit}

The review exposed prints, text over the crop, obscured ears, and tilted heads.
Detector geometry can still take in too much background or select a profile.
The filter can favor familiar breeds and clean, front-facing imagery. A next pass
could check pose with independent head landmarks and sample rejections to measure
coverage loss. No per-image dev information enters curation or model training.

Sources. [YOLO26](https://docs.ultralytics.com/models/yolo26/),
[head checkpoint](https://github.com/idobd3006/digital-image-processing-project/tree/b0e39199c9cb55f264a65dcda570330f8a82693a),
[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/),
[CLIP](https://arxiv.org/abs/2103.00020).
""", encoding="utf-8")


def build_report(name, student_id, data, training, weights, official, trials, audit, audit_complete):
    fonts = Path(get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold")
    ink, accent, muted = (colors.HexColor(value) for value in ("#182331", "#146459", "#506275"))
    body = ParagraphStyle("Body", fontName="DejaVu", fontSize=10, leading=15, textColor=ink, spaceAfter=10)
    title = ParagraphStyle("Title", parent=body, fontName="DejaVuBold", fontSize=24, leading=29, spaceAfter=12)
    heading = ParagraphStyle("Heading", parent=body, fontName="DejaVuBold", fontSize=13, leading=17,
                             spaceBefore=8, spaceAfter=7, textColor=accent)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=12, textColor=muted, spaceAfter=8)
    equation = ParagraphStyle("Equation", parent=body, fontSize=10, leading=20, leftIndent=14)
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def h(text):
        p(text, heading)

    def table(rows, widths):
        t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
            ("FONTNAME", (0, 0), (-1, 0), "DejaVuBold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf4f2")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend((t, Spacer(1, 10)))

    m, r = official["metrics"], official["resources"]
    cats, dogs = data["class_counts"]["cat"], data["class_counts"]["dog"]
    p("CISC3024  /  PATTERN RECOGNITION", small)
    p("Cat and Dog Head Generation", title)
    p(f"{escape(name) if name else 'Name ____________________'} &nbsp;&nbsp; Student ID {escape(student_id)}", small)
    if not name or not audit_complete:
        p("Draft pending student details, audit, or verification entries.", small)
    h("Summary")
    p(f"A class-conditional diffusion model generates 32×32 RGB cat and dog heads from seeded noise. "
      f"It was trained from scratch on {cats + dogs:,} crops curated from the course pool. The official "
      f"local evaluation measured {m['accuracy']:.1%} class accuracy, FID {m['fid']:.2f}, "
      f"and recall {m['recall']:.3f} using the provided development set as the reference.")
    h("Data and curation")
    p("A COCO YOLO26n detector establishes species [1]. An Oxford-IIIT Pet head detector [2, 3] supplies "
      "bounding boxes. Geometry filters target one centered head that fills 70% of the square along "
      "its longer dimension, with at least 32 pixels of head width in the source. CLIP [4] screens both "
      "source and crop for photo content and checks crop species. We used the starter kit's square crop rule.")
    table([["Curation stage", "Images"]] +
          [[label, f"{data['stage_counts'][key]:,}"] for key, label in STAGES], [355, 125])
    p(f"The manifest has {cats:,} cats and {dogs:,} dogs. Each pool image contributes at most one crop.")
    h("Student audit")
    p(escape(audit))
    story.append(PageBreak())

    p("Conditional diffusion", title)
    h("Denoiser")
    p("A U-Net predicts velocity in pixel space. A reversible 2×2 pixel rearrangement maps the 32×32 RGB "
      "input to 12 channels at 16×16. The downsampling path uses widths 64, 128 and 192. Four-head "
      "attention at 4×4 provides global context. The upsampling path uses spatial skip connections. "
      "Residual blocks receive the sum of a sinusoidal time embedding and a learned class embedding "
      "via scale and shift after group normalization.")
    h("Training objective")
    p("The forward process uses 1,000 cosine-scheduled noise levels [5, 6]. Let A<sub>t</sub> be the "
      "cumulative product of clipped signal-retention coefficients, and let z be standard Gaussian noise. "
      "Clean pixels x<sub>0</sub> are scaled to [−1, 1]. The velocity target follows Salimans and Ho [7].")
    p("x<sub>t</sub> = √A<sub>t</sub> x<sub>0</sub> + √(1 − A<sub>t</sub>) z<br/>"
      "v = √A<sub>t</sub> z − √(1 − A<sub>t</sub>) x<sub>0</sub><br/>"
      "L(θ) = E [ ‖v<sub>θ</sub>(x<sub>t</sub>, t, c̃) − v‖² ]", equation)
    p(f"Batches contain equal numbers of cat and dog crops sampled with replacement. The training "
      f"loop drops the class label with probability {training['label_dropout']:.0%} and uses a third "
      f"embedding for the null label. AdamW uses a peak learning rate of {training['lr']:.0e}, "
      f"{training['warmup']} warmup steps, and batch size {training['batch_size']}. "
      f"Sampling uses exponential moving average weights from step {weights['training_step']:,}.")
    h("Classifier-free guidance and DDIM")
    p("At each reverse step, the same denoiser predicts conditional and unconditional velocities. "
      "Guide scale s combines them as in classifier-free guidance [8]. The deterministic DDIM update [9] "
      "clips the clean estimate and recomputes the noise estimate from that clipped value.")
    p("v̂ = v<sub>null</sub> + s (v<sub>class</sub> − v<sub>null</sub>)<br/>"
      "x̂<sub>0</sub> = clip(√A<sub>t</sub> x<sub>t</sub> − √(1 − A<sub>t</sub>) v̂, −1, 1)<br/>"
      "ẑ = (x<sub>t</sub> − √A<sub>t</sub> x̂<sub>0</sub>) / √(1 − A<sub>t</sub>)<br/>"
      "x<sub>r</sub> = √A<sub>r</sub> x̂<sub>0</sub> + √(1 − A<sub>r</sub>) ẑ &nbsp; for r &lt; t", equation)
    p(f"The exported model uses {weights['sampling_steps']} DDIM steps and scale "
      f"{weights['guidance_scale']:g}. A local CPU random generator supplies each output position's "
      "initial noise. Group normalization and attention operate within one image, which keeps predictions "
      "independent of the other labels in the batch.")
    story.append(PageBreak())

    p("Official local evaluation", title)
    protocol = official["protocol"]
    p(f"The unmodified course evaluator generated {protocol['n_per_class']:,} images per class on "
      f"{protocol['device'].upper()}, using seed {protocol['seed']} and the course dev reference. "
      "The results below measure performance on this local protocol.")
    table([
        ["Measure", "Result"], ["Accuracy", f"{m['accuracy']:.3f}"], ["FID", f"{m['fid']:.2f}"],
        ["KID (mean ± SD)", f"{m['kid']:.5f} ± {m['kid_std']:.5f}"],
        ["Precision", f"{m['precision']:.3f}"], ["Recall", f"{m['recall']:.3f}"],
        ["Memorization rate", f"{m['memorization_rate']:.2%}"],
        ["Parameters", f"{r['num_parameters']:,}"], ["Submission size", f"{r['submission_mb']:.2f} MiB"],
        ["Hardware", "Apple M2 Pro, 12-core CPU, 32 GB RAM"],
        ["Training time to selected step", f"{weights['training_seconds'] / 60:.1f} min on MPS"],
        ["Sampling speed", f"{r['seconds_per_1000']:.1f} s / 1,000 images"],
        ["Local metric tiers", f"{official['marks']['total']}/15"],
    ], [185, 295])
    image_path = Path("results/official.png")
    if image_path.is_file():
        figure = Image(str(image_path), width=480, height=480 * 274 / 546)
        story.append(figure)
        p("Uncurated samples from the official evaluation. Cats in the top four rows and dogs below.", small)
    checks = official["checks"]
    p(f"The evaluator's mixed and single-class probes reached {checks['mixed_batch_accuracy']:.1%} "
      f"and {checks['single_class_batch_accuracy']:.1%} accuracy. The run reported "
      f"{len(official['violations'])} hard-limit violations.", small)
    story.append(PageBreak())

    p("Experiments and verification", title)
    h("Checkpoint and guidance comparisons")
    p("The following trials used the same local dev reference and seed. These FID values use fewer samples "
      "than the official result on the previous page.", small)
    if trials:
        rows = [["Step", "DDIM", "Scale", "Images/class", "Acc.", "FID", "Recall", "Mem."]]
        for trial in trials:
            t = trial["metrics"]
            rows.append([f"{trial['checkpoint_step']:,}", str(trial["ddim_steps"]), f"{trial['guidance_scale']:g}",
                         str(trial["n_per_class"]), f"{t['accuracy']:.3f}", f"{t['fid']:.2f}",
                         f"{t['recall']:.3f}", f"{t['memorization_rate']:.2%}"])
        table(rows, [65, 53, 48, 84, 56, 58, 58, 58])
    h("What did not work")
    p("The detector-only pilot included packaging, printed clothes, and artwork. The strict "
      "CLIP thresholds retained only nine of the 112 geometrically valid pilot crops. Thresholds "
      "based on the raw-pool pilot recovered many valid photos. The "
      "visual review still found a small number of obvious misses and added explicit exclusions.")
    h("Working with a coding agent")
    p("Rico chose diffusion and requested a classifier-free guidance comparison. The agent implemented "
      "the data pipeline, denoiser, tests, and evaluation commands. The project rules restrict pretrained "
      "models to curation, reserve the evaluator networks for aggregate evaluation, and bind export to "
      "the training manifest hash. AGENT_LOG.md is the record for the student's independent checks.")
    h("References")
    references = (
        ("Ultralytics. YOLO26 detection model and COCO weights.", "https://docs.ultralytics.com/models/yolo26/"),
        ("idobd3006. Oxford-IIIT Pet head detection checkpoint.",
         "https://github.com/idobd3006/digital-image-processing-project/tree/b0e39199c9cb55f264a65dcda570330f8a82693a"),
        ("Parkhi et al. Cats and Dogs. CVPR 2012.", "https://www.robots.ox.ac.uk/~vgg/data/pets/"),
        ("Radford et al. Learning Transferable Visual Models From Natural Language Supervision. 2021.",
         "https://arxiv.org/abs/2103.00020"),
        ("Ho, Jain and Abbeel. Denoising Diffusion Probabilistic Models. 2020.", "https://arxiv.org/abs/2006.11239"),
        ("Nichol and Dhariwal. Improved Denoising Diffusion Probabilistic Models. 2021.",
         "https://arxiv.org/abs/2102.09672"),
        ("Salimans and Ho. Progressive Distillation for Fast Sampling of Diffusion Models. 2022.",
         "https://arxiv.org/abs/2202.00512"),
        ("Ho and Salimans. Classifier-Free Diffusion Guidance. 2022.", "https://arxiv.org/abs/2207.12598"),
        ("Song, Meng and Ermon. Denoising Diffusion Implicit Models. 2020.", "https://arxiv.org/abs/2010.02502"),
    )
    for i, (citation, url) in enumerate(references, 1):
        p(f'[{i}] <link href="{url}" color="#146459">{citation}</link>', small)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd8d5"))
        canvas.line(54, 44, A4[0] - 54, 44)
        canvas.setFillColor(muted)
        canvas.setFont("DejaVu", 8)
        canvas.drawString(54, 30, "CISC3024  |  Cat and dog head generation")
        canvas.drawRightString(A4[0] - 54, 30, str(doc.page))
        canvas.restoreState()

    report = SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=54, rightMargin=54,
                               topMargin=48, bottomMargin=52, title="Cat and Dog Head Generation",
                               author=name)
    report.build(story, onFirstPage=footer, onLaterPages=footer)


def package(slides):
    if not slides.is_file():
        raise FileNotFoundError(slides)
    source = ["AGENTS.md", "AGENT_LOG.md", "README.md", "DATA_CARD.md", "curate.py", "photo_filter.py",
              "train.py", "demo.py", "compare.py", "deliver.py", "audit/audit.csv", "tests/test_project.py",
              "submission/generate.py", "submission/manifest.csv", "submission/weights.pt", "configs/exclusions.csv"]
    source += [str(path) for path in sorted(Path("results").iterdir()) if path.suffix in (".csv", ".json")]
    temporary = Path("project.zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(name)
        archive.write(slides, f"slides/{slides.name}")
    temporary.replace("project.zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--slides", type=Path)
    args = parser.parse_args()
    data = read_json("results/data.json")
    training = read_json("results/train_config.json")
    official = read_json("results/official.json")
    weights = torch.load("submission/weights.pt", map_location="cpu", weights_only=True)
    digest = file_digest("submission/manifest.csv")
    if any(digest != record["manifest_sha256"] for record in (data, training, weights)):
        raise ValueError("reports and weights must correspond to the final manifest")
    if official["protocol"]["n_per_class"] != 1000 or official["metrics"]["memorization_rate"] is None:
        raise ValueError("run the full official evaluator on the submitted model first")
    trials = [read_json(path) for path in sorted(Path("results").glob("trial-*.json"))]
    audit, complete = audit_summary(args.student_id)
    data_card(data, audit)
    build_report(args.name, args.student_id, data, training, weights, official, trials, audit, complete)
    print("Wrote DATA_CARD.md and report.pdf")
    if args.slides is not None:
        if not args.name or not complete:
            raise ValueError("the final archive requires your name and all 200 audit verdicts")
        package(args.slides)
        print("Wrote project.zip")


if __name__ == "__main__":
    main()
