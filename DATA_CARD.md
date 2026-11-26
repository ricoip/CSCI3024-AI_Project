# Cat and dog head training data

The final manifest contains **987 cat** and **1,085 dog** crops. Pixel data comes
from the course-supplied raw pool. Each row specifies a square and class, and the
starter kit reconstructs it at 32×32 RGB with Lanczos resizing.

| Stage | Images |
| --- | ---: |
| Pool images read | 76,250 |
| Confident cat or dog detection | 13,794 |
| One usable head detection | 8,749 |
| Head matched to species box | 8,300 |
| Valid square and head size | 4,325 |
| Photo and species screen passed | 2,084 |
| After visual-review exclusions | 2,075 |
| After near-duplicate removal | 2,072 |

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

188/200 crops met the specification (94.0%). The 95% Wilson interval is 89.8% to 96.5%. The other 12 rows have a recorded failure reason.

The review exposed prints, text over the crop, obscured ears, and tilted heads.
Detector geometry can still take in too much background or select a profile.
The filter can favor familiar breeds and clean, front-facing imagery. A next pass
could check pose with independent head landmarks and sample rejections to measure
coverage loss. No per-image dev information enters curation or model training.

Sources. [YOLO26](https://docs.ultralytics.com/models/yolo26/),
[head checkpoint](https://github.com/idobd3006/digital-image-processing-project/tree/b0e39199c9cb55f264a65dcda570330f8a82693a),
[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/),
[CLIP](https://arxiv.org/abs/2103.00020).
