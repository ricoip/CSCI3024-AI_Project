"""Zero-shot photo and species screening for raw-pool crop candidates."""

import torch
from PIL import ImageOps
from torch.nn.functional import normalize


SPECIES = ("cat", "dog")
PHOTO = (
    "a natural photo of a real pet {animal}",
    "a photograph of a living {animal} in a real scene",
    "a candid photograph of the face of a real {animal}",
)
NONPHOTO = (
    "a cartoon, painting or illustration of a {animal}",
    "a 3D render, statue, plastic figurine, or stuffed toy of a {animal}",
    "a pet food bag or box with a picture of a {animal} printed on it",
    "a shirt, mug, bag, or gift printed with a picture of a {animal}",
    "a product listing for merchandise with a printed image of a {animal}",
    "a greeting card, meme, or advertising design with a {animal} picture and text",
    "a collage, logo, or sticker of a {animal}",
)


class PhotoFilter:
    def __init__(self, weights, device):
        import clip

        self.device = device
        model, self.preprocess = clip.load(str(weights), device="cpu", jit=False)
        self.model = model.to(device).eval()
        prompts = [pattern.format(animal=animal)
                   for animal in SPECIES for pattern in PHOTO + NONPHOTO]
        with torch.inference_mode():
            text = self.model.encode_text(clip.tokenize(prompts).to(device))
            self.text = normalize(text.float()).T

    @torch.inference_mode()
    def margins(self, originals, crops, labels):
        # Pad the source view to preserve packaging and other context near borders.
        # The local crop view is more informative for species and visual texture.
        sources = [ImageOps.pad(image, (224, 224), color=(123, 117, 104)) for image in originals]
        image_batch = torch.stack([self.preprocess(image) for image in sources + crops])
        features = normalize(self.model.encode_image(image_batch.to(self.device)).float())
        source_scores, crop_scores = (features @ self.text).reshape(
            -1, len(SPECIES), len(PHOTO) + len(NONPHOTO)
        ).cpu().chunk(2)
        photo_source = source_scores[:, :, :len(PHOTO)].amax(dim=2)
        nonphoto_source = source_scores[:, :, len(PHOTO):].amax(dim=2)
        photo_crop = crop_scores[:, :, :len(PHOTO)].amax(dim=2)
        nonphoto_crop = crop_scores[:, :, len(PHOTO):].amax(dim=2)
        i = torch.arange(len(labels))
        labels = torch.tensor(labels)
        return torch.stack((
            photo_source[i, labels] - nonphoto_source[i, labels],
            photo_crop[i, labels] - nonphoto_crop[i, labels],
            photo_crop[i, labels] - photo_crop[i, 1 - labels],
        ), dim=1).tolist()
