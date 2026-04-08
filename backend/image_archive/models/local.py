"""Local model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from image_archive.models.base import CaptionModel, EmbeddingModel, EnrichmentModel
from image_archive.types import EnrichmentResult


def resolve_device(device: str) -> str:
    """Resolve a friendly device config to a torch device string."""

    if device != "auto":
        return device

    import torch  # pylint: disable=import-outside-toplevel

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEmbeddingModel(EmbeddingModel):
    """CLIP embedding model for text and images."""

    def __init__(self, model_name: str, device: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.device = resolve_device(device)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch  # pylint: disable=import-outside-toplevel
        from transformers import AutoProcessor, CLIPModel  # pylint: disable=import-outside-toplevel

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.clip(norms, a_min=1e-12, a_max=None)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        with self._torch.no_grad():
            inputs = self._processor(images=list(images), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self._model.get_image_features(**inputs)
            array = features.detach().cpu().numpy().astype(np.float32)
        return self._normalize(array)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        with self._torch.no_grad():
            inputs = self._processor(
                text=list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self._model.get_text_features(**inputs)
            array = features.detach().cpu().numpy().astype(np.float32)
        return self._normalize(array)


@dataclass
class ZeroShotCandidate:
    label: str
    prompt: str


CONTENT_TYPE_CANDIDATES = [
    ZeroShotCandidate("photo", "a photograph"),
    ZeroShotCandidate("document", "a scanned document or paper page"),
    ZeroShotCandidate("screenshot", "a computer or phone screenshot"),
    ZeroShotCandidate("ui", "a user interface screen"),
    ZeroShotCandidate("illustration", "an illustration or drawing"),
    ZeroShotCandidate("scan", "a scanned page or receipt"),
    ZeroShotCandidate("other", "an abstract or miscellaneous image"),
]

STYLE_CANDIDATES = [
    ZeroShotCandidate("editorial", "an editorial style image"),
    ZeroShotCandidate("minimal", "a minimal composition"),
    ZeroShotCandidate("warm-light", "warm light photography"),
    ZeroShotCandidate("cool", "a cool toned image"),
    ZeroShotCandidate("negative-space", "negative space composition"),
    ZeroShotCandidate("cinematic", "a cinematic image"),
    ZeroShotCandidate("product", "product photography"),
    ZeroShotCandidate("street", "street photography"),
    ZeroShotCandidate("portrait", "portrait photography"),
    ZeroShotCandidate("natural", "natural organic style"),
]

TAG_CANDIDATES = [
    ZeroShotCandidate("portrait", "a portrait of a person"),
    ZeroShotCandidate("landscape", "a landscape scene"),
    ZeroShotCandidate("screenshot", "a screenshot"),
    ZeroShotCandidate("document", "a document with text"),
    ZeroShotCandidate("food", "food photography"),
    ZeroShotCandidate("architecture", "architecture"),
    ZeroShotCandidate("nature", "plants or nature"),
    ZeroShotCandidate("animal", "an animal"),
    ZeroShotCandidate("travel", "a travel photo"),
    ZeroShotCandidate("art", "artwork or illustration"),
    ZeroShotCandidate("text", "text on a page or screen"),
    ZeroShotCandidate("workspace", "a desk or workspace"),
]

OBJECT_CANDIDATES = [
    ZeroShotCandidate("person", "a person"),
    ZeroShotCandidate("face", "a human face"),
    ZeroShotCandidate("laptop", "a laptop computer"),
    ZeroShotCandidate("phone", "a phone"),
    ZeroShotCandidate("car", "a car"),
    ZeroShotCandidate("book", "a book"),
    ZeroShotCandidate("plant", "a plant"),
    ZeroShotCandidate("dog", "a dog"),
    ZeroShotCandidate("cat", "a cat"),
    ZeroShotCandidate("table", "a table"),
    ZeroShotCandidate("chair", "a chair"),
    ZeroShotCandidate("window", "a window"),
]


class ClipZeroShotEnrichmentModel(EnrichmentModel):
    """Fast structured enrichment using CLIP zero-shot label matching."""

    def __init__(self, embedding_model: ClipEmbeddingModel, enrichment_version: str = "clip-zero-shot-v1") -> None:
        self.embedding_model = embedding_model
        self.model_name = f"clip-zero-shot:{embedding_model.model_name}"
        self.enrichment_version = enrichment_version

    def _rank(self, image_vector: np.ndarray, candidates: list[ZeroShotCandidate]) -> list[tuple[str, float]]:
        text_vectors = self.embedding_model.encode_texts([candidate.prompt for candidate in candidates])
        scores = text_vectors @ image_vector
        ranked = sorted(
            [(candidate.label, float(score)) for candidate, score in zip(candidates, scores)],
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked

    def _labels_above(
        self,
        image_vector: np.ndarray,
        candidates: list[ZeroShotCandidate],
        *,
        top_k: int,
        threshold: float,
    ) -> list[str]:
        ranked = self._rank(image_vector, candidates)
        selected = [label for label, score in ranked if score >= threshold][:top_k]
        return selected or [label for label, _score in ranked[: min(top_k, len(ranked))]]

    def enrich_images(self, images: Sequence[Image.Image], *, mode: str = "fast") -> list[EnrichmentResult]:
        image_vectors = self.embedding_model.encode_images(images)
        results: list[EnrichmentResult] = []
        for vector in image_vectors:
            content_type = self._rank(vector, CONTENT_TYPE_CANDIDATES)[0][0]
            tags = self._labels_above(vector, TAG_CANDIDATES, top_k=5, threshold=0.24)
            styles = self._labels_above(vector, STYLE_CANDIDATES, top_k=3, threshold=0.24)
            objects = self._labels_above(vector, OBJECT_CANDIDATES, top_k=4, threshold=0.24)
            has_person = "person" in objects or "portrait" in tags or "portrait" in styles
            has_face = "face" in objects or "portrait" in tags
            is_document = content_type in {"document", "scan"} or "document" in tags
            is_screenshot = content_type in {"screenshot", "ui"} or "screenshot" in tags
            text_heavy = is_document or is_screenshot or "text" in tags
            results.append(
                EnrichmentResult(
                    content_type=content_type,
                    auto_tags=tags,
                    style_labels=styles,
                    object_labels=objects,
                    has_person=has_person,
                    has_face=has_face,
                    person_count_bucket="1" if has_person else "0",
                    face_count_bucket="1" if has_face else "0",
                    is_document=is_document,
                    is_screenshot=is_screenshot,
                    text_heavy=text_heavy,
                    short_summary=" ".join([content_type, *styles[:2], *tags[:3]]),
                )
            )
        return results


class BlipCaptionModel(CaptionModel):
    """BLIP image captioning model."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch  # pylint: disable=import-outside-toplevel
        from transformers import BlipForConditionalGeneration, BlipProcessor  # pylint: disable=import-outside-toplevel

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def caption_images(self, images: Sequence[Image.Image]) -> list[str]:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        prompt = "a detailed description of this image"
        prompts = [prompt] * len(images)
        with self._torch.no_grad():
            inputs = self._processor(
                images=list(images),
                text=prompts,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            output_ids = self._model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=60,
            )
        return [
            self._processor.decode(item, skip_special_tokens=True).strip()
            for item in output_ids
        ]
