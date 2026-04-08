"""Ollama-backed Gemma 4 E2B enrichment model."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Sequence

import httpx
from PIL import Image

from image_archive.config import AppConfig
from image_archive.models.base import EnrichmentModel
from image_archive.types import EnrichmentResult

FAST_PROMPT = """
Analyze this image and return only valid JSON.
Use concise, search-friendly labels.
Schema:
{
  "content_type": "photo|document|screenshot|ui|illustration|scan|other",
  "auto_tags": ["short-tag"],
  "style_labels": ["style-label"],
  "object_labels": ["object"],
  "has_person": true,
  "has_face": false,
  "person_count_bucket": "0|1|2-3|4+",
  "face_count_bucket": "0|1|2-3|4+",
  "is_document": false,
  "is_screenshot": false,
  "text_heavy": false,
  "short_summary": "one short sentence"
}
Return no prose, markdown, or explanations.
""".strip()

DOCUMENT_PROMPT = """
Analyze this image carefully as a possible document, screenshot, scan, or UI capture.
Return only valid JSON.
Use concise, search-friendly labels and prioritize document/screenshot detection.
Schema:
{
  "content_type": "photo|document|screenshot|ui|illustration|scan|other",
  "auto_tags": ["short-tag"],
  "style_labels": ["style-label"],
  "object_labels": ["object"],
  "has_person": true,
  "has_face": false,
  "person_count_bucket": "0|1|2-3|4+",
  "face_count_bucket": "0|1|2-3|4+",
  "is_document": false,
  "is_screenshot": false,
  "text_heavy": false,
  "short_summary": "one short sentence"
}
Return no prose, markdown, or explanations.
""".strip()


def _extract_json_block(text: str) -> dict:
    """Parse the most likely JSON object from the model response."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class OllamaGemma4E2BEnrichmentModel(EnrichmentModel):
    """Structured image enrichment via Ollama chat API."""

    def __init__(self, config: AppConfig) -> None:
        self.model_name = config.enrichment_model
        self.enrichment_version = config.enrichment_version
        self.fast_image_max_size = int(config.fast_image_max_size)
        self.document_image_max_size = int(config.document_image_max_size)
        self._client = httpx.Client(
            base_url=config.ollama_host.rstrip("/"),
            timeout=config.ollama_timeout_seconds,
        )

    def _prepare_image(self, image: Image.Image, *, mode: str) -> str:
        max_size = self.fast_image_max_size if mode == "fast" else self.document_image_max_size
        prepared = image.copy()
        prepared.thumbnail((max_size, max_size))
        buffer = BytesIO()
        prepared.save(buffer, format="JPEG", quality=88)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _request_payload(self, encoded_image: str, *, mode: str) -> dict:
        prompt = FAST_PROMPT if mode == "fast" else DOCUMENT_PROMPT
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model_name,
                "stream": False,
                "format": "json",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [encoded_image],
                    }
                ],
                "options": {
                    "temperature": 0.1,
                    "top_k": 32,
                    "top_p": 0.9,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content", "{}")
        data = _extract_json_block(content)
        data["_ollama_model"] = self.model_name
        data["_mode"] = mode
        return data

    def enrich_images(self, images: Sequence[Image.Image], *, mode: str = "fast") -> list[EnrichmentResult]:
        results: list[EnrichmentResult] = []
        for image in images:
            payload = self._request_payload(self._prepare_image(image, mode=mode), mode=mode)
            results.append(EnrichmentResult.from_payload(payload))
        return results
