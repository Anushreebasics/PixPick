from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

from .local_backend import LocalImageBackend


class GeminiImageBackend:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-image",
        timeout_seconds: int = 120,
        local_fallback: Optional[LocalImageBackend] = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name or "gemini-2.5-flash-image"
        self.timeout_seconds = timeout_seconds
        self.local_fallback = local_fallback or LocalImageBackend()
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_image(
        self,
        source_image_bytes: bytes,
        prompt: str,
        source_mime_type: Optional[str] = None,
        strength: float = 0.7,
        seed: Optional[int] = None,
    ) -> bytes:
        _ = (strength, seed)  # Gemini API does not expose these controls in this integration.
        try:
            return self._call_gemini_generate_image(source_image_bytes, prompt, source_mime_type or "image/png")
        except Exception as exc:
            self.logger.warning("Gemini image generation failed, using local fallback: %s", exc)
            return self.local_fallback.generate_image(
                source_image_bytes=source_image_bytes,
                prompt=prompt,
                source_mime_type=source_mime_type,
                strength=strength,
                seed=seed,
            )

    def generate_multiple_images(
        self,
        source_image_bytes: bytes,
        prompt: str,
        num_variations: int = 8,
        source_mime_type: Optional[str] = None,
        strength: float = 0.7,
    ) -> Dict[str, Any]:
        source_image = Image.open(BytesIO(source_image_bytes)).convert("RGB")

        variations: List[bytes] = []
        scores: List[float] = []

        for i in range(num_variations):
            variation_prompt = (
                f"{prompt}\n\n"
                f"Create variation {i + 1}/{num_variations}. "
                "Preserve product identity and structure exactly while applying tasteful ad-style enhancement."
            )
            generated_bytes = self.generate_image(
                source_image_bytes=source_image_bytes,
                prompt=variation_prompt,
                source_mime_type=source_mime_type,
                strength=strength + (i * 0.02),
                seed=i,
            )

            generated_image = Image.open(BytesIO(generated_bytes)).convert("RGB")
            similarity_score = self._compute_image_similarity(source_image, generated_image)

            variations.append(generated_bytes)
            scores.append(similarity_score)

        if not variations:
            return self.local_fallback.generate_multiple_images(
                source_image_bytes=source_image_bytes,
                prompt=prompt,
                num_variations=num_variations,
                source_mime_type=source_mime_type,
                strength=strength,
            )

        best_idx = scores.index(max(scores))
        return {
            "best_image_bytes": variations[best_idx],
            "best_score": scores[best_idx],
            "variation_index": best_idx,
            "num_variations": num_variations,
            "scores": scores,
        }

    def validate_semantic_match(
        self,
        source_image_bytes: bytes,
        generated_image_bytes: bytes,
        source_mime_type: str,
        generated_mime_type: str,
        required_checks: List[str],
    ) -> Dict[str, Any]:
        _ = (source_mime_type, generated_mime_type, required_checks)
        return self.local_fallback.validate_semantic_match(
            source_image_bytes=source_image_bytes,
            generated_image_bytes=generated_image_bytes,
            source_mime_type=source_mime_type,
            generated_mime_type=generated_mime_type,
            required_checks=required_checks,
        )

    def _call_gemini_generate_image(self, source_image_bytes: bytes, prompt: str, source_mime_type: str) -> bytes:
        image_b64 = base64.b64encode(source_image_bytes).decode("utf-8")
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            f"?key={self.api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": source_mime_type,
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        response = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and inline_data.get("data"):
                    return base64.b64decode(inline_data["data"])

        raise RuntimeError("Gemini response did not contain image data")

    def _compute_image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        mse = np.mean((arr1 - arr2) ** 2)
        max_mse = 255.0 ** 2
        similarity = max(0.0, 1.0 - (mse / max_mse))
        return float(similarity)
