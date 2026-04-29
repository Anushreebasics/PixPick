from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from .local_backend import LocalImageBackend
from .quality import assess_image_quality


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
        variations: List[bytes] = []
        scores: List[float] = []
        check_counts: List[int] = []
        passed_checks: List[List[str]] = []

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
            quality_assessment = assess_image_quality(generated_image)

            variations.append(generated_bytes)
            scores.append(quality_assessment["score"])
            check_counts.append(quality_assessment["check_count"])
            passed_checks.append(quality_assessment["passed_checks"])

        if not variations:
            return self.local_fallback.generate_multiple_images(
                source_image_bytes=source_image_bytes,
                prompt=prompt,
                num_variations=num_variations,
                source_mime_type=source_mime_type,
                strength=strength,
            )

        best_idx = max(range(len(variations)), key=lambda idx: (check_counts[idx], scores[idx]))
        return {
            "best_image_bytes": variations[best_idx],
            "best_score": scores[best_idx],
            "variation_index": best_idx,
            "num_variations": num_variations,
            "scores": scores,
            "check_counts": check_counts,
            "passed_checks": passed_checks,
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

    def compare_images(self, source_image_bytes: bytes, image_a_bytes: bytes, image_b_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """Ask Gemini which of two images is better and parse a structured response.

        Returns: {winner_index: 0|1, passed: bool, rationale: str, scores: [float,float]}
        Falls back to local comparison on failure.
        """
        try:
            # Build a comparison prompt asking for a JSON response.
            human_prompt = (
                f"You are a QA assistant. Given the product prompt:\n{prompt}\n\n"
                "Compare the two candidate images: Image A and Image B. "
                "Which variation is better for product photography and why? "
                "Respond with a single JSON object exactly in the format: {\"winner\": 0 or 1, \"passed\": true|false, \"rationale\": \"...\"}."
            )

            a_b64 = base64.b64encode(image_a_bytes).decode("utf-8")
            b_b64 = base64.b64encode(image_b_bytes).decode("utf-8")

            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                f"?key={self.api_key}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": human_prompt},
                            {"inline_data": {"mime_type": "image/png", "data": a_b64}},
                            {"text": "--- Next image: Image B ---"},
                            {"inline_data": {"mime_type": "image/png", "data": b_b64}},
                        ]
                    }
                ],
                "generationConfig": {"responseModalities": ["TEXT"]},
            }

            resp = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()

            # Extract returned text
            text_out = ""
            for candidate in data.get("candidates", []):
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    if part.get("text"):
                        text_out += part.get("text") + "\n"

            # Try to locate a JSON object in the text and parse
            import re, json

            m = re.search(r"\{\s*\"winner\".*\}", text_out, re.S)
            if m:
                obj_text = m.group(0)
                parsed = json.loads(obj_text)
                winner = int(parsed.get("winner", 0))
                passed = bool(parsed.get("passed", False))
                rationale = str(parsed.get("rationale", ""))
                return {"winner_index": winner, "passed": passed, "rationale": rationale, "scores": [0.0, 0.0]}

            # Fallback: return local comparison
            return self.local_fallback.compare_images(source_image_bytes, image_a_bytes, image_b_bytes, prompt)
        except Exception:
            return self.local_fallback.compare_images(source_image_bytes, image_a_bytes, image_b_bytes, prompt)

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

