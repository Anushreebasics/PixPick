from __future__ import annotations

import math
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

from .quality import assess_image_quality


class LocalImageBackend:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", seed: Optional[int] = None) -> None:
        self.model_name = model_name
        self.seed = seed
        self._pipeline = None
        self._using_diffusers = False

    def generate_image(
        self,
        source_image_bytes: bytes,
        prompt: str,
        source_mime_type: Optional[str] = None,
        strength: float = 0.7,
        seed: Optional[int] = None,
    ) -> bytes:
        """Generate a single image (for backward compatibility)."""
        source_image = Image.open(BytesIO(source_image_bytes)).convert("RGB")
        seed_value = self.seed if seed is None else seed

        use_diffusers = os.getenv("LOCAL_USE_DIFFUSERS", "0").strip().lower() in {"1", "true", "yes", "on"}
        if use_diffusers:
            try:
                pipeline = self._load_diffusion_pipeline()
                if pipeline is not None:
                    return self._generate_with_diffusers(pipeline, source_image, prompt, strength, seed_value)
            except Exception:
                pass

        return self._generate_with_stylizer(source_image, prompt, strength, seed_value)

    def generate_multiple_images(
        self,
        source_image_bytes: bytes,
        prompt: str,
        num_variations: int = 8,
        source_mime_type: Optional[str] = None,
        strength: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate multiple image variations and return the best quality one.
        
        Returns:
            {
                'best_image_bytes': bytes,
                'best_score': float,
                'variation_index': int,
                'num_variations': int,
                'scores': List[float]
            }
        """
        variations = []
        scores = []
        check_counts = []
        passed_checks = []
        
        # Generate multiple variations and rank them by perceptual quality.
        for i in range(num_variations):
            seed_value = abs(hash((prompt, i))) % (2**32)
            
            generated_bytes = self.generate_image(
                source_image_bytes=source_image_bytes,
                prompt=prompt,
                source_mime_type=source_mime_type,
                strength=strength + (i * 0.02),  # Vary strength slightly
                seed=seed_value,
            )
            
            generated_image = Image.open(BytesIO(generated_bytes)).convert("RGB")
            quality_assessment = assess_image_quality(generated_image)
            
            variations.append(generated_bytes)
            scores.append(quality_assessment["score"])
            check_counts.append(quality_assessment["check_count"])
            passed_checks.append(quality_assessment["passed_checks"])
        
        # Prefer the variation that passes the most QA checks, then highest quality score.
        best_idx = max(range(len(variations)), key=lambda idx: (check_counts[idx], scores[idx]))
        best_score = scores[best_idx]
        best_image = variations[best_idx]
        
        return {
            'best_image_bytes': best_image,
            'best_score': best_score,
            'variation_index': best_idx,
            'num_variations': num_variations,
            'scores': scores,
            'check_counts': check_counts,
            'passed_checks': passed_checks,
            'variations': variations,
        }

    def validate_semantic_match(
        self,
        source_image_bytes: bytes,
        generated_image_bytes: bytes,
        source_mime_type: str,
        generated_mime_type: str,
        required_checks: List[str],
    ) -> Dict[str, Any]:
        """Validate the generated image using perceptual quality checks."""
        try:
            generated_image = Image.open(BytesIO(generated_image_bytes)).convert("RGB")
            assessment = assess_image_quality(generated_image)
            score = assessment["score"]
            passed_checks = assessment["passed_checks"]
            
            return {
                "passed": score > 0.75 and assessment["check_count"] >= 3,
                "score": score,
                "failure_types": [] if len(passed_checks) >= 3 else ["quality_checks_failed"],
                "corrective_constraints": [],
                "rationale": (
                    f"Perceptual quality score: {score:.2f}; "
                    f"passed checks: {', '.join(passed_checks) if passed_checks else 'none'}"
                ),
            }
        except Exception:
            return {
                "passed": False,
                "score": 0.0,
                "failure_types": ["image_comparison_failed"],
                "corrective_constraints": [],
                "rationale": "Failed to compare images",
            }

    def compare_images(self, source_image_bytes: bytes, image_a_bytes: bytes, image_b_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """Compare two generated images and pick the better one using perceptual heuristics.

        Returns a dict with keys: winner_index (0 or 1), passed (bool), rationale (str), scores (list)
        """
        try:
            from io import BytesIO

            a_img = Image.open(BytesIO(image_a_bytes)).convert("RGB")
            b_img = Image.open(BytesIO(image_b_bytes)).convert("RGB")

            a_assess = assess_image_quality(a_img)
            b_assess = assess_image_quality(b_img)

            # Prefer higher check_count then higher score
            a_key = (a_assess["check_count"], a_assess["score"])
            b_key = (b_assess["check_count"], b_assess["score"])

            if a_key == b_key:
                # tie-breaker: choose higher score
                winner = 0 if a_assess["score"] >= b_assess["score"] else 1
            else:
                winner = 0 if a_key > b_key else 1

            rationale = (
                f"Local comparison chosen winner={winner}; "
                f"A(score={a_assess['score']:.3f},checks={a_assess['check_count']}); "
                f"B(score={b_assess['score']:.3f},checks={b_assess['check_count']})"
            )

            passed = max(a_assess["score"], b_assess["score"]) > 0.75 and max(a_assess["check_count"], b_assess["check_count"]) >= 3

            return {
                "winner_index": int(winner),
                "passed": bool(passed),
                "rationale": rationale,
                "scores": [a_assess["score"], b_assess["score"]],
            }
        except Exception:
            return {"winner_index": 0, "passed": False, "rationale": "local_compare_failed", "scores": [0.0, 0.0]}

    def _load_diffusion_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except Exception:
            return None

        dtype = getattr(torch, "float16", None)
        use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        use_cuda = torch.cuda.is_available()

        model_name = self.model_name or "runwayml/stable-diffusion-v1-5"
        pipeline = AutoPipelineForImage2Image.from_pretrained(model_name, torch_dtype=dtype)
        if use_cuda:
            pipeline = pipeline.to("cuda")
        elif use_mps:
            pipeline = pipeline.to("mps")
        else:
            pipeline = pipeline.to("cpu")

        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

        self._pipeline = pipeline
        self._using_diffusers = True
        return self._pipeline

    def _generate_with_diffusers(self, pipeline, source_image: Image.Image, prompt: str, strength: float, seed: Optional[int]) -> bytes:
        import torch

        generator = None
        if seed is not None:
            if torch.cuda.is_available():
                generator = torch.Generator(device="cuda").manual_seed(seed)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                generator = torch.Generator(device="cpu").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)

        result = pipeline(
            prompt=prompt,
            image=source_image,
            strength=max(0.2, min(0.95, strength)),
            guidance_scale=7.0,
            num_inference_steps=20,
            generator=generator,
        )

        output_image = result.images[0].convert("RGB")
        buffer = BytesIO()
        output_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _generate_with_stylizer(self, source_image: Image.Image, prompt: str, strength: float, seed: Optional[int]) -> bytes:
        # Deterministic, local stylization with category-specific effects.
        if seed is None:
            seed = abs(hash(prompt)) % (2**32)

        working = source_image.copy().convert("RGB")
        
        # Detect category from prompt to apply category-specific stylization
        category = self._detect_category(prompt)
        
        if category == "electronics":
            working = self._stylize_electronics(working, strength, seed)
        elif category == "clothing":
            working = self._stylize_clothing(working, strength, seed)
        elif category == "home":
            working = self._stylize_home(working, strength, seed)
        else:
            working = self._stylize_generic(working, strength, seed)

        buffer = BytesIO()
        working.save(buffer, format="PNG")
        return buffer.getvalue()

    def _detect_category(self, prompt: str) -> str:
        """Detect product category from prompt keywords."""
        prompt_lower = prompt.lower()
        
        electronics_keywords = ["phone", "laptop", "tablet", "headphone", "camera", "electronics", "gadget", "device", "tech"]
        clothing_keywords = ["shirt", "dress", "pants", "jacket", "clothing", "apparel", "fashion", "wear", "shoe"]
        home_keywords = ["furniture", "sofa", "chair", "table", "home", "decor", "lamp", "curtain"]
        
        for keyword in electronics_keywords:
            if keyword in prompt_lower:
                return "electronics"
        for keyword in clothing_keywords:
            if keyword in prompt_lower:
                return "clothing"
        for keyword in home_keywords:
            if keyword in prompt_lower:
                return "home"
        
        return "generic"

    def _stylize_electronics(self, img: Image.Image, strength: float, seed: int) -> Image.Image:
        """Apply professional product styling for electronics: bright, sharp, subtle color enhancement."""
        working = img.copy()
        
        # Moderate brightness increase
        working = ImageEnhance.Brightness(working).enhance(1.12 + strength * 0.15)
        # Moderate saturation for vibrant product look
        working = ImageEnhance.Color(working).enhance(1.2 + strength * 0.15)
        # Enhance contrast moderately
        working = ImageEnhance.Contrast(working).enhance(1.2 + strength * 0.2)
        # Sharpen for clarity
        working = ImageEnhance.Sharpness(working).enhance(1.3 + strength * 0.2)
        
        # Apply color grading: subtle blue boost for tech
        working = self._apply_color_grade_tech(working, strength * 0.5)
        
        # Add subtle glow
        working = self._apply_glow(working, strength * 0.1, color=(200, 220, 255))
        
        return working

    def _stylize_clothing(self, img: Image.Image, strength: float, seed: int) -> Image.Image:
        """Apply fashion styling: warm, vibrant color with maintained structure."""
        working = img.copy()
        
        # Moderate brightness
        working = ImageEnhance.Brightness(working).enhance(1.1 + strength * 0.12)
        
        # Saturation boost
        working = ImageEnhance.Color(working).enhance(1.25 + strength * 0.15)
        
        # Moderate contrast
        working = ImageEnhance.Contrast(working).enhance(1.15 + strength * 0.15)
        
        # Sharpening
        working = ImageEnhance.Sharpness(working).enhance(1.2 + strength * 0.15)
        
        # Apply warm color grading
        working = self._apply_color_grade_fashion(working, strength * 0.5)
        
        # Subtle warm glow
        working = self._apply_glow(working, strength * 0.08, color=(255, 230, 200))
        
        return working

    def _stylize_home(self, img: Image.Image, strength: float, seed: int) -> Image.Image:
        """Apply warm, inviting styling for home furnishings."""
        working = img.copy()
        
        # Subtle brightness
        working = ImageEnhance.Brightness(working).enhance(1.08 + strength * 0.1)
        
        # Color enhancement
        working = ImageEnhance.Color(working).enhance(1.15 + strength * 0.12)
        
        # Contrast
        working = ImageEnhance.Contrast(working).enhance(1.1 + strength * 0.15)
        
        # Sharpening
        working = ImageEnhance.Sharpness(working).enhance(1.15 + strength * 0.12)
        
        # Apply luxury color grading
        working = self._apply_color_grade_luxury(working, strength * 0.5)
        
        # Subtle inviting glow
        working = self._apply_glow(working, strength * 0.08, color=(255, 240, 210))
        
        return working

    def _stylize_generic(self, img: Image.Image, strength: float, seed: int) -> Image.Image:
        """Apply general stylization for unknown categories."""
        working = img.copy()
        
        # Moderate enhancement
        working = ImageOps.autocontrast(working, cutoff=10)
        working = ImageEnhance.Brightness(working).enhance(1.1 + strength * 0.12)
        working = ImageEnhance.Color(working).enhance(1.2 + strength * 0.12)
        working = ImageEnhance.Contrast(working).enhance(1.15 + strength * 0.15)
        working = ImageEnhance.Sharpness(working).enhance(1.2 + strength * 0.15)
        
        # Subtle glow
        working = self._apply_glow(working, strength * 0.07, color=(220, 225, 230))
        
        return working

    def _apply_color_grade_tech(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle cool-tone color grading for tech products."""
        r, g, b = img.split()
        
        # Subtle blue boost
        b = ImageEnhance.Brightness(b).enhance(1.05 + strength * 0.08)
        # Slight red reduction
        r = ImageEnhance.Brightness(r).enhance(0.98 + strength * 0.02)
        
        return Image.merge("RGB", (r, g, b))

    def _apply_color_grade_fashion(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle warm color grading for fashion."""
        r, g, b = img.split()
        
        # Subtle warm tone
        r = ImageEnhance.Brightness(r).enhance(1.04 + strength * 0.06)
        g = ImageEnhance.Brightness(g).enhance(1.02 + strength * 0.04)
        b = ImageEnhance.Brightness(b).enhance(0.96 + strength * 0.04)
        
        return Image.merge("RGB", (r, g, b))

    def _apply_color_grade_luxury(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle luxury warm grading."""
        r, g, b = img.split()
        
        r = ImageEnhance.Brightness(r).enhance(1.03 + strength * 0.05)
        g = ImageEnhance.Brightness(g).enhance(1.02 + strength * 0.03)
        b = ImageEnhance.Brightness(b).enhance(0.97 + strength * 0.03)
        
        return Image.merge("RGB", (r, g, b))

    def _apply_glow(self, img: Image.Image, intensity: float, color: tuple = (255, 255, 255)) -> Image.Image:
        """Apply a subtle soft glow effect."""
        if intensity <= 0:
            return img
        
        # Create blurred version
        blurred = img.filter(ImageFilter.GaussianBlur(radius=2 + intensity * 2))
        
        # Subtle blend
        glow_alpha = max(0.02, min(0.15, intensity))
        return Image.blend(img, blurred, glow_alpha)

    def _apply_vignette(self, img: Image.Image, intensity: float) -> Image.Image:
        """Apply a subtle vignette (darkened edges) effect."""
        width, height = img.size
        
        # Create a radial gradient mask for vignette
        vignette = Image.new("L", (width, height), 255)
        vignette_pixels = vignette.load()
        
        center_x, center_y = width / 2, height / 2
        max_dist = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                falloff = max(0, 1 - (dist / max_dist))
                vignette_value = int(255 * (1 - intensity + (falloff * intensity)))
                vignette_pixels[x, y] = vignette_value
        
        # Apply vignette
        img_copy = img.copy()
        img_copy.putalpha(vignette)
        dark_bg = Image.new("RGB", img.size, (0, 0, 0))
        dark_bg.paste(img_copy, (0, 0), vignette)
        
        return dark_bg

