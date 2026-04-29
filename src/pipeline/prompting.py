from .config import PromptConfig
from .types import ProductRecord, ValidationResult


class PromptBuilder:
    def __init__(self, prompt_config: PromptConfig) -> None:
        self.prompt_config = prompt_config

    def build_initial_prompt(self, record: ProductRecord) -> str:
        category_prompt = self.prompt_config.categories.get(record.category, self.prompt_config.default_style)
        title_line = f"Product title: {record.product_title}." if record.product_title else ""
        return (
            f"{self.prompt_config.system_prompt}\n\n"
            f"Category prompt:\n{category_prompt}\n\n"
            f"Target style variant: {record.style}.\n"
            f"{title_line}\n"
            "Output requirements:\n"
            "- Photorealistic quality suitable for e-commerce listing\n"
            "- Keep the product as primary focus\n"
            "- Preserve identity and structural details from source image\n"
        )

    def build_retry_prompt(self, base_prompt: str, validation: ValidationResult) -> str:
        constraints = []
        for failure in validation.failure_types:
            constraints.extend(self.prompt_config.failure_constraints.get(failure, []))

        constraints.extend(validation.corrective_constraints)

        deduped = []
        seen = set()
        for item in constraints:
            normalized = item.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(item.strip())

        if not deduped:
            return base_prompt

        extra = "\n".join(f"- {line}" for line in deduped)
        return f"{base_prompt}\n\nCorrective constraints for retry:\n{extra}\n"
