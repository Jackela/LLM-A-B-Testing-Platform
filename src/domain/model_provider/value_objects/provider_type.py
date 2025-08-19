"""Provider type enumeration."""

from enum import Enum


class ProviderType(Enum):
    """Enumeration of supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    BAIDU = "baidu"
    ALIBABA = "alibaba"

    def __str__(self) -> str:
        """String representation."""
        return f"ProviderType.{self.name}"

    @classmethod
    def get_display_name(cls, provider_type: "ProviderType") -> str:
        """Get display name for provider type."""
        display_names = {
            cls.OPENAI: "OpenAI",
            cls.ANTHROPIC: "Anthropic",
            cls.GOOGLE: "Google",
            cls.BAIDU: "Baidu",
            cls.ALIBABA: "Alibaba",
        }
        return display_names.get(provider_type, provider_type.value.title())

    @classmethod
    def get_default_models(cls, provider_type: "ProviderType") -> list:
        """Get default model IDs for provider type."""
        default_models = {
            cls.OPENAI: ["gpt-4", "gpt-3.5-turbo"],
            cls.ANTHROPIC: ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
            cls.GOOGLE: ["gemini-1.5-pro", "gemini-1.5-flash"],
            cls.BAIDU: ["ernie-4.0-turbo", "ernie-3.5"],
            cls.ALIBABA: ["qwen-turbo", "qwen-plus"],
        }
        return default_models.get(provider_type, [])
