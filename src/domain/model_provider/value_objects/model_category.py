"""Model category enumeration."""

from enum import Enum


class ModelCategory(Enum):
    """Enumeration of model categories."""

    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    AUDIO_PROCESSING = "audio_processing"

    def __str__(self) -> str:
        """String representation."""
        return f"ModelCategory.{self.name}"

    @property
    def display_name(self) -> str:
        """Get display name for the category."""
        display_names = {
            self.TEXT_GENERATION: "Text Generation",
            self.CHAT_COMPLETION: "Chat Completion",
            self.CODE_GENERATION: "Code Generation",
            self.EMBEDDING: "Text Embedding",
            self.IMAGE_GENERATION: "Image Generation",
            self.AUDIO_PROCESSING: "Audio Processing",
        }
        return display_names.get(self, self.value.title())

    @property
    def supports_streaming(self) -> bool:
        """Check if category typically supports streaming responses."""
        streaming_categories = {
            self.TEXT_GENERATION,
            self.CHAT_COMPLETION,
            self.CODE_GENERATION,
        }
        return self in streaming_categories

    @property
    def typical_use_cases(self) -> list:
        """Get typical use cases for the category."""
        use_cases = {
            self.TEXT_GENERATION: ["Content creation", "Summarization", "Translation"],
            self.CHAT_COMPLETION: ["Conversational AI", "Q&A", "Virtual assistants"],
            self.CODE_GENERATION: ["Code completion", "Code explanation", "Debugging"],
            self.EMBEDDING: ["Semantic search", "Similarity comparison", "Classification"],
            self.IMAGE_GENERATION: ["Art creation", "Image editing", "Visual content"],
            self.AUDIO_PROCESSING: ["Speech recognition", "Audio transcription", "Voice synthesis"],
        }
        return use_cases.get(self, [])
