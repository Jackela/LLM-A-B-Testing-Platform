"""Main FastAPI application."""

import logging

from .app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create app instance
app = create_app()

logger.info("🚀 LLM A/B Testing Platform API initialized")
