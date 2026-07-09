# backend/core/utils.py
import re
import logging
from typing import Tuple, Optional

from qdrant_client import QdrantClient
# backend/core/models.py

from sentence_transformers import CrossEncoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("Loading reranker...")

try:
    RERANKER = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    logger.info("Reranker loaded")

except Exception as e:
    RERANKER = None
    logger.exception(f"Failed to load reranker: {e}")


def simple_tokenize(text: str):
    return re.findall(r"\w+", text.lower())
