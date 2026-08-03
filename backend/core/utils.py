# backend/core/utils.py
import re
import logging
from typing import Tuple, Optional

from qdrant_client import QdrantClient
# backend/core/models.py

# FastEmbed is no longer used. We use the Jina Reranker v3 API.


def simple_tokenize(text: str):
    return re.findall(r"\w+", text.lower())
