# backend/core/utils.py
import re
import os
from typing import Tuple, Optional

from qdrant_client import QdrantClient
# backend/core/models.py

from sentence_transformers import CrossEncoder

print("Loading reranker...")

RERANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded")

def simple_tokenize(text: str):
    return re.findall(r"\w+", text.lower())
