import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asyncio
import json

from backend.core.retriever import Retriever
from backend.db.base import AsyncSessionLocal
async def main():
    async with AsyncSessionLocal() as db:

        retriever = Retriever(
            document_id=7,
            db=db
        )

        retrieval = await retriever.similarity_search(
            "What is this document about?"
        )

        print(json.dumps(retrieval, indent=2))


if __name__ == "__main__":
    asyncio.run(main())