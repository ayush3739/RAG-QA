from pathlib import Path
import base64
import mimetypes
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import  QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv('./.env')

class Indexer():
    def __init__(self, file_paths: list[Path] | Path, collection_name: str | None = None):
        if isinstance(file_paths, (str, Path)):
            self.file_paths = [Path(file_paths)]
        else:
            self.file_paths = [Path(path) for path in file_paths]
        if not self.file_paths:
            raise ValueError("At least one file is required for indexing.")

        self.collection_name = collection_name or self.file_paths[0].name
        self.embedding_model = OpenAIEmbeddings(
            api_key=os.getenv('GITHUB_TOKEN'),
            model="text-embedding-3-large",
            openai_api_base="https://models.github.ai/inference",
        )
        self.openai_client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
        )

    def _load_pdf(self, file_path: Path) -> list[Document]:
        loader = PyPDFLoader(file_path=str(file_path))
        return loader.load()

    def _load_image(self, file_path: Path) -> list[Document]:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "image/png"

        with open(file_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all readable text from this image. If there is little text, also describe important visual details for retrieval.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
        )
        extracted_text = (response.choices[0].message.content or "").strip()
        if not extracted_text:
            extracted_text = "No readable text extracted from image."

        return [
            Document(
                page_content=extracted_text,
                metadata={"source": str(file_path), "page_label": "image"},
            )
        ]

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for file_path in self.file_paths:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                docs.extend(self._load_pdf(file_path))
            elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                docs.extend(self._load_image(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.name}")
        return docs

    def index(self):
        # Step 1: load supported files (PDF + images)
        docs = self._load_documents()

        # Step 2: chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400,
        )
        chunks = text_splitter.split_documents(documents=docs)
        print(f"Total Chunks created: {len(chunks)}")

        # Step 3: embed & index
        self.vector_db = QdrantVectorStore.from_documents(
            documents=chunks,
            url="http://localhost:6333",
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        print(f"Indexing done → collection: '{self.collection_name}'")