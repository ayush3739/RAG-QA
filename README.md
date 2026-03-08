# 📚 Rag QA

> Upload any PDF → Index it → Chat with it. Answers are grounded purely in your document — if it's not there, the system says so.

---

## ✨ What does it do?

1. **Upload** a PDF — saved locally to `docs/`
2. **Index** it — chunked, embedded, and stored in Qdrant vector DB
3. **Chat** — ask anything; get streamed, context-grounded answers with page references
4. **Hallucination-free** — if the answer isn't in the PDF, the system tells you: *"This information is not available in the provided context"*

---

## 🖼️ Screenshots

**Upload & Index** — drop a PDF, hit Index Document, watch 3 live progress steps.

![Upload and Index](src/localhost_8501_%20(3).png)

**Streaming Answer** — answers stream token by token with page references from the document.

![Streaming Answer](src/localhost_8501_%20(1).png)

**Honest Response** — if it's not in the PDF, it says so. No hallucination.

![Out of Context](src/localhost_8501_%20(2).png)

---

## 🏗️ Architecture

```
PDF File
   │
   ▼
Indexer (indexing.py)
   ├── PyPDFLoader                      → load pages
   ├── RecursiveCharacterTextSplitter   → chunk (1000 chars, 400 overlap)
   ├── OpenAIEmbeddings                 → embed (text-embedding-3-large)
   └── QdrantVectorStore                → persist in local Qdrant

Query
   │
   ▼
Retriver (retrieving.py)
   ├── similarity_search   → top-10 chunks from Qdrant
   └── answer_stream       → streamed LLM response (token by token)

app.py  (Streamlit UI)
   ├── Tab 1 — Upload new PDF → index → chat
   └── Tab 2 — Pick existing collection → chat instantly
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Vector DB | Qdrant (Docker) |
| Embeddings | `text-embedding-3-large` via GitHub Models |
| LLM | `gpt-4o-mini` via GitHub Models |
| PDF Loader | LangChain `PyPDFLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

---

## 🚀 Setup

### 1. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env`
```env
GITHUB_TOKEN=your_github_models_token
```

### 4. Run
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
rag-qa/
├── app.py          # Streamlit UI — tabs, caching, streaming
├── indexing.py     # Indexer class — load → chunk → embed → store
├── retrieving.py   # Retriver class — search → stream answer
├── docs/           # Uploaded PDFs saved here automatically
├── src/            # Screenshots
└── .env            # API keys (not committed)
```
