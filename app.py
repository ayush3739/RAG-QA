import streamlit as st
from pathlib import Path
from indexing import Indexer
from retrieving import Retriver
from qdrant_client import QdrantClient

DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ── helpers

@st.cache_data(ttl=30)
def get_existing_collections() -> list[str]:
    client = QdrantClient(url="http://localhost:6333")
    names = [col.name for col in client.get_collections().collections]
    client.close()
    return names

@st.cache_resource
def get_retriver(collection_name: str) -> Retriver:
    return Retriver(collection_name=collection_name)


def index_pdf(pdf_path: Path, status_container) -> None:
    status_container.info("Step 1 / 3 — Loading PDF pages...")
    # Indexer.index() handles load → chunk → embed → store in one call
    indexer = Indexer(file_path=pdf_path)
    status_container.info("Step 2 / 3 — Chunking & embedding...")
    indexer.index()
    status_container.success("Step 3 / 3 — Stored in Qdrant ✓")


# ── page config 

st.set_page_config(page_title="RAG System", page_icon="📚", layout="wide")
st.title("📚 RAG System")

# ── tabs

tab_new, tab_existing = st.tabs(["Upload New Document", "Query Existing Collection"])

# ── Tab 1: Upload & index new PDF 
with tab_new:
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="uploader")

    if uploaded_file:
        pdf_path = DOCS_DIR / uploaded_file.name
        already_indexed = uploaded_file.name in get_existing_collections()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**File:** `{uploaded_file.name}`")
            if already_indexed:
                st.info("This document is already indexed in Qdrant.")
        with col2:
            index_btn = st.button(
                "Re-index" if already_indexed else "Index Document",
                use_container_width=True,
            )

        if index_btn:
            # save to docs/
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            status = st.empty()
            index_pdf(pdf_path, status)
            st.session_state["active_collection"] = uploaded_file.name
            st.success(f"Ready to query `{uploaded_file.name}`!")

    # ── chat (new doc)
    active = st.session_state.get("active_collection")
    if active:
        st.divider()
        st.subheader(f"Chat — {active}")

        if "messages_new" not in st.session_state:
            st.session_state["messages_new"] = []

        for msg in st.session_state["messages_new"]:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask a question about your document…", key="input_new")
        if query:
            st.session_state["messages_new"].append({"role": "user", "content": query})
            st.chat_message("user").write(query)
            with st.chat_message("assistant"):
                retriver = get_retriver(active)
                response = st.write_stream(retriver.answer_stream(query=query))
            st.session_state["messages_new"].append({"role": "assistant", "content": response})
    else:
        st.info("Upload and index a PDF above to start chatting.")

# ── Tab 2: Query existing collection

with tab_existing:
    st.subheader("Select an existing collection")
    collections = get_existing_collections()

    if not collections:
        st.warning("No collections found in Qdrant. Index a document first.")
    else:
        selected = st.selectbox("Available collections", collections)

        st.divider()
        st.subheader(f"Chat — {selected}")

        if "messages_existing" not in st.session_state:
            st.session_state["messages_existing"] = []

        for msg in st.session_state["messages_existing"]:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask a question about this collection…", key="input_existing")
        if query:
            st.session_state["messages_existing"].append({"role": "user", "content": query})
            st.chat_message("user").write(query)
            with st.chat_message("assistant"):
                retriver = get_retriver(selected)
                response = st.write_stream(retriver.answer_stream(query=query))
            st.session_state["messages_existing"].append({"role": "assistant", "content": response})