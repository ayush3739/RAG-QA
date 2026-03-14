import streamlit as st
from pathlib import Path
from datetime import datetime
import re
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


@st.cache_data(ttl=15)
def get_collection_chunk_count(collection_name: str) -> int:
    client = QdrantClient(url="http://localhost:6333")
    try:
        info = client.get_collection(collection_name)
        return int(info.points_count or 0)
    finally:
        client.close()

@st.cache_resource
def get_retriver(collection_name: str) -> Retriver:
    return Retriver(collection_name=collection_name)


def build_collection_name(file_names: list[str]) -> str:
    if len(file_names) == 1:
        base = Path(file_names[0]).stem
    else:
        base = f"batch-{len(file_names)}-files"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-").lower()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{safe}-{timestamp}"


def index_files(file_paths: list[Path], collection_name: str, status_container) -> None:
    status_container.info("Step 1 / 3 — Loading PDF/image content...")
    # Indexer.index() handles load → chunk → embed → store in one call
    indexer = Indexer(file_paths=file_paths, collection_name=collection_name)
    status_container.info("Step 2 / 3 — Chunking & embedding...")
    indexer.index()
    status_container.success("Step 3 / 3 — Stored in Qdrant ✓")


# ── page config 

st.set_page_config(page_title="MULTI-MODAL RAG System", page_icon="📚", layout="wide")
st.title("📚 MULTI-MODAL RAG System")

sidebar_collection = st.session_state.get("active_collection")
if st.session_state.get("selected_collection"):
    sidebar_collection = st.session_state["selected_collection"]

st.sidebar.header("Collection Stats")
if sidebar_collection:
    chunk_count = get_collection_chunk_count(sidebar_collection)
    st.sidebar.write(f"Collection: {sidebar_collection}")
    st.sidebar.metric("Chunks", chunk_count)
else:
    st.sidebar.info("Select or index a collection to view chunk count.")

# ── tabs

tab_new, tab_existing = st.tabs(["Upload New Document", "Query Existing Collection"])

# ── Tab 1: Upload & index new files 
with tab_new:
    st.subheader("Upload PDFs and images")
    uploaded_files = st.file_uploader(
        "Choose up to 5 files (PDF, PNG, JPG, JPEG, WEBP)",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        key="uploader",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("You can upload at most 5 files at a time.")
        else:
            file_names = [file.name for file in uploaded_files]
            default_collection_name = build_collection_name(file_names)
            collection_name = st.text_input(
                "Collection name",
                value=default_collection_name,
                help="All uploaded files will be indexed into this one collection.",
            )
            already_indexed = collection_name in get_existing_collections()

            st.write("**Selected files:**")
            for name in file_names:
                st.write(f"- {name}")

            if already_indexed:
                st.info("This collection name already exists in Qdrant.")

            col1, col2 = st.columns([3, 1])
            with col2:
                index_btn = st.button(
                    "Re-index" if already_indexed else "Index Files",
                    use_container_width=True,
                )

            if index_btn:
                saved_paths: list[Path] = []
                for uploaded_file in uploaded_files:
                    save_path = DOCS_DIR / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(save_path)

                status = st.empty()
                index_files(saved_paths, collection_name, status)
                st.session_state["active_collection"] = collection_name
                st.success(f"Ready to query `{collection_name}`!")

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
        st.info("Upload and index up to 5 PDF/image files above to start chatting.")

# ── Tab 2: Query existing collection

with tab_existing:
    st.subheader("Select an existing collection")
    collections = get_existing_collections()

    if not collections:
        st.warning("No collections found in Qdrant. Index a document first.")
    else:
        selected = st.selectbox("Available collections", collections, key="selected_collection")

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