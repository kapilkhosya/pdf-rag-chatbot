"""Streamlit UI for the PDF RAG app. Run with: streamlit run app.py"""

import streamlit as st
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, RAGIndex, answer

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


if "index" not in st.session_state:
    st.session_state.index = RAGIndex(get_embedder())
    st.session_state.messages = []  # {"role", "content", "sources"}

index: RAGIndex = st.session_state.index

# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)

    if files:
        for f in files:
            if f.name not in index.sources:
                with st.spinner(f"Indexing {f.name}..."):
                    n = index.add_pdf(f.name, f.getvalue())
                if n:
                    st.success(f"{f.name}: {n} chunks")
                else:
                    st.warning(f"{f.name}: no extractable text (scanned PDF?)")

    if index.sources:
        st.caption("Indexed: " + ", ".join(sorted(index.sources)))

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main: chat ----------
st.title("📄 Chat with your PDFs")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for src in m["sources"]:
                    st.markdown(f"**{src['file']} - page {src['page']}** (score {src['score']:.2f})")
                    st.caption(src["text"])

if question := st.chat_input("Ask something about your documents"):
    if not index.sources:
        st.warning("Upload at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            hits = index.search(question, k=top_k)
            # Send only the plain text of earlier turns (last 6) for follow-ups
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-6:]
            ]
            try:
                reply = answer(question, hits, history)
            except Exception as e:
                reply = f"Error calling the LLM: {e}"
        st.markdown(reply)

        sources = [
            {"file": c.source, "page": c.page, "score": s, "text": c.text[:300] + "..."}
            for c, s in hits
        ]
        with st.expander("Sources"):
            for src in sources:
                st.markdown(f"**{src['file']} - page {src['page']}** (score {src['score']:.2f})")
                st.caption(src["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
