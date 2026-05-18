"""Streamlit PDF Chatbot."""
from __future__ import annotations

import hashlib

import streamlit as st
from dotenv import load_dotenv

from embeddings import VectorStore
from llm import (
    SUPPORTED_MODELS,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    generate_answer,
)
from pdf_processor import PDFExtractionError, process_pdf

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_vector_store() -> VectorStore:
    return VectorStore()


def file_fingerprint(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()


def reset_chat() -> None:
    st.session_state.history = []


def main() -> None:
    st.title("PDF Chatbot")
    st.caption("Upload a PDF and ask questions grounded in its contents.")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "pdf_id" not in st.session_state:
        st.session_state.pdf_id = None
    if "pdf_meta" not in st.session_state:
        st.session_state.pdf_meta = None

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Model", SUPPORTED_MODELS, index=0)
        top_k = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=3)
        if st.button("Clear chat history"):
            reset_chat()
            st.success("Chat history cleared.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    store = get_vector_store()

    if uploaded is not None:
        fingerprint = file_fingerprint(uploaded)
        if st.session_state.pdf_id != fingerprint:
            with st.spinner("Extracting text and building the index…"):
                try:
                    uploaded.seek(0)
                    chunks, page_count = process_pdf(uploaded)
                    store.build(chunks)
                except PDFExtractionError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to process PDF: {e}")
                    st.stop()

            st.session_state.pdf_id = fingerprint
            st.session_state.pdf_meta = {
                "name": uploaded.name,
                "pages": page_count,
                "chunks": len(chunks),
            }
            reset_chat()

    meta = st.session_state.pdf_meta
    if meta:
        st.success(
            f"Processed **{meta['name']}** — {meta['pages']} page(s), "
            f"{meta['chunks']} chunk(s) indexed."
        )

    if not meta:
        st.info("Upload a PDF to get started.")
        return

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("sources"):
                with st.expander(f"Sources ({len(turn['sources'])})"):
                    for src in turn["sources"]:
                        st.markdown(
                            f"**Chunk #{src['index']} — page {src['page']} "
                            f"(score {src['score']:.3f})**"
                        )
                        st.markdown(f"> {src['text']}")

    question = st.chat_input("Ask a question about the PDF…")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            results = store.search(question, k=top_k)
        except Exception as e:  # noqa: BLE001
            st.error(f"Search failed: {e}")
            return

        contexts = [r.chunk.text for r in results]
        try:
            with st.spinner("Generating answer…"):
                answer = generate_answer(model=model, question=question, contexts=contexts)
        except LLMTimeoutError as e:
            st.error(str(e))
            return
        except LLMUnavailableError as e:
            st.error(str(e))
            return
        except LLMError as e:
            st.error(str(e))
            return

        st.markdown(answer)

        sources = [
            {
                "index": r.chunk.index,
                "page": r.chunk.page,
                "score": r.score,
                "text": r.chunk.text,
            }
            for r in results
        ]

        with st.expander(f"Sources ({len(sources)})"):
            for src in sources:
                st.markdown(
                    f"**Chunk #{src['index']} — page {src['page']} "
                    f"(score {src['score']:.3f})**"
                )
                st.markdown(f"> {src['text']}")

    st.session_state.history.append(
        {"question": question, "answer": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
