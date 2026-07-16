from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

from prompts.message_classifier_prompt import get_message_classifier_prompt

import streamlit as st

from config.config import get_config
from ingestion.loader import DocumentLoader
from ingestion.parser import DocumentParser
from ingestion.chunker import Chunker
from ingestion.embedding import OllamaEmbeddingService
from ingestion.indexer import FaissIndexer
from retrieval.retriever import DocumentRetriever
from retrieval.reranker import SimpleReranker
from retrieval.prompt_builder import build_prompt
from retrieval.generator import OllamaGenerator
from evaluation.context_relevancy import evaluate_context_relevancy
from evaluation.groundedness import evaluate_groundedness
from evaluation.answer_relevancy import evaluate_answer_relevancy
from utils.helper import ensure_directory
from utils.logger import get_logger


logger = get_logger("App")
CONFIG = get_config()
ensure_directory(CONFIG.raw_dir)
ensure_directory(CONFIG.vector_store_dir)


def sidebar_index_controls():
    st.sidebar.header("Indexing")
    uploaded_files = st.sidebar.file_uploader("Upload documents (PDF, DOCX, TXT, MD)", accept_multiple_files=True)

    if uploaded_files:
        if st.sidebar.button("Save uploads to raw/"):
            for uploaded in uploaded_files:
                dest = Path(CONFIG.raw_dir) / uploaded.name
                with dest.open("wb") as fh:
                    fh.write(uploaded.getbuffer())
            st.sidebar.success("Saved uploaded files to data/raw/")

    if st.sidebar.button("Build Index"):
        build_index_flow()

    if st.sidebar.button("Clear Index"):
        FaissIndexer(CONFIG).clear()
        st.sidebar.success("Cleared FAISS index artifacts")


def build_index_flow():
    st.sidebar.info("Building index — this may take a few minutes.")
    loader = DocumentLoader(raw_dir=CONFIG.raw_dir, config=CONFIG)
    parser = DocumentParser()
    chunker = Chunker(CONFIG)
    embedding_service = OllamaEmbeddingService(CONFIG)
    indexer = FaissIndexer(CONFIG)

    documents = loader.load_documents()
    if not documents:
        st.sidebar.error("No documents found in data/raw/. Upload files first.")
        return

    parsed = [parser.parse(doc) for doc in documents]
    chunks = chunker.chunk_documents(parsed)
    indexer.build_index(chunks, embedding_service)
    st.sidebar.success(f"Index built with {len(chunks)} chunks.")


def classify_message(message: str) -> dict[str, str]:
    """Classify a customer support message into a support category."""
    message = message.strip()
    if not message:
        return {"category": "general inquiry", "reason": "No message was provided.", "mode": "rule-based"}

    lowered = message.lower()
    billing_keywords = ["invoice", "charge", "billing", "refund", "payment", "price", "subscription", "credit card", "receipt"]
    technical_keywords = ["error", "bug", "crash", "login", "server", "connection", "technical", "device", "app", "broken", "problem", "not working"]

    if any(keyword in lowered for keyword in billing_keywords):
        return {"category": "billing", "reason": "The message mentions billing or payment concerns.", "mode": "rule-based"}
    if any(keyword in lowered for keyword in technical_keywords):
        return {"category": "technical issues", "reason": "The message references a technical problem or product issue.", "mode": "rule-based"}

    try:
        generator = OllamaGenerator(CONFIG)
        prompt = get_message_classifier_prompt(message)
        response = generator.generate(prompt)
        category = "general inquiry"
        reason = response.strip() or "Used the local LLM classifier."
        lowered_response = response.lower()
        if "billing" in lowered_response:
            category = "billing"
        elif "technical" in lowered_response or "issue" in lowered_response:
            category = "technical issues"
        else:
            category = "general inquiry"
        return {"category": category, "reason": reason, "mode": "llm"}
    except Exception as exc:
        logger.warning("LLM classification failed, falling back to heuristics: %s", exc)
        return {"category": "general inquiry", "reason": "Fallback rules were used because the LLM classifier was unavailable.", "mode": "rule-based"}


def message_classifier_interface():
    st.header("Support Message Classifier")
    st.caption("Paste a customer support message and classify it into a support category.")

    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []

    with st.form("classification_form", clear_on_submit=True):
        message = st.text_area(
            "Customer support message",
            height=140,
            placeholder="Example: I was charged twice for my subscription and need a refund.",
        )
        submitted = st.form_submit_button("Classify")

    if submitted and message.strip():
        result = classify_message(message)
        st.session_state.classification_history.append({"message": message, **result})

        st.success(f"Predicted category: {result['category']}")
        st.write(result["reason"])
        st.caption(f"Mode: {result['mode']}")

    if st.session_state.classification_history:
        st.subheader("Recent classifications")
        for item in reversed(st.session_state.classification_history[-5:]):
            st.markdown(f"**Category:** {item['category']}")
            st.write(item["message"])
            st.caption(item["reason"])
            st.markdown("---")


def chat_interface():
    st.title("Local RAG Chatbot (Ollama + FAISS)")
    retriever = DocumentRetriever(CONFIG)
    reranker = SimpleReranker()
    generator = OllamaGenerator(CONFIG)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}

    with st.form("query_form", clear_on_submit=True):
        question = st.text_area("Enter your question", height=120)
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        try:
            results = retriever.retrieve(question, top_k=CONFIG.top_k)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            logger.exception("Retrieval error")
            return

        reranked = reranker.rerank(results, question)
        top_contexts = reranked[: CONFIG.top_k]

        prompt = build_prompt(question, top_contexts)
        try:
            answer = generator.generate(prompt)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            logger.exception("Generation error")
            return

        # Basic evaluation
        context_relevancy = evaluate_context_relevancy(question, [r["chunk"] for r in top_contexts])
        groundedness = evaluate_groundedness(answer, [r["chunk"] for r in top_contexts])
        answer_relevancy = evaluate_answer_relevancy(question, answer)

        metrics = {
            "context_relevancy": context_relevancy,
            "groundedness": groundedness,
            "answer_relevancy": answer_relevancy,
        }

        # Save session state
        st.session_state.history.append({"question": question, "answer": answer, "ts": time.time()})
        st.session_state.sources = top_contexts
        st.session_state.metrics = metrics

    # Render conversation history
    st.header("Conversation")
    for turn in reversed(st.session_state.history[-10:]):
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(f"**A:** {turn['answer']}")
        st.markdown("---")

    # Sources and metadata
    with st.expander("Retrieved Sources", expanded=False):
        if not st.session_state.sources:
            st.info("No sources retrieved yet.")
        else:
            for i, item in enumerate(st.session_state.sources, start=1):
                chunk = item.get("chunk", {})
                st.markdown(f"**Source {i}**: {chunk.get('document_name')} — {chunk.get('page_number')}")
                st.write(chunk.get("text")[:1000])
                st.json({"score": item.get("score"), "chunk_index": chunk.get("chunk_index"), "source_path": chunk.get("source_path")})

    # Evaluation metrics
    with st.expander("Evaluation Metrics", expanded=True):
        if st.session_state.metrics:
            st.json(st.session_state.metrics)
        else:
            st.info("No evaluation metrics yet.")


def main():
    st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
    sidebar_index_controls()

    tabs = st.tabs(["RAG Chatbot", "Message Classifier"])
    with tabs[0]:
        chat_interface()
    with tabs[1]:
        message_classifier_interface()


if __name__ == "__main__":
    main()
