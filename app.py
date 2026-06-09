"""
Streamlit UI for the RAG-Powered Project Intelligence Assistant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import src.cache_config  # noqa: F401 — forces all caches to D:

import streamlit as st
from src.config import (
    LLM_PROVIDER, LLM_MODEL, OLLAMA_URL, OLLAMA_API_KEY, API_KEY, API_BASE,
    TOP_K_RETRIEVE, TOP_K_RERANK, PPTX_PATH
)
from src.slide_parser import parse_pptx
from src.chunker import chunk_projects
from src.embedder import build_vectorstore
from src.rag_engine import RAGEngine

# Page config
st.set_page_config(
    page_title="Project Intelligence Assistant",
    page_icon="🔍",
    layout="wide",
)

st.title("RAG-Powered Project Intelligence Assistant")
st.caption("Query 100 real-world projects using natural language. Answers are grounded in source slides.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    PROVIDERS = ["ollama_cloud", "ollama_local", "openai_compatible"]
    provider = st.selectbox(
        "LLM Provider",
        PROVIDERS,
        index=PROVIDERS.index(LLM_PROVIDER) if LLM_PROVIDER in PROVIDERS else 0,
    )

    if provider == "ollama_cloud":
        model_name = st.text_input("Model", value=LLM_MODEL)
        api_key = st.text_input("Ollama API Key", value=OLLAMA_API_KEY, type="password")
        ollama_url = None
        api_base = None
    elif provider == "ollama_local":
        model_name = st.text_input("Model", value=LLM_MODEL)
        ollama_url = st.text_input("Ollama URL", value=OLLAMA_URL)
        api_key = None
        api_base = None
    else:
        model_name = st.text_input("Model", value=LLM_MODEL)
        api_base = st.text_input("API Base URL", value=API_BASE)
        api_key = st.text_input("API Key", value=API_KEY, type="password")
        ollama_url = None

    st.divider()
    st.header("Retrieval Settings")
    top_k_retrieve = st.slider("Chunks to retrieve", 5, 40, TOP_K_RETRIEVE)
    top_k_rerank = st.slider("Chunks after re-ranking", 3, 15, TOP_K_RERANK)

    st.divider()
    st.header("Metadata Filters")
    filter_domain = st.text_input("Filter by domain (optional)", placeholder="e.g., Healthcare")
    filter_max_team = st.number_input("Max team size (0 = no filter)", min_value=0, max_value=50, value=0)
    filter_max_duration = st.number_input("Max duration in months (0 = no filter)", min_value=0, max_value=24, value=0)

    st.divider()
    if st.button("Rebuild Vector Store", type="secondary"):
        st.session_state.pop('vectorstore_built', None)
        st.session_state.pop('rag_engine', None)
        st.rerun()


# Initialize / build vectorstore
@st.cache_resource(show_spinner="Parsing slides and building vector store...")
def init_vectorstore():
    projects = parse_pptx(PPTX_PATH)
    chunks = chunk_projects(projects)
    build_vectorstore(chunks)
    return len(projects), len(chunks)


@st.cache_resource(show_spinner="Loading RAG engine (embedding + re-ranker models)...")
def init_rag_engine(_provider, _model, _ollama_url, _api_key, _api_base):
    return RAGEngine(
        llm_provider=_provider,
        llm_model=_model,
        ollama_url=_ollama_url or OLLAMA_URL,
        api_key=_api_key,
        api_base=_api_base,
    )


# Build vectorstore
num_projects, num_chunks = init_vectorstore()
st.sidebar.success(f"Indexed {num_projects} projects ({num_chunks} chunks)")

# Init RAG engine
engine = init_rag_engine(provider, model_name, ollama_url, api_key, api_base)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])} projects) | Confidence: {msg['confidence']:.0%}"):
                for src in msg['sources']:
                    st.markdown(
                        f"- **{src['project_id']}**: {src['title']} | "
                        f"Domain: {src['domain']} | Slides: {src['slides']} | "
                        f"Relevance: {src['relevance_score']:.4f}"
                    )

# Example queries
if not st.session_state.messages:
    st.markdown("### Example queries:")
    cols = st.columns(2)
    examples = [
        "Show me all projects involving NLP or large language models.",
        "Which projects were delivered for the healthcare or pharma domain?",
        "What solution approach was used for supply chain optimization?",
        "List projects where the team size was under 5 people and duration under 3 months.",
        "What technologies were used across data engineering projects?",
        "Which project is most similar to building a customer churn prediction model?",
    ]
    for i, example in enumerate(examples):
        col = cols[i % 2]
        if col.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

# Handle pending query from example buttons
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
else:
    query = st.chat_input("Ask about projects...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build metadata filter
    where_filter = None
    filters = []
    if filter_domain:
        filters.append({"domain": {"$contains": filter_domain}})
    if filter_max_team > 0:
        filters.append({"team_size": {"$lte": filter_max_team}})
    if filter_max_duration > 0:
        filters.append({"duration_months": {"$lte": filter_max_duration}})

    if len(filters) == 1:
        where_filter = filters[0]
    elif len(filters) > 1:
        where_filter = {"$and": filters}

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and analyzing projects..."):
            result = engine.query(
                user_query=query,
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                where_filter=where_filter,
            )

        st.markdown(result['answer'])

        # Grounding verification badge
        verification = result.get('verification', {})
        if verification.get('is_grounded', True):
            grounding_label = "Grounded"
        else:
            grounding_label = "Partially Grounded"

        with st.expander(
            f"Sources ({len(result['sources'])} projects) | "
            f"Confidence: {result['confidence']:.0%} | "
            f"{grounding_label} ({verification.get('grounding_ratio', 1.0):.0%}) | "
            f"Vector: {result.get('num_chunks_retrieved', 0)} + "
            f"Keyword: {result.get('num_keyword_matches', 0)} -> "
            f"Re-ranked: {result.get('num_chunks_after_rerank', 0)} -> "
            f"Context: {result.get('num_projects_in_context', 0)} projects"
        ):
            for src in result['sources']:
                st.markdown(
                    f"- **{src['project_id']}**: {src['title']} | "
                    f"Domain: {src['domain']} | Slides: {src['slides']} | "
                    f"Relevance: {src['relevance_score']:.4f}"
                )

            # Show query analysis
            qa = result.get('query_analysis', {})
            if qa.get('keywords') or qa.get('metadata_filters'):
                st.caption(
                    f"Auto-detected: keywords={qa.get('keywords', [])} | "
                    f"filters={qa.get('metadata_filters', {})} | "
                    f"listing={qa.get('is_listing_query')} | "
                    f"similarity={qa.get('is_similarity_query')}"
                )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "sources": result['sources'],
        "confidence": result['confidence'],
    })
