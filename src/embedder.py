"""
Embedding pipeline using BAAI/bge-m3 and ChromaDB vector store.
"""
import os
import src.cache_config  # noqa: F401 — must be first to redirect all caches to D:

from sentence_transformers import SentenceTransformer
import chromadb


VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstore")
MODEL_NAME = "BAAI/bge-m3"


def get_embedding_model():
    """Load the BGE-M3 model."""
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_embedding_dimension()}")
    return model


def get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=VECTORSTORE_DIR)


def build_vectorstore(chunks: list[dict], force_rebuild: bool = False):
    """Embed all chunks and store in ChromaDB."""
    client = get_chroma_client()

    existing = [c.name for c in client.list_collections()]
    if "projects" in existing and not force_rebuild:
        collection = client.get_collection("projects")
        if collection.count() == len(chunks):
            print(f"Vector store already built with {collection.count()} chunks. Skipping.")
            return collection

    if "projects" in existing:
        client.delete_collection("projects")

    collection = client.create_collection(
        name="projects",
        metadata={"hnsw:space": "cosine"}
    )

    model = get_embedding_model()

    texts = [chunk['text'] for chunk in chunks]
    ids = [chunk['id'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]

    batch_size = 32
    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]

        embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")

    print(f"Vector store built with {collection.count()} chunks.")
    return collection


def query_vectorstore(query: str, model: SentenceTransformer, n_results: int = 10,
                      where_filter: dict = None, chunk_types: list[str] = None):
    """Query the vector store and return relevant chunks."""
    client = get_chroma_client()
    collection = client.get_collection("projects")

    where = {}
    if where_filter:
        where = where_filter
    if chunk_types:
        if where:
            where = {"$and": [where, {"chunk_type": {"$in": chunk_types}}]}
        else:
            where = {"chunk_type": {"$in": chunk_types}}

    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    return results
