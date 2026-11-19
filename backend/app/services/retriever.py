import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_collection = None
_model = None

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rag", "chroma_store")


def init_retriever():
    global _collection, _model

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=CHROMA_DIR)

        try:
            _collection = client.get_collection("medical_knowledge")
            logger.info(f"Loaded Chroma collection with {_collection.count()} chunks")
        except Exception:
            logger.warning("Chroma collection 'medical_knowledge' not found. Run build_index.py first.")
            _collection = None

        return True
    except Exception as e:
        logger.warning(f"Failed to init retriever: {e}")
        return False


def is_ready() -> bool:
    return _collection is not None


def retrieve(disease: str, query_text: str, k: int = 4) -> list[dict]:
    if _collection is None or _model is None:
        return []

    query_embedding = _model.encode([query_text]).tolist()

    try:
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where={"disease_slug": disease.lower().replace(" ", "_")},
        )
    except Exception:
        try:
            results = _collection.query(
                query_embeddings=query_embedding,
                n_results=k,
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

    docs = []
    if results and results.get("documents"):
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        # Chroma returns cosine *distance*; similarity is 1 - distance.
        distances = (results.get("distances") or [[]])[0]
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            distance = distances[i] if i < len(distances) else None
            docs.append(
                {
                    "text": doc,
                    "source_file": metadata.get("source_file", "unknown"),
                    "disease_slug": metadata.get("disease_slug", ""),
                    "chunk_index": metadata.get("chunk_index", i),
                    "similarity": (
                        round(1.0 - float(distance), 4) if distance is not None else None
                    ),
                }
            )

    return docs
