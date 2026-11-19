import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start + overlap >= len(words):
            break
    return chunks


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating Chroma collection...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection("medical_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="medical_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    md_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".md")]
    print(f"Found {len(md_files)} knowledge base files")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for filename in sorted(md_files):
        disease_slug = filename.replace(".md", "")
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{disease_slug}_{i}"
            all_chunks.append(chunk)
            all_metadatas.append({
                "disease_slug": disease_slug,
                "source_file": filename,
                "chunk_index": i,
            })
            all_ids.append(chunk_id)

        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)

    print("Adding to Chroma...")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:end],
            documents=all_chunks[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=all_metadatas[i:end],
        )

    print(f"\nDone! Indexed {collection.count()} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
