"""Build a RAG index from the bundled knowledge-base Markdown files.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
normalized embeddings, and a FAISS inner-product index.

Writes faiss_index.bin and chunks.json to NUTRITION_AI_RAG_DIR when set,
or to artifacts/rag under the repository root.

Run:
    python -m app.ai.rag.build_index

The retriever uses the same directory policy as this builder.
"""

import json
import os

import numpy as np

from app.paths import KNOWLEDGE_BASE_DIR, get_rag_dir

KB_DIR = KNOWLEDGE_BASE_DIR
OUTPUT_DIR = get_rag_dir()

INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(OUTPUT_DIR, "chunks.json")

KB_FILES = [
    "nutrition_basics.md",
    "pre_post_workout.md",
    "weight_management.md",
    "exercise_nutrition.md",
]


def _load_chunks() -> list:
    """Split each markdown file into non-empty paragraphs."""
    chunks = []
    for fname in KB_FILES:
        path = os.path.join(KB_DIR, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Skip pure heading lines (they have no retrieval value on their own)
            if para.startswith("#") and "\n" not in para:
                continue
            chunks.append({"text": para, "source_file": fname})
    return chunks


def build() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from app.ai.retrieval.runtime import prepare_transformers

    prepare_transformers()

    import faiss
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Multilingual model — handles Arabic + English queries equally well.
    from app.ai.runtime import configuration_error, get_device, get_runtime

    if get_runtime() == "zerogpu":
        raise configuration_error(
            "Build RAG assets offline in cpu or cuda mode before deployment."
        )

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=get_device(),
    )

    print("Loading knowledge-base chunks...")
    chunks = _load_chunks()
    print(f"  {len(chunks)} chunks across {len(KB_FILES)} files")

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(
        dim
    )  # inner product on normalised vectors = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Index saved to {INDEX_PATH}")
    print(f"Chunks saved to {CHUNKS_PATH}")
    print("Done.")


if __name__ == "__main__":
    build()
