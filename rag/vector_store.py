"""
Agricultural Vector Store (ChromaDB)
Handles embedding, indexing, and searching of agricultural knowledge chunks.

Uses sentence-transformers/all-MiniLM-L6-v2 for local embeddings (free, no API).
ChromaDB stores everything locally — zero ongoing cost.
"""

import os
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

# Attempt imports with fallback messaging
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ── Default Paths ───────────────────────────────────────────────────────

DEFAULT_CHROMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "chroma_db"
)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "_chunked_output.json"
)


# ── Embedding Function ──────────────────────────────────────────────────

class LocalEmbeddingFunction:
    """Wrapper that makes sentence-transformers compatible with ChromaDB."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


# ── Vector Store ────────────────────────────────────────────────────────

class AgriculturalVectorStore:
    """
    Manages the ChromaDB collection for agricultural knowledge.
    
    Features:
    - Creates/loads persistent ChromaDB collection
    - Indexes chunks with metadata
    - Hybrid search: semantic (vector) + metadata filtering
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "agriculture_knowledge",
        embedding_model: str = DEFAULT_MODEL_NAME,
    ):
        if chromadb is None:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        self.persist_directory = persist_directory or DEFAULT_CHROMA_PATH
        self.collection_name = collection_name
        self.embedding_function = LocalEmbeddingFunction(embedding_model)

        # Create persistent ChromaDB client
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection (no embedding function passed — we handle embeddings ourselves)
        self.collection = None
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
            )
        except Exception:
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                )
            except Exception:
                self.client.delete_collection(collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name,
                )

    @property
    def count(self) -> int:
        """Number of chunks in the collection."""
        return self.collection.count()

    def index_chunks(self, chunks: List[Dict], overwrite: bool = False):
        """
        Index a list of chunk dicts into ChromaDB.
        
        Args:
            chunks: List of chunk dicts with 'chunk_id', 'content', and metadata
            overwrite: If True, drop existing collection and re-index
        """
        if overwrite and self.count > 0:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
            )

        if not chunks:
            print("No chunks to index.")
            return

        ids = []
        documents = []
        metadatas = []
        all_embeddings = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id:
                continue
            content = chunk.get("content", "")
            if not content.strip():
                continue

            ids.append(chunk_id)
            documents.append(content)

            # Build metadata dict (only fields ChromaDB can store as strings/numbers)
            metadata = {
                "crop_name": chunk.get("crop_name", "general"),
                "crop_urdu": chunk.get("crop_urdu", ""),
                "category": chunk.get("category", "GENERAL"),
                "season": chunk.get("season", "Both"),
                "pakistan_region": chunk.get("pakistan_region", "All"),
                "urgency_indicator": chunk.get("urgency_indicator", "planned"),
            }

            # Add disease/pest names if present
            if chunk.get("disease_name"):
                metadata["disease_name"] = chunk["disease_name"]
            if chunk.get("pest_name"):
                metadata["pest_name"] = chunk["pest_name"]

            # Store keywords and chemicals as comma-separated
            keywords = chunk.get("symptoms_keywords", [])
            if keywords:
                metadata["symptoms_keywords"] = ",".join(keywords[:10])
            chemicals = chunk.get("chemical_names", [])
            if chemicals:
                metadata["chemical_names"] = ",".join(chemicals[:8])

            metadatas.append(metadata)

        # Pre-compute all embeddings at once (much faster than individual calls)
        all_embeddings = self.embedding_function(documents)

        # Add in batches of 100 to avoid memory issues
        batch_size = 100
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=all_embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"Indexed {total} chunks into '{self.collection_name}' collection.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        crop_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search the vector store with optional metadata filters.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            crop_filter: Filter by crop name (e.g., "wheat")
            category_filter: Filter by category (e.g., "DISEASE")
            region_filter: Filter by region (e.g., "Punjab")
        
        Returns:
            List of dicts with chunk_id, content, metadata, and distance score
        """
        # Build ChromaDB filter
        where_filter = {}
        if crop_filter:
            where_filter["crop_name"] = crop_filter.lower()
        if category_filter:
            where_filter["category"] = category_filter.upper()
        if region_filter:
            where_filter["pakistan_region"] = region_filter

        # ChromaDB requires $and for multiple filters
        if len(where_filter) > 1:
            where_clause = {"$and": [{k: v} for k, v in where_filter.items()]}
        elif len(where_filter) == 1:
            where_clause = where_filter
        else:
            where_clause = None

        # Pre-compute query embedding using our local model
        query_embedding = self.embedding_function([query])[0]

        # Execute search with pre-computed embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            where=where_clause,
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "chunk_id": doc_id,
                    "content": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx] if results["distances"] else None,
                })

        return formatted

    def search_by_crop(self, query: str, crop_name: str, top_k: int = 3) -> List[Dict]:
        """Convenience: search only within a specific crop's knowledge."""
        return self.search(query, top_k=top_k, crop_filter=crop_name)

    def search_by_category(self, query: str, category: str, top_k: int = 3) -> List[Dict]:
        """Convenience: search only within a specific category."""
        return self.search(query, top_k=top_k, category_filter=category)

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.count == 0:
            return {"total_chunks": 0}

        # Get all metadatas for statistics
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data["metadatas"]

        categories = {}
        crops = {}
        regions = {}

        for meta in metadatas:
            cat = meta.get("category", "UNKNOWN")
            categories[cat] = categories.get(cat, 0) + 1
            crop = meta.get("crop_name", "unknown")
            crops[crop] = crops.get(crop, 0) + 1
            region = meta.get("pakistan_region", "Unknown")
            regions[region] = regions.get(region, 0) + 1

        return {
            "total_chunks": self.count,
            "categories": dict(sorted(categories.items())),
            "crops": dict(sorted(crops.items())),
            "regions": dict(sorted(regions.items())),
        }


# ── Convenience Initializer ─────────────────────────────────────────────

def initialize_store(
    data_dir: str = None,
    chroma_path: str = None,
    force_reindex: bool = False,
) -> AgriculturalVectorStore:
    """
    One-call setup: load knowledge files, process into chunks, index into ChromaDB.
    
    Args:
        data_dir: Path to data directory with .txt files
        chroma_path: Path to store ChromaDB
        force_reindex: If True, delete existing index and re-create
    
    Returns:
        Initialized AgriculturalVectorStore ready for search
    """
    from .knowledge_processor import process_all_documents

    store = AgriculturalVectorStore(
        persist_directory=chroma_path or DEFAULT_CHROMA_PATH,
    )

    # Only index if empty or forced
    if store.count == 0 or force_reindex:
        print("Processing documents into chunks...")
        chunks = process_all_documents(data_dir)
        print(f"Generated {len(chunks)} chunks.")
        
        print("Indexing into ChromaDB...")
        store.index_chunks(chunks, overwrite=force_reindex)
    else:
        print(f"Collection already has {store.count} chunks. Use force_reindex=True to rebuild.")

    return store


if __name__ == "__main__":
    # Quick test
    store = initialize_store(force_reindex=True)
    print("\n--- Collection Stats ---")
    stats = store.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Categories: {stats['categories']}")
    print(f"Crops: {stats['crops']}")

    # Test search
    print("\n--- Test Search: 'wheat yellow rust treatment' ---")
    results = store.search("wheat yellow rust treatment", top_k=3)
    for r in results:
        print(f"  [{r['chunk_id']}] score={r['distance']:.4f}")
        print(f"  {r['content'][:120]}...")
        print()
