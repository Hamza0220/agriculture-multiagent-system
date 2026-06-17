"""
Agricultural Knowledge Retriever (M1-1 Contract Export)
Exports the single function that Member 2's agent pipeline calls.

Contract function:
    retrieve_crop_knowledge(query, crop_name, category, top_k) -> str
        Returns compressed agricultural context string ready for agent consumption.

Usage from agent_pipeline.py:
    from rag.retriever import retrieve_crop_knowledge
    context = retrieve_crop_knowledge(query="yellow stripes on wheat leaves", crop_name="wheat")
"""

import os
import json
from typing import Optional, List, Dict
from pathlib import Path


# ── Lazy-loaded store ──────────────────────────────────────────────────

_store = None


def _get_store():
    """Lazy-load and cache the vector store singleton."""
    global _store
    if _store is None:
        from .vector_store import AgriculturalVectorStore
        chroma_path = os.getenv(
            "CHROMA_DB_PATH",
            os.path.join(os.path.dirname(__file__), "..", "chroma_db"),
        )
        _store = AgriculturalVectorStore(persist_directory=chroma_path)
    return _store


def _compress_knowledge(chunks: List[Dict], max_tokens: int = 800) -> str:
    """
    Compress retrieved chunks into a concise, actionable context string.
    Follows Prompt M1-4 compression rules:
    - Keep all specific numbers, product names, dosages, timing info
    - Remove academic text, citations, historical info
    - Prioritize cheapest/local solutions first
    - Add simple Urdu equivalents for technical terms
    """
    if not chunks:
        return "No relevant agricultural knowledge found for this query."

    sections = []
    seen_content = set()
    token_budget = max_tokens

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        metadata = chunk.get("metadata", {})
        score = chunk.get("distance", 0)

        # Skip near-duplicates
        content_preview = content[:100]
        if content_preview in seen_content:
            continue
        seen_content.add(content_preview)

        # Estimate tokens
        estimated_tokens = len(content) // 4
        if estimated_tokens > token_budget:
            # Truncate at sentence boundary
            sentences = content.split(".")
            truncated = ""
            for sent in sentences:
                if (len(truncated) + len(sent)) // 4 <= token_budget:
                    truncated += sent + "."
                else:
                    break
            content = truncated.strip() if truncated else content[:max_tokens * 4]

        # Build section header from metadata
        header_parts = []
        crop = metadata.get("crop_name", "")
        if crop and crop != "general":
            crop_urdu = metadata.get("crop_urdu", "")
            if crop_urdu:
                header_parts.append(f"{crop.title()} ({crop_urdu})")
            else:
                header_parts.append(crop.title())

        category = metadata.get("category", "")
        if category and category != "GENERAL":
            header_parts.append(category.title())

        disease = metadata.get("disease_name", "")
        if disease:
            disease_urdu = metadata.get("disease_urdu", "")
            if disease_urdu:
                header_parts.append(f"{disease} ({disease_urdu})")
            else:
                header_parts.append(disease)

        header = " | ".join(header_parts) if header_parts else "Agricultural Knowledge"
        section = f"---{header}---\n{content}"
        sections.append(section)

        token_budget -= estimated_tokens
        if token_budget <= 50:
            break

    return "\n\n".join(sections)


# ── CONTRACT FUNCTION (called by Member 2's agent_pipeline.py) ─────────

def retrieve_crop_knowledge(
    query: str,
    crop_name: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
    use_relevance_filter: bool = True,
    use_compression: bool = True,
) -> str:
    """
    Retrieve and compress agricultural knowledge for a farmer's query.

    This is the integration contract function. Member 2 calls this to get
    context for Crop Doctor and Irrigation Advisor agents.

    Args:
        query: Farmer's query or search text (e.g., "yellow stripes on wheat leaves")
        crop_name: Specific crop to filter by (e.g., "wheat", "rice", "cotton", etc.)
                   If None, searches across all crops.
        category: Knowledge category filter: DISEASE, PEST, IRRIGATION, FERTILIZER,
                  SOIL, HARVESTING, STORAGE, VARIETIES. If None, searches all.
        top_k: Number of chunks to retrieve (default: 5)
        use_relevance_filter: If True, applies M1-3 relevance scoring/filtering

    Returns:
        Compressed context string (max ~800 tokens) ready for agent use.
        Returns a fallback message if nothing relevant is found.

    Example:
        >>> context = retrieve_crop_knowledge(
        ...     query="meri gandum ki fasal peeli ho rahi hai",
        ...     crop_name="wheat",
        ...     category="DISEASE"
        ... )
        >>> print(context)
        ---Wheat (Gandum) | Disease | Yellow Rust (Zard Zang)---
        Propiconazole 25EC (Tilt 250EC) — 200ml per acre in 100L water...
    """
    try:
        store = _get_store()

        # If the collection is empty, try to initialize from knowledge files
        if store.count == 0:
            from .vector_store import initialize_store
            store = initialize_store()

        # Perform search
        results = store.search(
            query=query,
            top_k=top_k,
            crop_filter=crop_name,
            category_filter=category,
        )

        # If no results with crop filter, retry without (fallback)
        if not results and crop_name:
            results = store.search(query=query, top_k=top_k // 2)
            if results:
                if use_compression:
                    from .context_compressor import compress_context
                    from .relevance_filter import filter_chunks
                    scored = filter_chunks(results, query, crop_name, use_llm=False)
                    context = compress_context(
                        scored.get("filtered_chunks", results),
                        query, crop_name or "general", use_llm=False,
                    )
                elif use_relevance_filter:
                    from .relevance_filter import filter_and_format_context
                    context = filter_and_format_context(
                        chunks=results,
                        user_query=query,
                        crop_name=crop_name,
                        use_llm=False,
                    )
                else:
                    context = _compress_knowledge(results)
                return (
                    f"NOTE: No {crop_name}-specific results found. "
                    f"Showing general agricultural knowledge:\n\n{context}"
                )

        if not results:
            return "No relevant agricultural knowledge found for this query."

        # Apply relevance filter + compression
        if use_compression:
            from .context_compressor import compress_retrieved_context
            return compress_retrieved_context(
                chunks=results,
                user_query=query,
                crop_name=crop_name,
                use_llm=False,
            )
        elif use_relevance_filter:
            from .relevance_filter import filter_and_format_context
            return filter_and_format_context(
                chunks=results,
                user_query=query,
                crop_name=crop_name,
                use_llm=False,
            )
        else:
            return _compress_knowledge(results)

    except ImportError as e:
        return f"Knowledge base unavailable: {e}. Please install required dependencies."
    except Exception as e:
        return f"Error retrieving agricultural knowledge: {e}"


# ── Convenience wrappers for common query types ────────────────────────

def get_disease_info(
    query: str, crop_name: Optional[str] = None, top_k: int = 3
) -> str:
    """Quick lookup for disease information."""
    return retrieve_crop_knowledge(
        query=query, crop_name=crop_name, category="DISEASE", top_k=top_k
    )


def get_pest_info(
    query: str, crop_name: Optional[str] = None, top_k: int = 3
) -> str:
    """Quick lookup for pest information."""
    return retrieve_crop_knowledge(
        query=query, crop_name=crop_name, category="PEST", top_k=top_k
    )


def get_irrigation_info(
    query: str, crop_name: Optional[str] = None, top_k: int = 2
) -> str:
    """Quick lookup for irrigation information."""
    return retrieve_crop_knowledge(
        query=query, crop_name=crop_name, category="IRRIGATION", top_k=top_k
    )


def get_fertilizer_info(
    query: str, crop_name: Optional[str] = None, top_k: int = 2
) -> str:
    """Quick lookup for fertilizer recommendations."""
    return retrieve_crop_knowledge(
        query=query, crop_name=crop_name, category="FERTILIZER", top_k=top_k
    )


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Simple interactive test
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your agriculture query: ")

    print("\n=== Query ===")
    print(query)

    print("\n=== Retrieved Knowledge ===")
    context = retrieve_crop_knowledge(query)
    print(context)

    print("\n--- Try with crop filter ---")
    crop = input("\nEnter crop name (or press Enter to skip): ").strip()
    if crop:
        context = retrieve_crop_knowledge(query, crop_name=crop)
        print(context)
