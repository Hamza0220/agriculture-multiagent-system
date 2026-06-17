# RAG Pipeline Package
"""Kisaan Dost Agricultural RAG Pipeline.

Auto-loads .env from project root on first import.
"""

import os

# Auto-load .env from project root (one level up from kisaan_dost/)
_env_loaded = False

def _load_env():
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    
    # Check if already loaded (GROQ_API_KEY would be set)
    if os.getenv("GROQ_API_KEY") and "your_" not in os.getenv("GROQ_API_KEY", ""):
        return
    
    try:
        from dotenv import load_dotenv
        
        # Try project root (one up from kisaan_dost/)
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
        
        # Try current working directory
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
            
        # Try parent of cwd
        env_path = os.path.join(os.path.dirname(os.getcwd()), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
    except ImportError:
        pass  # python-dotenv not installed
    except Exception:
        pass

_load_env()

from .knowledge_processor import process_all_documents, load_knowledge_files
from .vector_store import AgriculturalVectorStore, initialize_store
from .retriever import retrieve_crop_knowledge
from .query_expander import expand_query, expand_and_retrieve
from .relevance_filter import filter_chunks, filter_and_format_context
from .context_compressor import compress_context, compress_retrieved_context

__all__ = [
    "process_all_documents",
    "load_knowledge_files",
    "AgriculturalVectorStore",
    "initialize_store",
    "retrieve_crop_knowledge",
    "expand_query",
    "expand_and_retrieve",
    "filter_chunks",
    "filter_and_format_context",
    "compress_context",
    "compress_retrieved_context",
]