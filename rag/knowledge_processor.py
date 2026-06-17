"""
Agricultural Knowledge Base Processor (M1-1)
Chunks agricultural texts for RAG vector database following strict chunking rules.

Chunking Rules:
- One topic per chunk (no mixing DISEASE + IRRIGATION)
- Max 500 tokens per chunk, min 80 tokens
- Every chunk has metadata header
- Pakistan-specific info preserved exactly
- Urdu/local names included alongside English
- Specific numbers preserved (never generalized)
"""

import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional
import re


# ── Constants ──────────────────────────────────────────────────────────

CHUNK_CATEGORIES = [
    "DISEASE", "PEST", "IRRIGATION", "FERTILIZER",
    "SOIL", "HARVESTING", "STORAGE", "VARIETIES"
]

SEASONS = ["Rabi", "Kharif", "Both"]
REGIONS = ["Punjab", "Sindh", "KPK", "Balochistan", "All"]

TARGET_CROPS = [
    ("wheat", "gandum"), ("rice", "chawal"), ("cotton", "kapas"),
    ("sugarcane", "ganna"), ("maize", "makki"), ("tomato", "tamatar"),
    ("onion", "pyaaz"), ("potato", "aloo"), ("mango", "aam"),
    ("chilli", "mirch"), ("mustard", "sarson"), ("general", "general")
]


def estimate_token_count(text: str) -> int:
    """Rough token estimation (~4 chars = 1 token for English/Urdu mixed)."""
    return len(text) // 4


class ChunkBuilder:
    """Builds metadata-rich chunks from agricultural text."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.chunks: List[Dict] = []
        self._current_chunk: Dict = None
        self._current_lines: List[str] = []
        # Track crop context from CROP: headers for subsequent disease/pest chunks
        self._context_crop_name = "general"
        self._context_crop_urdu = "general"
        self._context_season = "Both"
        self._context_region = "All"

    def process_document(self, text: str) -> List[Dict]:
        """Process a full agricultural document into structured chunks."""
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip visual separators (==== and --- lines) — they are NOT chunk boundaries
            if line.startswith("==") or line.startswith("---"):
                i += 1
                continue

            # Detect a crop/category header line (actual chunk boundary)
            if self._is_chunk_header(line):
                self._finalize_chunk()
                metadata = self._parse_metadata_from_header(line, lines, i)

                # If this is a CROP header, save context for subsequent chunks
                if line.upper().startswith("CROP:"):
                    self._context_crop_name = metadata.get("crop_name", "general")
                    self._context_crop_urdu = metadata.get("crop_urdu", "general")
                    self._context_season = metadata.get("season", "Both")
                    self._context_region = metadata.get("pakistan_region", "All")

                # Apply context to non-CROP chunks
                if not line.upper().startswith("CROP:"):
                    if metadata.get("crop_name") == "general":
                        metadata["crop_name"] = self._context_crop_name
                        metadata["crop_urdu"] = self._context_crop_urdu
                    if metadata.get("season") == "Both":
                        metadata["season"] = self._context_season
                    if metadata.get("pakistan_region") == "All":
                        metadata["pakistan_region"] = self._context_region

                self._current_chunk = metadata
                self._current_lines = []
                i += 1
                continue

            # Accumulate content lines
            if self._current_chunk and line:
                self._current_lines.append(line)

            i += 1

        # Finalize last chunk
        self._finalize_chunk()

        return self.chunks

    def _is_chunk_header(self, line: str) -> bool:
        """Check if a line indicates a new chunk boundary.

        Valid boundaries:
        - DISEASE:, PEST:, IRRIGATION:, etc. headers
        - CROP: (all caps) for section starts
        - NOT === lines (those are just section dividers, not chunk boundaries)
        """
        if re.match(r"^(DISEASE|PEST|IRRIGATION|FERTILIZER|SOIL|HARVESTING|STORAGE|VARIETIES)\s*:", line, re.IGNORECASE):
            return True
        # Crop SECTION headers are ALL CAPS "CROP: Wheat..." — NOT "Crop: wheat..." (metadata)
        if line.startswith("CROP:"):
            return True
        return False

    def _parse_metadata_from_header(self, line: str, lines: List[str], idx: int) -> Dict:
        """Parse metadata from header lines and following context (2-3 lines)."""
        meta = {
            "crop_name": "general",
            "crop_urdu": "general",
            "category": "GENERAL",
            "season": "Both",
            "pakistan_region": "All",
            "symptoms_keywords": [],
            "chemical_names": [],
            "urgency_indicator": "planned",
            "disease_name": "",
            "disease_urdu": "",
            "pest_name": "",
            "pest_urdu": "",
            "source_file": os.path.basename(self.source_file),
        }

        # Parse the main header line
        if "DISEASE:" in line:
            meta["category"] = "DISEASE"
            # Extract disease name
            match = re.search(r"DISEASE:\s*(.+?)(\s*—\s*|\s*$)", line)
            if match:
                name_part = match.group(1).strip()
                # Extract English name and Urdu name if present
                urdu_match = re.search(r'"([^"]+)"', line)
                if urdu_match:
                    meta["disease_urdu"] = urdu_match.group(1)
                meta["disease_name"] = name_part

        elif "PEST:" in line:
            meta["category"] = "PEST"
            match = re.search(r"PEST:\s*(.+?)(\s*—\s*|\s*$)", line)
            if match:
                name_part = match.group(1).strip()
                urdu_match = re.search(r'"([^"]+)"', line)
                if urdu_match:
                    meta["pest_urdu"] = urdu_match.group(1)
                meta["pest_name"] = name_part

        elif line.upper().startswith("CROP:"):
            # Parse "CROP: Wheat (gandum) | Season: Rabi ... | Major Regions: Punjab..."
            meta["category"] = "GENERAL"
            
            # Extract crop name and Urdu name
            # Format: CROP: Wheat (gandum) or CROP: Rice (chawal/dhan)
            crop_part = line[5:].strip()  # After "CROP:"
            
            # Try to find the crop name before any "|" separator
            crop_info = crop_part.split("|")[0].strip()
            crop_match = re.match(r"([^(]+)\s*\(([^)]+)\)", crop_info)
            if crop_match:
                meta["crop_name"] = crop_match.group(1).strip().lower()
                # Urdu name — take first word before / or just the whole thing
                urdu_text = crop_match.group(2).strip()
                # Get primary Urdu name (before any slash)
                meta["crop_urdu"] = urdu_text.split("/")[0].strip()
            
            # Parse Season from CROP line
            season_match = re.search(r"Season:\s*(\w+)", crop_part, re.IGNORECASE)
            if season_match:
                s = season_match.group(1).capitalize()
                if s in SEASONS:
                    meta["season"] = s

        # Scan next few lines for additional metadata
        for j in range(idx + 1, min(idx + 4, len(lines))):
            next_line = lines[j].strip()

            crop_match = re.search(r"Crop:\s*([^(]+)\s*\(?([^)]*)\)?", next_line, re.IGNORECASE)
            if crop_match:
                meta["crop_name"] = crop_match.group(1).strip().lower()
                urdu_part = crop_match.group(2).strip() if crop_match.group(2) else ""
                if urdu_part:
                    meta["crop_urdu"] = urdu_part

            cat_match = re.search(r"Category:\s*(\w+)", next_line, re.IGNORECASE)
            if cat_match and cat_match.group(1).upper() in CHUNK_CATEGORIES:
                meta["category"] = cat_match.group(1).upper()

            season_match = re.search(r"Season:\s*(\w+)", next_line, re.IGNORECASE)
            if season_match:
                s = season_match.group(1).capitalize()
                if s in SEASONS:
                    meta["season"] = s

            region_match = re.search(r"Region:\s*(.+)", next_line, re.IGNORECASE)
            if region_match:
                meta["pakistan_region"] = region_match.group(1).strip()

            urgency_match = re.search(r"Urgency:\s*(\w+)", next_line, re.IGNORECASE)
            if urgency_match:
                meta["urgency_indicator"] = urgency_match.group(1).lower()

        return meta

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract symptom keywords from text."""
        keywords = set()
        # Common symptom patterns
        symptom_patterns = [
            r"(yellow|brown|black|white|gray|purple|orange|red|dark)\s+\w+",
            r"\w+\s+(spots|lesions|stripes|pustules|powder|rot|wilt|blight)",
            r"(leaves|stems|fruit|roots|pods|grains)\s+\w+",
        ]
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    keywords.add(m[0])
                else:
                    keywords.add(m)
        return list(keywords)[:10]

    def _extract_chemicals(self, text: str) -> List[str]:
        """Extract chemical/pesticide names from text."""
        chemicals = set()
        # Match chemical names (trade names, active ingredients)
        patterns = [
            r"([A-Z][a-z]+[a-z]*\s?\d+[A-Z]{0,3})",  # Propiconazole 25EC
            r"\(([A-Z][a-zA-Z]+)\)",  # (Tilt), (Confidor)
            r"\b(\w+)\s*—\s*Rs\.",  # Product — Rs.
            r"(Carbendazim|Mancozeb|Imidacloprid|Chlorantraniliprole|"
            r"Propiconazole|Tebuconazole|Azoxystrobin|Difenoconazole|"
            r"Spinosad|Lambda|Dimethoate|Metalaxyl|Chlorpyrifos|"
            r"Buprofezin|Pyriproxyfen|Tricyclazole|Carbofuran|Fipronil"
            r"Hexaconazole|Carbendazim|Copper|Sulfur|Neem)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    chemicals.add(m[0])
                else:
                    chemicals.add(m)
        return list(chemicals)[:8]

    def _build_chunk_id(self, crop: str, category: str, content: str, index: int = 0) -> str:
        """Create a unique chunk ID using crop, category, and unique content hash."""
        hash_part = hashlib.md5(content.encode()).hexdigest()[:12]
        cat_short = category.lower()[:4]
        crop_clean = crop.lower().replace(" ", "_")[:10]
        return f"{crop_clean}_{cat_short}_{hash_part}"

    def _finalize_chunk(self):
        """Finalize and add the current chunk to the list."""
        if not self._current_chunk or not self._current_lines:
            self._current_chunk = None
            self._current_lines = []
            return

        content = "\n".join(self._current_lines).strip()
        token_count = estimate_token_count(content)

        # Skip if too small (unless we can merge — simple skip for now)
        if token_count < 20:
            self._current_chunk = None
            self._current_lines = []
            return

        # Truncate if too long (500 token limit)
        if token_count > 500:
            # Smart truncate at sentence boundary
            sentences = content.split(".")
            truncated = ""
            for sent in sentences:
                if estimate_token_count(truncated + sent + ".") <= 500:
                    truncated += sent + "."
                else:
                    break
            content = truncated.strip() if truncated else content[:2000]

        chunk = dict(self._current_chunk)
        chunk["content"] = content
        chunk["chunk_id"] = self._build_chunk_id(
            chunk["crop_name"],
            chunk["category"],
            content
        )

        # Extract keywords and chemicals from actual content
        if not chunk["symptoms_keywords"]:
            chunk["symptoms_keywords"] = self._extract_keywords(content)
        if not chunk["chemical_names"]:
            chunk["chemical_names"] = self._extract_chemicals(content)

        self.chunks.append(chunk)
        self._current_chunk = None
        self._current_lines = []


# ── Public API ──────────────────────────────────────────────────────────

def load_knowledge_files(data_dir: str = None) -> Dict[str, str]:
    """
    Load all knowledge base text files from the data directory.
    
    Args:
        data_dir: Path to data directory. Defaults to ../data relative to this file.
    
    Returns:
        Dict mapping filename -> file content
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    data_path = Path(data_dir)
    if not data_path.exists():
        # Try relative path as fallback
        data_path = Path("data")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = {}
    for filepath in sorted(data_path.glob("*.txt")):
        if filepath.name.startswith("_"):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            documents[filepath.name] = f.read()

    return documents


def process_all_documents(data_dir: str = None) -> List[Dict]:
    """
    Process all knowledge base documents into structured chunks.
    
    Returns:
        List of chunk dicts with metadata and content.
    """
    documents = load_knowledge_files(data_dir)
    all_chunks = []

    for filename, text in documents.items():
        builder = ChunkBuilder(filename)
        chunks = builder.process_document(text)
        all_chunks.extend(chunks)

    return all_chunks


def chunk_metadata_to_json(chunks: List[Dict]) -> str:
    """Convert chunks to pretty-printed JSON string."""
    return json.dumps(chunks, indent=2, ensure_ascii=False)


def print_chunk_summary(chunks: List[Dict]):
    """Print a summary of processed chunks."""
    print(f"Total chunks: {len(chunks)}")
    
    categories = {}
    crops = {}
    for c in chunks:
        cat = c.get("category", "UNKNOWN")
        crop = c.get("crop_name", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        crops[crop] = crops.get(crop, 0) + 1

    print("\n--- By Category ---")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print("\n--- By Crop ---")
    for crop, count in sorted(crops.items()):
        print(f"  {crop}: {count}")


if __name__ == "__main__":
    # Test run
    chunks = process_all_documents()
    print_chunk_summary(chunks)
    
    # Save chunked output
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "_chunked_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(chunk_metadata_to_json(chunks))
    print(f"\nChunked output saved to: {output_path}")
