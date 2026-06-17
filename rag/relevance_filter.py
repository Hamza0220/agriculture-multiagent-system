"""
Agricultural Relevance Filter (M1-3)
Scores and filters ChromaDB-retrieved chunks for relevance to the farmer's query.
Removes off-topic chunks, deduplicates, ranks by usefulness.

Two-tier architecture:
  - LLM path (Groq): Full M1-3 prompt for best quality
  - Rule-based path: Keyword/crop/category overlap scoring (zero-cost, always works)
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple

try:
    from groq import Groq
except ImportError:
    Groq = None


# ── M1-3 System Prompt ─────────────────────────────────────────────────

M1_3_SYSTEM_PROMPT = """You are an agricultural knowledge quality assessor for Pakistani farmers.

Evaluate each retrieved knowledge chunk for relevance to the farmer's query.

SCORING GUIDE:

HIGHLY RELEVANT (score 8-10):
→ Directly describes the same disease/pest/problem the farmer has
→ Specifically mentions this crop
→ Applicable in Pakistan's climate/conditions
→ Contains specific treatment or action steps

RELEVANT (score 5-7):
→ Related condition that might be relevant
→ General crop management that applies
→ Similar disease with overlapping treatments

NOT RELEVANT (score 0-4):
→ Different crop (rice info when farmer has wheat problem)
→ Different region's conditions not applicable to Pakistan
→ General agriculture theory without practical steps
→ Same information already covered by a higher-scored chunk

OUTPUT ONLY VALID JSON (no markdown, no code fences):
{
  "filtered_chunks": [
    {
      "chunk_id": "...",
      "relevance_score": 8,
      "relevance_reason": "Directly describes the problem with specific treatment",
      "actionable_sentence": "Spray Propiconazole 25EC at 0.5ml/liter water as soon as symptoms appear",
      "has_specific_dosage": true,
      "timing_relevant": true,
      "safety_warning": "Use gloves and mask when spraying",
      "local_product_name": "Tilt 250EC (locally available brand)"
    }
  ],
  "excluded_chunk_ids": ["id_001"],
  "total_included": 3,
  "most_critical_chunk": "id_001",
  "information_gaps": ["No organic treatment alternatives found"]
}"""


def build_m1_3_prompt(
    user_query: str,
    crop_name: str,
    location: str,
    season: str,
    problem_description: str,
    retrieved_chunks: List[Dict],
) -> str:
    """Build the M1-3 prompt with all context."""
    chunks_json = json.dumps([
        {
            "chunk_id": c.get("chunk_id", "unknown"),
            "crop_name": c.get("metadata", {}).get("crop_name", ""),
            "category": c.get("metadata", {}).get("category", ""),
            "content_preview": c.get("content", "")[:300],
        }
        for c in retrieved_chunks
    ], indent=2, ensure_ascii=False)

    return f"""Farmer's Query: {user_query}
Crop in Question: {crop_name}
Farmer's Location: {location}
Current Season: {season}
Problem Description: {problem_description}

Retrieved knowledge chunks from database:
{chunks_json}"""


def parse_m1_3_response(text: str) -> Dict:
    """Parse LLM response JSON, handling markdown wrapping."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"error": "Could not parse LLM response", "raw": text[:300]}


# ── Rule-based relevance scorer ───────────────────────────────────────

def _score_chunk_rule_based(
    chunk: Dict,
    user_query: str,
    crop_name: Optional[str],
    location: Optional[str],
    season: Optional[str],
) -> Tuple[int, str, bool, bool, str, str]:
    """
    Score a single chunk using rule-based heuristics.

    Returns:
        (score, reason, has_dosage, timing_relevant, safety_warning, local_product)
    """
    content = (chunk.get("content") or "").lower()
    metadata = chunk.get("metadata") or {}
    chunk_crop = (metadata.get("crop_name") or "").lower()
    chunk_cat = (metadata.get("category") or "").upper()
    query_lower = user_query.lower()

    score = 5  # start neutral
    reason_parts = []
    has_dosage = False
    timing_relevant = True
    safety_warning = ""
    local_product = ""

    # ── Crop match (highest weight) ──
    if crop_name and crop_name.lower() == chunk_crop:
        score += 3
        reason_parts.append(f"crop matches ({chunk_crop})")
    elif crop_name and chunk_crop != "general" and crop_name.lower() != chunk_crop:
        score -= 3
        reason_parts.append(f"different crop ({chunk_crop} vs {crop_name})")
    elif chunk_crop == "general":
        score += 0  # general info, not penalized
    else:
        score += 1  # general applicability

    # ── Query keyword overlap ──
    query_keywords = set(query_lower.split())
    content_keywords = set(content.split())
    overlap = query_keywords & content_keywords
    overlap_score = len(overlap) / max(len(query_keywords), 1)
    if overlap_score > 0.3:
        score += 2
        reason_parts.append(f"high keyword overlap ({len(overlap)} matches)")
    elif overlap_score > 0.15:
        score += 1
        reason_parts.append("moderate keyword overlap")
    else:
        score -= 1

    # ── Specific numbers / dosages ──
    dosage_patterns = [
        r"\d+\s*ml\s+per", r"\d+\s*g\s+per", r"\d+\s*kg\s+per",
        r"\d+\s*liter", r"\d+\s*acre", r"Rs\.\s*\d+",
        r"\d+%\s*(EC|WP|SL|SC|WDG)",
    ]
    for pattern in dosage_patterns:
        if re.search(pattern, content):
            has_dosage = True
            score += 2
            reason_parts.append("has specific dosage")
            break

    # ── Product names (chemicals) ──
    chem_pattern = r"(Tilt|Confidor|Coragen|Dithane|Ridomil|Nativo|Amistar|Folicur|Carbendazim|Mancozeb|Imidacloprid|Chlorantraniliprole|Propiconazole|Tebuconazole|Spinosad|Lambda|Buprofezin|Pyriproxyfen|Tricyclazole|Carbofuran)"
    chem_match = re.search(chem_pattern, content, re.IGNORECASE)
    if chem_match:
        local_product = chem_match.group(1)
        # Find brand name if mentioned
        brand_patterns = [
            r"Local brand[s]?:\s*([^.\n]+)",
            r"local(?:ly\s+)?(?:available)?\s+br(?:and)?\s*:?\s*([^.\n]+)",
        ]
        for bp in brand_patterns:
            bm = re.search(bp, content, re.IGNORECASE)
            if bm:
                local_product = bm.group(1).strip()[:60]
                break
        reason_parts.append(f"has product name ({local_product})")
        score += 1

    # ── Safety warnings ──
    safety_patterns = [
        r"safety[- ]?warn", r"wear\s+(gloves|mask|PPE)", r"safety\s+precaution",
        r"keep\s+(children|animals)", r"do\s+not\s+(spray|apply|eat)",
        r"pre[- ]?harvest interval",
    ]
    for sp in safety_patterns:
        if re.search(sp, content, re.IGNORECASE):
            safety_warning = content[:200]  # extract around safety mention
            # Try to get the actual safety sentence
            lines = content.split("\n")
            for line in lines:
                if re.search(sp, line, re.IGNORECASE):
                    safety_warning = line.strip()[:150]
                    break
            break

    # ── Timing/season match ──
    if season:
        season_lower = season.lower()
        if season_lower in content:
            timing_relevant = True
            score += 1
        else:
            timing_relevant = False  # not penalized, just not boosted

    # ── Location match ──
    if location:
        loc_lower = location.lower()
        # Extract first word (city name)
        city = loc_lower.split()[0] if loc_lower else ""
        if city and city in content:
            score += 1
            reason_parts.append(f"location-specific ({city})")

    # ── Penalize if chunk category doesn't match query intent ──
    query_categories = set()
    for kw, cat in [
        ("disease", "DISEASE"), ("bimari", "DISEASE"), ("zang", "DISEASE"),
        ("pest", "PEST"), ("keera", "PEST"), ("makhi", "PEST"),
        ("irrigation", "IRRIGATION"), ("paani", "IRRIGATION"),
        ("fertilizer", "FERTILIZER"), ("khad", "FERTILIZER"),
        ("market", "MARKET"), ("bhav", "MARKET"), ("mandi", "MARKET"),
        ("harvest", "HARVESTING"), ("katai", "HARVESTING"),
    ]:
        if kw in query_lower:
            query_categories.add(cat)

    if query_categories and chunk_cat in query_categories:
        score += 1

    # Clamp score to 0-10
    score = max(0, min(10, score))

    reason = "; ".join(reason_parts) if reason_parts else "no strong signals"
    return score, reason, has_dosage, timing_relevant, safety_warning, local_product


# ── Public API ─────────────────────────────────────────────────────────

def filter_chunks(
    chunks: List[Dict],
    user_query: str,
    crop_name: Optional[str] = None,
    location: Optional[str] = None,
    season: Optional[str] = None,
    min_score: int = 5,
    use_llm: bool = True,
) -> Dict:
    """
    Filter and score retrieved chunks by relevance.

    Args:
        chunks: List of chunk dicts from vector store search
        user_query: Original farmer query
        crop_name: Crop being asked about
        location: Farmer's location
        season: Current season (Rabi/Kharif)
        min_score: Minimum relevance score to include (0-10)
        use_llm: If True, try Groq scoring; always falls back to rules

    Returns:
        Dict with: filtered_chunks, excluded_chunk_ids, total_included,
                   most_critical_chunk, information_gaps
    """
    if not chunks:
        return {
            "filtered_chunks": [],
            "excluded_chunk_ids": [],
            "total_included": 0,
            "most_critical_chunk": None,
            "information_gaps": ["No chunks retrieved to evaluate"],
        }

    # Try LLM path first
    if use_llm:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and Groq is not None:
            try:
                client = Groq(api_key=groq_api_key)
                model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

                prompt = build_m1_3_prompt(
                    user_query=user_query,
                    crop_name=crop_name or "unknown",
                    location=location or "Pakistan",
                    season=season or "current",
                    problem_description=user_query,
                    retrieved_chunks=chunks,
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": M1_3_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                )

                result = parse_m1_3_response(response.choices[0].message.content)
                if "filtered_chunks" in result:
                    # Map back to original chunk content
                    id_map = {c.get("chunk_id", ""): c for c in chunks}
                    for fc in result.get("filtered_chunks", []):
                        original = id_map.get(fc.get("chunk_id", ""), {})
                        fc["content"] = original.get("content", "")
                        fc["metadata"] = original.get("metadata", {})

                    result.setdefault("excluded_chunk_ids", [])
                    result.setdefault("information_gaps", [])
                    return result

            except Exception:
                pass  # Fall through to rule-based

    # Rule-based scoring
    scored = []
    for chunk in chunks:
        score, reason, has_dosage, timing_relevant, safety_warning, local_product = \
            _score_chunk_rule_based(chunk, user_query, crop_name, location, season)

        scored.append({
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "content": chunk.get("content", ""),
            "metadata": chunk.get("metadata", {}),
            "relevance_score": score,
            "relevance_reason": reason,
            "has_specific_dosage": has_dosage,
            "timing_relevant": timing_relevant,
            "safety_warning": safety_warning,
            "local_product_name": local_product,
            "actionable_sentence": _extract_actionable(chunk.get("content", "")),
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Apply dedup: if two chunks have same actionable sentence, keep higher-scored
    seen_sentences = set()
    deduped = []
    for s in scored:
        sent = s.get("actionable_sentence", "")[:80]
        if sent and sent in seen_sentences:
            continue
        if sent:
            seen_sentences.add(sent)
        deduped.append(s)

    # Split into included / excluded
    included = [s for s in deduped if s["relevance_score"] >= min_score]
    excluded_ids = [s["chunk_id"] for s in deduped if s["relevance_score"] < min_score]

    # Detect information gaps
    information_gaps = []
    has_chemical = any(s.get("has_specific_dosage") for s in included)
    has_organic = any(
        "organic" in (s.get("content") or "").lower()
        or "neem" in (s.get("content") or "").lower()
        for s in included
    )
    if not has_chemical:
        information_gaps.append("No chemical treatment with specific dosage found")
    if not has_organic:
        information_gaps.append("No organic/natural treatment alternative found")

    return {
        "filtered_chunks": included,
        "excluded_chunk_ids": excluded_ids,
        "total_included": len(included),
        "most_critical_chunk": included[0]["chunk_id"] if included else None,
        "information_gaps": information_gaps,
    }


def _extract_actionable(content: str, max_len: int = 200) -> str:
    """Extract the single most actionable sentence from chunk content."""
    if not content:
        return ""

    # Priority patterns for actionable sentences
    patterns = [
        r"(Spray|Apply|Use|Mix|Treat|Irrigate|Sow|Plant)\s[^.\n]+(?:ml|g|kg|liter|acre|Rs\.)[^.\n]+",
        r"(Spray|Apply|Use|Mix|Treat)\s[^.\n]{10,100}\.",
        r"(Recommended|Best|Most effective)\s[^.\n]{10,100}\.",
        r"(Option\s+A|Option\s+1|Treatment)\s*:?\s*[^.\n]{10,100}",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            sent = match.group().strip()
            if not sent.endswith("."):
                sent += "."
            return sent[:max_len]

    # Fallback: first substantive line
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    for line in lines:
        if len(line) > 30 and not line.startswith("==") and not line.startswith("Crop:"):
            return line[:max_len]

    return content[:max_len]


def filter_and_format_context(
    chunks: List[Dict],
    user_query: str,
    crop_name: Optional[str] = None,
    location: Optional[str] = None,
    season: Optional[str] = None,
    min_score: int = 5,
    use_llm: bool = True,
) -> str:
    """
    Convenience: filter chunks and return a compressed context string.
    Replaces the old compression logic in retriever.py.

    Args:
        Same as filter_chunks()

    Returns:
        Compressed context string with only relevant info
    """
    result = filter_chunks(
        chunks=chunks,
        user_query=user_query,
        crop_name=crop_name,
        location=location,
        season=season,
        min_score=min_score,
        use_llm=use_llm,
    )

    included = result.get("filtered_chunks", [])
    if not included:
        return "No relevant agricultural knowledge found for this query."

    sections = []
    seen_content = set()

    for c in included:
        content = c.get("content", "").strip()
        metadata = c.get("metadata", {})

        # Skip near-duplicates
        preview = content[:80]
        if preview in seen_content:
            continue
        seen_content.add(preview)

        # Build section header
        parts = []
        crop_m = metadata.get("crop_name", "")
        if crop_m and crop_m != "general":
            urdu = metadata.get("crop_urdu", "")
            parts.append(f"{crop_m.title()} ({urdu})" if urdu else crop_m.title())
        cat = metadata.get("category", "")
        if cat and cat != "GENERAL":
            parts.append(cat.title())
        disease = metadata.get("disease_name", "")
        if disease:
            du = metadata.get("disease_urdu", "")
            parts.append(f"{disease} ({du})" if du else disease)
        header = " | ".join(parts) if parts else "Knowledge"

        # Build section
        section_parts = [f"---{header}---"]
        section_parts.append(f"Relevance: {c['relevance_score']}/10 — {c['relevance_reason']}")

        # Add actionable sentence prominently
        action = c.get("actionable_sentence", "")
        if action:
            section_parts.append(f"→ {action}")

        # Add safety warning if present
        safety = c.get("safety_warning", "")
        if safety:
            section_parts.append(f"⚠️ {safety}")

        # Add local product name if present
        product = c.get("local_product_name", "")
        if product:
            section_parts.append(f"Local brand: {product}")

        # Add the full content
        section_parts.append(content)
        sections.append("\n".join(section_parts))

    result_text = "\n\n".join(sections)

    # Add info gaps footer
    gaps = result.get("information_gaps", [])
    if gaps:
        result_text += f"\n\n---Information Gaps---\n" + "\n".join(f"• {g}" for g in gaps)

    return result_text


# ── CLI entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("M1-3 Relevance Filter module loaded.")
    print("\nQuick test with mock data...")

    # Test rule-based scorer
    test_chunks = [
        {
            "chunk_id": "wheat_disease_001",
            "content": "Yellow Rust (Puccinia striiformis) on wheat. "
                       "Spray Propiconazole 25EC (Tilt 250EC) at 200ml per acre in 100L water. "
                       "Cost: Rs. 800-1200 per acre. Apply at first symptom appearance. "
                       "Safety: Wear gloves and mask when spraying.",
            "metadata": {
                "crop_name": "wheat",
                "crop_urdu": "gandum",
                "category": "DISEASE",
                "season": "Rabi",
                "pakistan_region": "Punjab",
            },
        },
        {
            "chunk_id": "rice_blast_001",
            "content": "Rice Blast disease symptoms and management. "
                       "Diamond-shaped lesions on leaves. Avoid excessive nitrogen.",
            "metadata": {
                "crop_name": "rice",
                "crop_urdu": "chawal",
                "category": "DISEASE",
                "season": "Kharif",
                "pakistan_region": "All",
            },
        },
    ]

    result = filter_chunks(
        chunks=test_chunks,
        user_query="wheat yellow rust treatment",
        crop_name="wheat",
        location="Faisalabad",
        season="Rabi",
        use_llm=False,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n--- filter_and_format_context ---")
    context = filter_and_format_context(
        chunks=test_chunks,
        user_query="wheat yellow rust treatment",
        crop_name="wheat",
        use_llm=False,
    )
    print(context[:500])
