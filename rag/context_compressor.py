"""
Agricultural Context Compressor (M1-4)
Compresses filtered knowledge chunks into 800-token actionable brief.
Removes academic text, keeps numbers/products/timing.

Outputs exact M1-4 format with emoji headers for farmer readability.

Two-tier:
  - LLM path (Groq): Full M1-4 prompt for best formatting
  - Rule-based: Template-based compression (always works, zero cost)
"""

import os
import re
from typing import List, Dict, Optional, Tuple

try:
    from groq import Groq
except ImportError:
    Groq = None


# ── M1-4 System Prompt ─────────────────────────────────────────────────

M1_4_SYSTEM_PROMPT = """You are compressing agricultural knowledge for a Pakistani farmer advisory system.
The farmer needs PRACTICAL, ACTIONABLE information — not academic text.

COMPRESSION RULES:
1. Maximum output: 800 tokens
2. KEEP all specific numbers: dosages, temperatures, quantities, days
   Example: Keep "Apply 40kg DAP per acre" — never compress numbers
3. KEEP all product names and their local Pakistani brand equivalents
4. KEEP all timing information: "spray at flowering stage", "apply before rainfall"
5. REMOVE: Historical information, research citations, academic explanations
6. REMOVE: Information about regions not relevant to farmer's location
7. PRIORITIZE: Cheapest and most locally available solutions first
8. ADD: Simple Urdu equivalent for any technical term used

OUTPUT ONLY the compressed format as plain text (no JSON, no markdown code blocks):

---COMPRESSED AGRICULTURAL KNOWLEDGE---

PROBLEM IDENTIFIED:
[Disease/pest/issue name in English and Urdu]
[How to confirm — visible signs to look for]

IMMEDIATE TREATMENT:
Option 1 (Chemical): [Product name] — [Dose] — [How to apply]
  Local brand: [Commonly available Pakistani brand]
  Cost estimate: Approx Rs. [X] per acre
Option 2 (Organic/Cheap): [If available]

TIMING:
[When exactly to apply/act — crop stage, time of day, weather conditions]

PRECAUTIONS:
[Safety warnings, what NOT to do]

PREVENTION NEXT SEASON:
[1-2 key prevention tips]

---END COMPRESSED CONTEXT---"""


def build_m1_4_prompt(
    problem_description: str,
    crop_name: str,
    filtered_chunks: List[Dict],
) -> str:
    """Build the M1-4 prompt with filtered chunk context."""
    chunks_text = ""
    for i, c in enumerate(filtered_chunks, 1):
        content = c.get("content", "")[:600]  # limit per chunk
        score = c.get("relevance_score", "?")
        chunks_text += f"\n--- Chunk {i} (Score: {score}/10) ---\n{content}\n"

    return f"""Farmer's Specific Problem: {problem_description}
Crop: {crop_name}
Retrieved Knowledge Chunks: {chunks_text}"""


# ── Rule-based compressor ─────────────────────────────────────────────

def _find_value(content: str, patterns: List[str]) -> str:
    """Extract first match for any of the patterns from content."""
    for p in patterns:
        m = re.search(p, content, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                result = m.group(1).strip()[:200]
                if result:
                    return result
            except (IndexError, AttributeError):
                pass
            # Try group(0) as fallback
            try:
                result = m.group(0).strip()[:200]
                if result:
                    return result
            except (IndexError, AttributeError):
                pass
    return ""


def _section_between(content: str, start: str, end: str = None) -> str:
    """Extract text between start marker and end marker."""
    idx = content.find(start)
    if idx == -1:
        return ""
    idx += len(start)
    if end:
        end_idx = content.find(end, idx)
        if end_idx != -1:
            return content[idx:end_idx].strip()
    return content[idx:].strip()[:300]


def _query_match_score(content: str, query_keywords: set) -> int:
    """Score how well content matches query keywords (for tiebreaking same-score chunks)."""
    content_lower = content.lower()
    return sum(1 for kw in query_keywords if len(kw) > 2 and kw in content_lower)


def _rule_based_compress(
    filtered_chunks: List[Dict],
    crop_name: str,
    problem_description: str,
) -> str:
    """Compress chunks into M1-4 format without LLM."""
    if not filtered_chunks:
        return "---COMPRESSED AGRICULTURAL KNOWLEDGE---\n\nNo specific information available for this query.\n\n---END COMPRESSED CONTEXT---"

    # Combine all chunk content (sorted by relevance score descending)
    # For same scores, prefer chunks with query keyword match
    query_keywords = set(problem_description.lower().split())
    filtered_chunks = sorted(
        filtered_chunks,
        key=lambda c: (
            c.get("relevance_score", 0) or 0,
            _query_match_score(c.get("content", ""), query_keywords),
        ),
        reverse=True,
    )
    all_text = "\n".join(c.get("content", "") for c in filtered_chunks)

    # Use ONLY the top chunk for problem identification.
    # Score tiebreaker: prefer chunk whose disease name or content best matches the query.
    query_lower = problem_description.lower()
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    
    def content_tiebreak(c):
        """Score tiebreaker: match query keywords in disease name (weight 5x) and content."""
        meta = c.get("metadata", {})
        dn = (meta.get("disease_name", "") + " " + meta.get("pest_name", "")).lower()
        content = (c.get("content", "") or "").lower()
        name_score = sum(5 for kw in query_words if kw in dn)
        content_score = sum(1 for kw in query_words if kw in content)
        return name_score + content_score
    
    filtered_chunks = sorted(
        filtered_chunks,
        key=lambda c: ((c.get("relevance_score", 0) or 0), content_tiebreak(c)),
        reverse=True,
    )
    
    top_chunk = filtered_chunks[0] if filtered_chunks else {}
    top_content = top_chunk.get("content", "")
    top_meta = top_chunk.get("metadata", {})
    problem_category = top_meta.get("category", "")

    # Extract disease/pest name from top chunk metadata
    disease_name = top_meta.get("disease_name", "") or top_meta.get("pest_name", "")
    disease_urdu = top_meta.get("disease_urdu", "") or top_meta.get("pest_urdu", "")
    if disease_name:
        disease_name = f"{disease_name} ({disease_urdu})" if disease_urdu else disease_name
    else:
        disease_name = f"{crop_name.title()} — {problem_category.title()}" if problem_category else f"{crop_name.title()} Issue"

    # Extract confirmation signs from top chunk only
    confirm_signs = ""
    if problem_category in ("DISEASE", "PEST"):
        confirm_signs = _section_between(top_content, "Symptoms:")
        if not confirm_signs:
            symptom_lines = []
            for line in top_content.split("\n"):
                l = line.strip()
                if l.startswith("- ") and any(w in l.lower() for w in [
                    "yellow", "brown", "black", "white", "spots", "stripes",
                    "lesions", "wilt", "rot", "curl", "holes", "powder",
                ]):
                    symptom_lines.append(l.strip("- "))
                    if len(symptom_lines) >= 4:
                        break
            confirm_signs = "\n".join(symptom_lines[-4:]) if symptom_lines else "Check crop for visible symptoms"

    # ── Extract treatment options (category-aware) ──
    treatment_opt1 = ""
    option1_product = ""
    local_brand = ""
    cost_estimate = ""

    if problem_category == "FERTILIZER":
        # Look for NPK recommendations (search all chunks for most complete data)
        search_text = top_content if "Nitrogen" in top_content or "NPK" in top_content else all_text
        npk_lines = []
        in_npk = False
        for line in search_text.split("\n"):
            l = line.strip()
            if any(kw in l.lower() for kw in ["npk", "recommended", "per acre", "nitrogen:", "phosphorus:", "potash:"]):
                if l and not l.startswith("==") and not l.startswith("---"):
                    npk_lines.append(l.lstrip("- •").strip())
                    in_npk = True
            elif in_npk:
                if any(k in l.lower() for k in ["kg/acre", "kg per acre", "urea", "ssp", "mop", "dap"]):
                    npk_lines.append(l.lstrip("- •").strip())
                elif l.startswith("==") or l.startswith("---") or "Application" in l or "Timing" in l:
                    break
        if npk_lines:
            treatment_opt1 = "\n".join(npk_lines[:5])
        else:
            # Fallback: extract lines with kg/acre
            for line in all_text.split("\n"):
                l = line.strip()
                if re.search(r"\d+\s*kg/acre", l) or re.search(r"\d+\s*kg\s+per\s+acre", l):
                    treatment_opt1 = l[:200]
                    break

    elif problem_category == "IRRIGATION":
        for line in all_text.split("\n"):
            l = line.strip()
            if any(p in l.lower() for p in ["cm", "inches", "hours", "irrigate", "water depth"]):
                if "Critical" in l or "CRITICAL" in l or l.startswith("- "):
                    continue
                treatment_opt1 = l[:200]
                break
        if not treatment_opt1:
            for line in all_text.split("\n"):
                l = line.strip()
                if "stage" in l.lower() and any(c.isdigit() for c in l):
                    treatment_opt1 = l[:200]
                    break

    else:
        # DISEASE/PEST — look for chemical treatments
        dosage_patterns = [
            r"(?:Option A|Option 1|1\.)\s*[^:]*:\s*(.+?)(?:\n|$)",
        ]
        for dp in dosage_patterns:
            m = re.search(dp, all_text, re.IGNORECASE)
            if m:
                opt_text = m.group(1)
                break
        else:
            for line in all_text.split("\n"):
                l = line.strip().lower()
                if any(p in l for p in ["ml per acre", "g per acre", "kg per acre"]):
                    option1_product = line[:60]
                    break

        brand_match = re.search(r"Local brand[s]?:\s*([^.\n]+)", all_text)
        if brand_match:
            local_brand = brand_match.group(1).strip()

        cost_match = re.search(r"Cost:\s*(Rs\.?\s*\d[\d,.-]*)", all_text)
        if cost_match:
            cost_estimate = f"Approx {cost_match.group(1)} per acre"

        if option1_product and ("ml" in option1_product or "g" in option1_product):
            treatment_opt1 = option1_product.capitalize()

    # Fallback treatment text
    if not treatment_opt1:
        for line in all_text.split("\n"):
            l = line.strip()
            if any(p in l.lower() for p in ["ml per acre", "g per acre", "200ml", "500g", "5ml", "400ml", "kg/acre"]):
                treatment_opt1 = l[:200]
                break
        if not treatment_opt1:
            chem_pattern = r"([A-Z][a-z]+[a-z]*\s*\d+[A-Z]{0,3}\s*[—–-]\s*\d+[^.\n]+)"
            chem_match = re.search(chem_pattern, all_text)
            treatment_opt1 = chem_match.group(1)[:200] if chem_match else "See details above"

    # Option 2 (Organic): look for organic/neem alternatives
    option2 = ""
    for line in all_text.split("\n"):
        l = line.strip()
        if any(w in l.lower() for w in ["organic", "neem", "option c", "option 3", "bio"]):
            if "not effective" not in l.lower():
                option2 = l[:150]
                break

    # ── Extract timing (from top chunk first) ──
    timing_text = top_content if any(w in top_content.lower() for w in ["timing", "spray", "apply", "morning"]) else all_text
    timing = _find_value(timing_text, [
        r"Timing:\s*(.+?)(?:\n|$)",
        r"Application\s*(?:Guidelines|Timing):\s*(.+?)(?:\n\n|\Z)",
        r"Apply\s*(?:at|when|before|after)\s[^.\n]+",
        r"Spray\s*(?:in|at|before|after)\s[^.\n]+",
        r"(Best|Optimal)\s*(?:time|timing)\s[^.\n]+",
    ])
    if not timing:
        # Try first line with morning/evening/after
        for line in all_text.split("\n"):
            if any(w in line.lower() for w in ["morning", "evening", "booting", "flowering", "tillering"]):
                timing = line.strip()[:150]
                break

    # ── Extract precautions ──
    precautions = _section_between(all_text, "Safety", "Prevention")
    if not precautions:
        precaution_lines = []
        for line in all_text.split("\n"):
            if any(w in line.lower() for w in ["safety", "wear", "gloves", "mask", "do not", "keep children", "warning"]):
                precaution_lines.append(line.strip())
                if len(precaution_lines) >= 3:
                    break
        precautions = "\n".join(precaution_lines) if precaution_lines else "Follow label instructions"

    # ── Extract prevention ──
    prevention = _section_between(all_text, "Prevention")
    if not prevention:
        prev_lines = []
        in_prev_section = False
        for line in all_text.split("\n"):
            l = line.strip()
            if l.lower().startswith("prevention"):
                in_prev_section = True
                continue
            if in_prev_section:
                if l.startswith("==") or l.startswith("---"):
                    break
                if l.startswith("- ") or l.startswith("•") or re.match(r"^\d+\.", l):
                    prev_lines.append(l.lstrip("- •0123456789.").strip())
                elif l and not l.startswith("Crop:") and not l.startswith("Region:"):
                    prev_lines.append(l)
                if len(prev_lines) >= 5:
                    break
        prevention = "\n".join(prev_lines[-5:]) if prev_lines else ""

    # ── Assemble output ──
    lines = []
    lines.append("---COMPRESSED AGRICULTURAL KNOWLEDGE---")
    lines.append("")
    lines.append("PROBLEM IDENTIFIED:")
    lines.append(disease_name)
    lines.append(confirm_signs[:400])
    lines.append("")
    lines.append("IMMEDIATE TREATMENT:")
    lines.append(f"Option 1 (Chemical): {treatment_opt1[:200]}")
    if local_brand:
        lines.append(f"  Local brand: {local_brand}")
    if cost_estimate:
        lines.append(f"  Cost estimate: {cost_estimate}")
    if option2:
        lines.append(f"Option 2 (Organic/Cheap): {option2}")
    lines.append("")
    lines.append("TIMING:")
    lines.append(timing[:200] if timing else "Apply as soon as symptoms are detected")
    lines.append("")
    lines.append("PRECAUTIONS:")
    lines.append(precautions[:300])
    lines.append("")
    lines.append("PREVENTION NEXT SEASON:")
    if prevention:
        lines.append(prevention[:300])
    else:
        lines.append("Practice crop rotation and use resistant varieties")
    lines.append("")
    lines.append("---END COMPRESSED CONTEXT---")

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────

def compress_context(
    filtered_chunks: List[Dict],
    problem_description: str,
    crop_name: str,
    use_llm: bool = True,
) -> str:
    """
    Compress filtered knowledge chunks into M1-4 structured format.

    Args:
        filtered_chunks: List of scored/filtered chunk dicts from relevance_filter
        problem_description: Original farmer query/problem
        crop_name: Crop being discussed
        use_llm: If True, try Groq; falls back to rules

    Returns:
        Compressed text in M1-4 format (~800 tokens)
    """
    # Try LLM path
    if use_llm:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and Groq is not None:
            try:
                client = Groq(api_key=groq_api_key)
                model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

                prompt = build_m1_4_prompt(problem_description, crop_name, filtered_chunks)

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": M1_4_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                )

                result = response.choices[0].message.content.strip()

                # Validate output has expected structure
                if "---COMPRESSED AGRICULTURAL KNOWLEDGE---" in result:
                    return result

            except Exception:
                pass  # Fall through to rule-based

    # Rule-based fallback
    return _rule_based_compress(filtered_chunks, crop_name, problem_description)


def compress_retrieved_context(
    chunks: List[Dict],
    user_query: str,
    crop_name: Optional[str] = None,
    use_llm: bool = False,
) -> str:
    """
    One-step: apply relevance filter then compress, all in one call.
    Used by retrieve_crop_knowledge() as the final formatting step.

    Args:
        chunks: Raw chunks from vector store
        user_query: Original query for context
        crop_name: Crop filter
        use_llm: Whether to try LLM compression

    Returns:
        Formatted compressed context string
    """
    from .relevance_filter import filter_and_format_context

    # Apply relevance filter first
    filtered = filter_and_format_context(
        chunks=chunks,
        user_query=user_query,
        crop_name=crop_name,
        use_llm=False,  # Rule-based filter is fine
    )

    # Check if we got meaningful content
    if "No relevant" in filtered:
        return filtered

    # Now compress using the full chunk data
    # We need scored chunks for compression — re-filter to get full data
    from .relevance_filter import filter_chunks
    scored_result = filter_chunks(
        chunks=chunks,
        user_query=user_query,
        crop_name=crop_name,
        use_llm=False,
    )

    included = scored_result.get("filtered_chunks", [])
    if not included:
        return "No relevant agricultural knowledge found for this query."

    compressed = compress_context(
        filtered_chunks=included,
        problem_description=user_query,
        crop_name=crop_name or "general",
        use_llm=use_llm,
    )

    return compressed


# ── CLI entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("M1-4 Context Compressor loaded.")
    print("\nQuick test with mock data...")

    test_chunks = [
        {
            "chunk_id": "wheat_disease_001",
            "content": "Yellow Rust (Puccinia striiformis) — Zard Zang. "
                       "Symptoms: Yellow to orange-yellow stripes on leaves. "
                       "Spray Propiconazole 25EC (Tilt 250EC) at 200ml per acre in 100L water. "
                       "Apply at first symptom appearance. Second spray after 10-14 days. "
                       "Cost: Rs. 800-1200 per acre. "
                       "Safety: Wear gloves and mask when spraying. Avoid spraying before rain. "
                       "Prevention: Use resistant varieties Punjab-2011, avoid early sowing.",
            "metadata": {
                "crop_name": "wheat", "crop_urdu": "gandum",
                "category": "DISEASE", "season": "Rabi",
                "disease_name": "Yellow Rust", "disease_urdu": "Zard Zang",
            },
            "relevance_score": 10,
            "has_specific_dosage": True,
            "local_product_name": "Tilt 250EC",
            "safety_warning": "Wear gloves and mask when spraying",
            "actionable_sentence": "Spray Propiconazole 25EC (Tilt 250EC) at 200ml per acre",
        }
    ]

    result = compress_context(
        test_chunks,
        problem_description="Meri gandum ki fasal peeli ho rahi hai paton par zard dhariyan",
        crop_name="wheat",
        use_llm=False,
    )
    print(result)
