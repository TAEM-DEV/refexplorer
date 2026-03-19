#!/usr/bin/env python3
"""
refexplorer — Co-citation graph explorer for TAEM domain_knowledge.

Runs as a k3s Job (C-009-009). Queries Semantic Scholar, OpenAlex, CrossRef
for co-citation graph data, scores candidates by relevance, embeds with
nomic-embed-text via cluster Ollama (C-009-008), and writes vectors to the
domain_knowledge Qdrant collection.

Environment variables:
  TASK_DESCRIPTION  — The integration task to explore (required)
  DOMAIN_TAGS       — Comma-separated domain tags (required)
  MISSION_ID        — Optional mission ID that triggered this run
  OLLAMA_URL        — Ollama HTTP endpoint (default: http://ollama.fabric-sdk:11434)
  QDRANT_URL        — Qdrant HTTP endpoint (default: http://qdrant.fabric-sdk:6333)
  REDIS_URL         — Redis URL (default: redis://redis-master.fabric-sdk:6379)
  MAX_PAPERS        — Max papers to index per run (default: 20)
  SS_RATE_LIMIT     — Minimum seconds between Semantic Scholar calls (default: 1.1)
"""

import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [refexplorer] %(levelname)s %(message)s",
)
log = logging.getLogger("refexplorer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_DESCRIPTION = os.environ.get("TASK_DESCRIPTION", "")
DOMAIN_TAGS = [t.strip() for t in os.environ.get("DOMAIN_TAGS", "").split(",") if t.strip()]
MISSION_ID = os.environ.get("MISSION_ID", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.fabric-sdk:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant.fabric-sdk:6333")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis-master.fabric-sdk:6379")
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "20"))
SS_RATE_LIMIT = float(os.environ.get("SS_RATE_LIMIT", "1.1"))  # C-009-004

COLLECTION = "domain_knowledge"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

# Track last Semantic Scholar call for rate limiting (C-009-004)
_last_ss_call: float = 0.0


def _rate_limit_ss() -> None:
    """Enforce minimum 1.1s between Semantic Scholar API calls (C-009-004)."""
    global _last_ss_call
    elapsed = time.monotonic() - _last_ss_call
    if elapsed < SS_RATE_LIMIT:
        sleep_time = SS_RATE_LIMIT - elapsed
        log.debug("rate limit: sleeping %.2fs", sleep_time)
        time.sleep(sleep_time)
    _last_ss_call = time.monotonic()


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------

SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_FIELDS = "title,abstract,citationCount,year,externalIds"


def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar for papers matching query."""
    _rate_limit_ss()
    try:
        resp = httpx.get(
            f"{SS_BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": SS_FIELDS},
            timeout=30,
        )
        if resp.status_code == 429:
            log.warning("Semantic Scholar rate limited, waiting 5s")
            time.sleep(5)
            return search_semantic_scholar(query, limit)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        log.error("Semantic Scholar search failed: %s", e)
        return []


def get_citations(paper_id: str, limit: int = 10) -> list[dict]:
    """Get co-citations for a paper from Semantic Scholar."""
    _rate_limit_ss()
    try:
        resp = httpx.get(
            f"{SS_BASE}/paper/{paper_id}/citations",
            params={"limit": limit, "fields": SS_FIELDS},
            timeout=30,
        )
        if resp.status_code == 429:
            log.warning("Semantic Scholar rate limited on citations, waiting 5s")
            time.sleep(5)
            return get_citations(paper_id, limit)
        resp.raise_for_status()
        data = resp.json()
        return [c.get("citingPaper", {}) for c in data.get("data", []) if c.get("citingPaper")]
    except Exception as e:
        log.error("Semantic Scholar citations failed for %s: %s", paper_id, e)
        return []


# ---------------------------------------------------------------------------
# OpenAlex API (fallback)
# ---------------------------------------------------------------------------

OA_BASE = "https://api.openalex.org"


def search_openalex(query: str, limit: int = 10) -> list[dict]:
    """Fallback: search OpenAlex for papers."""
    try:
        resp = httpx.get(
            f"{OA_BASE}/works",
            params={
                "search": query,
                "per_page": limit,
                "select": "id,title,doi,publication_year,cited_by_count",
            },
            headers={"User-Agent": "refexplorer/0.1 (TAEM-DEV)"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        papers = []
        for r in results:
            title = r.get("title", "")
            papers.append({
                "title": title,
                "abstract": None,  # OpenAlex doesn't return abstracts in search
                "citationCount": r.get("cited_by_count", 0),
                "year": r.get("publication_year"),
                "source": "openalex",
                "externalIds": {"DOI": r.get("doi", "").replace("https://doi.org/", "")} if r.get("doi") else {},
            })
        return papers
    except Exception as e:
        log.error("OpenAlex search failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# CrossRef API (last-resort abstract fallback)
# ---------------------------------------------------------------------------

CR_BASE = "https://api.crossref.org/works"


def fetch_abstract_crossref(doi: str) -> str | None:
    """Last-resort: fetch abstract from CrossRef by DOI."""
    if not doi:
        return None
    try:
        resp = httpx.get(
            f"{CR_BASE}/{doi}",
            headers={"User-Agent": "refexplorer/0.1 (TAEM-DEV; mailto:noreply@taem.dev)"},
            timeout=20,
        )
        resp.raise_for_status()
        message = resp.json().get("message", {})
        abstract = message.get("abstract", "")
        if abstract:
            # CrossRef abstracts sometimes have JATS XML tags
            import re
            return re.sub(r"<[^>]+>", "", abstract).strip()
        return None
    except Exception as e:
        log.debug("CrossRef abstract fetch failed for %s: %s", doi, e)
        return None


# ---------------------------------------------------------------------------
# Ollama embedding (C-009-008)
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float] | None:
    """Embed text using nomic-embed-text via cluster Ollama (C-009-008)."""
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding", [])
        if len(embedding) != EMBED_DIM:
            log.warning("Unexpected embedding dimension: %d (expected %d)", len(embedding), EMBED_DIM)
            return None
        return embedding
    except Exception as e:
        log.error("Ollama embedding failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Qdrant write
# ---------------------------------------------------------------------------


def write_to_qdrant(entry_id: str, vector: list[float], payload: dict[str, Any]) -> bool:
    """Write a point to the domain_knowledge Qdrant collection."""
    body = {
        "points": [
            {
                "id": entry_id,
                "vector": vector,
                "payload": payload,
            }
        ]
    }
    try:
        resp = httpx.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.error("Qdrant write failed for %s: %s", entry_id, e)
        return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_candidate(paper: dict, task: str, domain_tags: list[str]) -> float:
    """Score a paper candidate by relevance to the task and domain."""
    score = 0.0

    # Citation count (logarithmic scaling, max 0.3)
    citations = paper.get("citationCount", 0) or 0
    if citations > 0:
        import math
        score += min(0.3, math.log10(citations + 1) / 10)

    # Title relevance (simple keyword overlap, max 0.4)
    title = (paper.get("title") or "").lower()
    task_words = set(task.lower().split())
    title_words = set(title.split())
    overlap = len(task_words & title_words)
    if task_words:
        score += min(0.4, (overlap / len(task_words)) * 0.4)

    # Abstract presence bonus (0.1)
    if paper.get("abstract"):
        score += 0.1

    # Recency bonus (max 0.2)
    year = paper.get("year")
    if year:
        current_year = datetime.now().year
        age = current_year - year
        if age <= 2:
            score += 0.2
        elif age <= 5:
            score += 0.1
        elif age <= 10:
            score += 0.05

    return round(min(1.0, score), 3)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_embed_text(paper: dict) -> str:
    """Build text for embedding from paper fields."""
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"])
    return ". ".join(parts) if parts else ""


def run_pipeline(task: str, domain_tags: list[str], mission_id: str = "") -> dict:
    """Run the full refexplorer pipeline."""
    log.info("Starting refexplorer pipeline")
    log.info("Task: %s", task)
    log.info("Domain tags: %s", domain_tags)

    stats = {"searched": 0, "scored": 0, "embedded": 0, "indexed": 0, "errors": 0}
    now = datetime.now(timezone.utc).isoformat()

    # Step 1: Search Semantic Scholar
    log.info("Searching Semantic Scholar...")
    papers = search_semantic_scholar(task, limit=MAX_PAPERS)
    stats["searched"] += len(papers)

    # Step 1b: Get co-citations for top 3 results
    if papers:
        top_papers = papers[:3]
        for p in top_papers:
            pid = p.get("paperId")
            if pid:
                citations = get_citations(pid, limit=5)
                papers.extend(citations)
                stats["searched"] += len(citations)

    # Step 2: Fallback to OpenAlex if Semantic Scholar returned nothing
    if not papers:
        log.info("Semantic Scholar returned nothing, trying OpenAlex...")
        papers = search_openalex(task, limit=MAX_PAPERS)
        stats["searched"] += len(papers)

    if not papers:
        log.warning("No papers found from any source")
        return stats

    # Step 3: Deduplicate by title
    seen_titles: set[str] = set()
    unique_papers: list[dict] = []
    for p in papers:
        title = (p.get("title") or "").strip().lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(p)
    papers = unique_papers[:MAX_PAPERS]
    log.info("After dedup: %d papers", len(papers))

    # Step 4: Score, enrich abstracts, embed, write
    for paper in papers:
        # Determine source
        source = paper.get("source", "semantic_scholar")

        # Try to enrich abstract from CrossRef if missing
        if not paper.get("abstract"):
            doi = None
            ext_ids = paper.get("externalIds") or {}
            if isinstance(ext_ids, dict):
                doi = ext_ids.get("DOI")
            if doi:
                abstract = fetch_abstract_crossref(doi)
                if abstract:
                    paper["abstract"] = abstract
                    log.info("Enriched abstract from CrossRef for: %s", paper.get("title", "")[:60])

        # Score
        relevance = score_candidate(paper, task, domain_tags)
        stats["scored"] += 1

        # Skip low-relevance papers
        if relevance < 0.1:
            log.debug("Skipping low-relevance paper: %s (%.3f)", paper.get("title", "")[:60], relevance)
            continue

        # Embed
        embed_input = build_embed_text(paper)
        if not embed_input:
            log.debug("Skipping paper with no embeddable text")
            stats["errors"] += 1
            continue

        vector = embed_text(embed_input)
        if vector is None:
            log.error("Failed to embed: %s", paper.get("title", "")[:60])
            stats["errors"] += 1
            continue
        stats["embedded"] += 1

        # Build payload per ADR-009a schema (C-009-001)
        entry_id = str(uuid.uuid4())
        payload = {
            "entry_id": entry_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "domain_tags": domain_tags,
            "score": relevance,
            "source": source,
            "advisory_source": "literature",
            "operator_annotations": [],
            "mission_refs": [mission_id] if mission_id else [],
            "created_at": now,
            "updated_at": now,
        }

        # Write to Qdrant
        if write_to_qdrant(entry_id, vector, payload):
            stats["indexed"] += 1
            log.info("Indexed: %s (score=%.3f, source=%s)", paper.get("title", "")[:60], relevance, source)
        else:
            stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# Redis cache updates
# ---------------------------------------------------------------------------


def update_redis_cache(task: str, domain_tags: list[str], stats: dict) -> None:
    """Update Redis cache keys per ADR-009a schema."""
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)

        # domain:classification:{task_hash} — TTL 7 days (C-009-003)
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:16]
        r.setex(
            f"domain:classification:{task_hash}",
            604800,  # 7 days
            json.dumps({"task": task, "domain_tags": domain_tags, "stats": stats}),
        )

        # domain:warm:{domain_tag} — TTL 24h
        for tag in domain_tags:
            r.setex(f"domain:warm:{tag}", 86400, "1")

        log.info("Redis cache updated: classification=%s, warm tags=%s", task_hash, domain_tags)
    except Exception as e:
        log.error("Redis cache update failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not TASK_DESCRIPTION:
        log.error("TASK_DESCRIPTION env var is required")
        return 1
    if not DOMAIN_TAGS:
        log.error("DOMAIN_TAGS env var is required")
        return 1

    # Check Ollama connectivity
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(EMBED_MODEL in m for m in models):
            log.error("Model %s not found on Ollama. Available: %s", EMBED_MODEL, models)
            return 1
        log.info("Ollama OK, model %s available", EMBED_MODEL)
    except Exception as e:
        log.error("Ollama not reachable at %s: %s", OLLAMA_URL, e)
        return 1

    # Check Qdrant connectivity
    try:
        resp = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
        resp.raise_for_status()
        log.info("Qdrant collection %s OK", COLLECTION)
    except Exception as e:
        log.error("Qdrant collection %s not reachable: %s", COLLECTION, e)
        return 1

    # Run pipeline
    stats = run_pipeline(TASK_DESCRIPTION, DOMAIN_TAGS, MISSION_ID)
    log.info("Pipeline complete: %s", json.dumps(stats))

    # Update Redis cache
    update_redis_cache(TASK_DESCRIPTION, DOMAIN_TAGS, stats)

    if stats["indexed"] == 0 and stats["errors"] > 0:
        log.error("No papers indexed and errors occurred")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
