"""Auto metadata enrichment on ingestion — extract entities, topics, summaries."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class MetadataEnrichmentService:
    """Automatically enriches ingested content with extracted metadata."""

    def __init__(self):
        self._llm_available = False
        self._check_llm()

    def _check_llm(self):
        try:
            from services.rag_service import openai_chat_completion
            self._llm_available = True
        except Exception:
            logger.info("LLM not available — metadata enrichment will use regex only")

    def enrich(self, text: str, existing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract entities, topics, and summary from text."""
        enriched = dict(existing_metadata or {})

        enriched["_word_count"] = len(text.split())
        enriched["_char_count"] = len(text)

        emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
        if emails:
            enriched["_emails"] = emails

        urls = re.findall(r'https?://[^\s]+', text)
        if urls:
            enriched["_urls"] = urls

        if self._llm_available and len(text) > 20:
            try:
                from services.rag_service import openai_chat_completion
                prompt = (
                    "Extract from the following text: entities (people, orgs, places), "
                    "topics (max 3), and a one-sentence summary. "
                    "Return JSON: {\"entities\": [...], \"topics\": [...], \"summary\": \"...\"}\n\n"
                    f"Text: {text[:2000]}"
                )
                response = openai_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500, temperature=0.1,
                )
                data = json.loads(response)
                enriched["_entities"] = data.get("entities", [])
                enriched["_topics"] = data.get("topics", [])
                enriched["_summary"] = data.get("summary", "")
            except Exception as e:
                logger.debug("LLM enrichment failed: %s", e)

        return enriched

metadata_enricher = MetadataEnrichmentService()
