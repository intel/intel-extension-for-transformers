from haystack.nodes import BaseRanker

from fastrag.utils import fastrag_timing, safe_import

BaseRanker.timing = fastrag_timing

ColBERTRanker = safe_import("fastrag.rankers.colbert", "ColBERTRanker")