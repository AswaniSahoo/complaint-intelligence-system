"""Base retriever interface. All retrievers inherit from BaseRetriever."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time


@dataclass
class RetrievalResult:
    """Standardized result from any retrieval strategy."""
    text: str
    product: str = "Unknown"
    issue: str = "Unknown"
    date: str = ""
    score: float = 0.0
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "product": self.product,
            "issue": self.issue,
            "date": self.date,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""

    name: str = "base"

    @abstractmethod
    def build_index(self, df, embeddings=None, **kwargs):
        """Build the retrieval index from complaint data."""
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k relevant complaints for a query."""
        pass

    def retrieve_with_timing(self, query: str, k: int = 5):
        """Retrieve results and return (results, latency_ms)."""
        start = time.perf_counter()
        results = self.retrieve(query, k)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms

    def _build_result(self, row, score: float, rank: int,
                      extra_metadata=None):
        """Build a RetrievalResult from a DataFrame row."""
        metadata = {}
        if "cluster" in row.index:
            metadata["cluster"] = int(row["cluster"]) if not isinstance(row["cluster"], float) else None
        if "llm_summary" in row.index:
            metadata["llm_summary"] = row.get("llm_summary")
            metadata["llm_category"] = row.get("llm_category")
            metadata["llm_urgency"] = row.get("llm_urgency")
        if extra_metadata:
            metadata.update(extra_metadata)

        return RetrievalResult(
            text=str(row.get("complaint_text", row.get("clean_text", ""))),
            product=str(row.get("product", "Unknown")),
            issue=str(row.get("issue", "Unknown")),
            date=str(row.get("date", "")),
            score=float(score),
            rank=rank,
            metadata=metadata,
        )
