"""
LLM-based topic labeler for BERTopic clusters.

Uses Gemini (or OpenRouter / HuggingFace fallback) to generate
human-readable topic labels from c-TF-IDF keywords, replacing
the raw keyword strings with concise, meaningful names.

Labels are cached to disk to avoid redundant API calls.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from google import genai

logger = logging.getLogger(__name__)

LABEL_PROMPT = """You are an expert customer support analyst. Given the following keywords extracted from a cluster of customer complaints, generate a concise topic label (3-5 words maximum) that summarizes the core issue.

Keywords: {keywords}

Rules:
- Be specific and descriptive (e.g., "Credit Card Billing Disputes" not "Card Issues")
- Use title case
- Return ONLY the label, nothing else

Topic label:"""


class TopicLabeler:
    """Generate human-readable topic labels from cluster keywords using an LLM."""

    def __init__(self, provider="gemini", api_key=None, cache_path=None):
        """
        Args:
            provider: LLM provider ("gemini", "openrouter", or "huggingface").
            api_key: API key (or reads from environment).
            cache_path: Path to cache file for labels. If None, caching is disabled.
        """
        self.provider = provider.lower()
        self.cache_path = cache_path
        self._cache = self._load_cache()

        if self.provider == "gemini":
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.5-flash"
            print("TopicLabeler: initialized with Gemini")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _load_cache(self) -> Dict[str, str]:
        """Load cached labels from disk."""
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                cache = json.load(f)
            print(f"TopicLabeler: loaded {len(cache)} cached labels")
            return cache
        return {}

    def _save_cache(self):
        """Save cached labels to disk."""
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)

    def label_topic(self, keywords: List[str]) -> str:
        """
        Generate a label for a single topic from its keywords.

        Args:
            keywords: List of keywords (or list of (word, score) tuples).

        Returns:
            Human-readable topic label string.
        """
        # Handle (word, score) tuples from BERTopic
        if keywords and isinstance(keywords[0], (list, tuple)):
            keyword_strs = [w for w, _ in keywords]
        else:
            keyword_strs = keywords

        cache_key = "|".join(sorted(keyword_strs[:10]))

        if cache_key in self._cache:
            return self._cache[cache_key]

        keyword_text = ", ".join(keyword_strs[:10])
        prompt = LABEL_PROMPT.format(keywords=keyword_text)

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            label = response.text.strip().strip('"').strip("'")
            # Sanity check: label should be short
            if len(label) > 60:
                label = label[:60].rsplit(" ", 1)[0]
        except Exception as e:
            logger.error("TopicLabeler: LLM call failed: %s", e)
            label = keyword_text[:40]

        self._cache[cache_key] = label
        self._save_cache()

        return label

    def label_all_topics(self, topic_keywords: Dict[int, list],
                         delay: float = 0.5) -> Dict[int, str]:
        """
        Generate labels for all topics.

        Args:
            topic_keywords: Dict mapping topic_id to list of keywords
                            (from BERTopicClusterer.get_topic_keywords()).
            delay: Seconds to wait between API calls (rate limiting).

        Returns:
            Dict mapping topic_id to human-readable label string.
        """
        labels = {}
        total = len(topic_keywords)

        print(f"TopicLabeler: generating labels for {total} topics...")

        for i, (topic_id, keywords) in enumerate(topic_keywords.items()):
            if topic_id == -1:
                labels[topic_id] = "Outlier / Noise"
                continue

            label = self.label_topic(keywords)
            labels[topic_id] = label
            print(f"  Topic {topic_id}: {label}")

            if i < total - 1:
                time.sleep(delay)

        print(f"TopicLabeler: generated {len(labels)} labels")
        return labels

    def save_labels(self, labels: Dict[int, str], output_path: str):
        """Save topic labels to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Convert int keys to strings for JSON
        serializable = {str(k): v for k, v in labels.items()}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Topic labels saved to {output_path}")

    @staticmethod
    def load_labels(input_path: str) -> Dict[int, str]:
        """Load topic labels from JSON."""
        with open(input_path, "r") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
