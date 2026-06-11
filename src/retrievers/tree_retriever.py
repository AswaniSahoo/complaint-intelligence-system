"""
Tree retriever: hierarchical reasoning-based retrieval (vectorless).
Builds a Product -> Issue -> Cluster tree and uses LLM to navigate it.
Inspired by VectifyAI's PageIndex approach.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)



BRANCH_SELECTION_PROMPT = """You are an expert complaint analyst. Given a user query and a list of categories, 
select the TOP {top_n} most relevant categories that would contain complaints matching this query.

User Query: {query}

Available categories:
{categories}

Return ONLY a JSON array of the selected category names, most relevant first.
Example: ["Category A", "Category B"]

Your selection:"""

COMPLAINT_RANKING_PROMPT = """You are an expert complaint analyst. Given a user query, rank these complaint 
snippets by relevance. Return the indices (0-based) of the top {top_n} most relevant complaints.

User Query: {query}

Complaint snippets:
{snippets}

Return ONLY a JSON array of indices, most relevant first.
Example: [2, 0, 4]

Your ranking:"""




class TreeNode:
    """A node in the complaint hierarchy tree."""

    def __init__(self, name: str, level: str, parent=None):
        self.name = name
        self.level = level  # "root", "product", "issue", "cluster"
        self.parent = parent
        self.children: Dict[str, "TreeNode"] = {}
        self.complaint_indices: List[int] = []  # DataFrame row indices at leaf

    @property
    def path(self) -> str:
        """Full path from root to this node."""
        parts = []
        node = self
        while node and node.level != "root":
            parts.append(node.name)
            node = node.parent
        return " → ".join(reversed(parts)) or "root"

    @property
    def count(self) -> int:
        """Total complaints in this subtree."""
        if self.complaint_indices:
            return len(self.complaint_indices)
        return sum(child.count for child in self.children.values())

    def summary(self) -> str:
        """One-line summary for display."""
        return f"{self.name} ({self.count} complaints)"




class TreeRetriever(BaseRetriever):
    """PageIndex-inspired hierarchical tree retrieval (vectorless)."""

    name = "tree"

    def __init__(self, llm_summarizer=None, max_leaf_results=20):
        self.llm = llm_summarizer
        self.max_leaf_results = max_leaf_results
        self.root: Optional[TreeNode] = None
        self.df: Optional[pd.DataFrame] = None

    def build_index(self, df, embeddings=None, **kwargs):
        """Build hierarchical tree: Product -> Issue -> Cluster -> Complaints."""
        self.df = df.reset_index(drop=True)
        self.root = TreeNode(name="All Complaints", level="root")

        has_cluster = "cluster" in self.df.columns

        for idx, row in self.df.iterrows():
            product = str(row.get("product", "Unknown"))
            issue = str(row.get("issue", "Unknown"))
            cluster = str(int(row["cluster"])) if has_cluster and pd.notna(row.get("cluster")) else "0"

            if product not in self.root.children:
                self.root.children[product] = TreeNode(product, "product", parent=self.root)
            product_node = self.root.children[product]

            if issue not in product_node.children:
                product_node.children[issue] = TreeNode(issue, "issue", parent=product_node)
            issue_node = product_node.children[issue]

            cluster_name = f"Cluster {cluster}"
            if cluster_name not in issue_node.children:
                issue_node.children[cluster_name] = TreeNode(cluster_name, "cluster", parent=issue_node)
            cluster_node = issue_node.children[cluster_name]

            cluster_node.complaint_indices.append(idx)

        n_products = len(self.root.children)
        n_issues = sum(len(p.children) for p in self.root.children.values())
        logger.info("TreeRetriever: built tree with %d products, %d issues, %d total complaints",
                     n_products, n_issues, len(self.df))

    def _llm_select_branches(self, query: str, options: List[str], top_n: int = 3) -> List[str]:
        """Use LLM to select the most relevant branches."""
        if self.llm is None or not options:
            return options[:top_n]

        categories_str = "\n".join(f"- {opt}" for opt in options)
        prompt = BRANCH_SELECTION_PROMPT.format(
            query=query, categories=categories_str, top_n=top_n
        )

        try:
            if self.llm.provider == "gemini":
                response = self.llm.model.generate_content(prompt)
                raw = response.text.strip()
            else:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200,
                )
                raw = response.choices[0].message.content.strip()

            selected = json.loads(raw)
            if isinstance(selected, list):
                valid = [s for s in selected if s in options]
                return valid[:top_n] if valid else options[:top_n]
        except Exception as e:
            logger.warning("TreeRetriever: LLM branch selection failed: %s", e)

        return options[:top_n]

    def _llm_rank_complaints(self, query: str, snippets: List[str], top_n: int = 5) -> List[int]:
        """Use LLM to rank complaint snippets by relevance."""
        if self.llm is None or not snippets:
            return list(range(min(top_n, len(snippets))))

        snippets_str = "\n".join(f"[{i}] {s[:200]}" for i, s in enumerate(snippets))
        prompt = COMPLAINT_RANKING_PROMPT.format(
            query=query, snippets=snippets_str, top_n=top_n
        )

        try:
            if self.llm.provider == "gemini":
                response = self.llm.model.generate_content(prompt)
                raw = response.text.strip()
            else:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=100,
                )
                raw = response.choices[0].message.content.strip()

            indices = json.loads(raw)
            if isinstance(indices, list):
                valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(snippets)]
                return valid[:top_n] if valid else list(range(min(top_n, len(snippets))))
        except Exception as e:
            logger.warning("TreeRetriever: LLM complaint ranking failed: %s", e)

        return list(range(min(top_n, len(snippets))))

    def retrieve(self, query, k=5):
        """Navigate the tree using LLM reasoning, return top-k complaints."""
        if self.root is None or self.df is None:
            raise RuntimeError("Tree not built. Call build_index() first.")
        if not query or not query.strip():
            logger.warning("TreeRetriever: empty query received")
            return []

        product_options = sorted(
            self.root.children.keys(),
            key=lambda p: self.root.children[p].count,
            reverse=True,
        )
        selected_products = self._llm_select_branches(query, product_options, top_n=3)
        logger.info("TreeRetriever: selected products = %s", selected_products)

        candidate_indices = []
        retrieval_path: List[str] = []

        for product_name in selected_products:
            product_node = self.root.children.get(product_name)
            if not product_node:
                continue

            issue_options = sorted(
                product_node.children.keys(),
                key=lambda i: product_node.children[i].count,
                reverse=True,
            )
            selected_issues = self._llm_select_branches(query, issue_options, top_n=2)

            for issue_name in selected_issues:
                issue_node = product_node.children.get(issue_name)
                if not issue_node:
                    continue

                retrieval_path.append(f"{product_name} → {issue_name}")

                for cluster_node in issue_node.children.values():
                    candidate_indices.extend(cluster_node.complaint_indices)

        if not candidate_indices:
            logger.warning("TreeRetriever: no candidates found for query")
            return []

        candidate_indices = candidate_indices[:self.max_leaf_results]
        snippets = [str(self.df.iloc[idx].get("clean_text", ""))[:200] for idx in candidate_indices]

        ranked_positions = self._llm_rank_complaints(query, snippets, top_n=k)

        results = []
        for rank, pos in enumerate(ranked_positions, start=1):
            if pos >= len(candidate_indices):
                continue
            idx = candidate_indices[pos]
            row = self.df.iloc[idx]
            score = 1.0 / rank
            result = self._build_result(
                row, score=score, rank=rank,
                extra_metadata={"retrieval_path": retrieval_path, "method": "tree_reasoning"},
            )
            results.append(result)

        return results
