"""
Dual-corpus semantic retrieval (Stage 5) and RAG context builder (Stage 6).

Design: two separate in-memory collections, one over CUAD clause spans and
one over contract-relevant excerpts from ILDC judgment text. Queries hit
both, and the results are merged into a structured context block for an LLM
rewrite prompt. Retrieval quality per corpus is evaluated independently
(Recall@k, MRR) against a train-only index so test clauses can't retrieve
themselves.

ChromaDB is the preferred backend; if it's not installed or its C extensions
fail to load (common on Streamlit Cloud with constrained wheels), the module
falls back to an in-memory numpy implementation with the same API. The
fallback is slightly slower but dependency-free.
"""

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CONTRACT_KEYWORDS = [
    "agreement", "contract", "clause", "breach", "termination", "indemnif",
    "liability", "governing law", "jurisdiction", "consideration", "covenant",
    "non-compete", "exclusivity", "confidential",
]
CONTRACT_KEYWORD_PATTERN = re.compile("|".join(CONTRACT_KEYWORDS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# In-memory backend (always available, no C extensions needed)
# ---------------------------------------------------------------------------

class _InMemoryCollection:
    """Minimal drop-in for a ChromaDB Collection."""

    def __init__(self, name):
        self.name = name
        self._embeddings = None
        self._documents = []
        self._metadatas = []

    def add(self, documents, embeddings, metadatas=None):
        arr = np.array(embeddings, dtype=np.float32)
        self._embeddings = arr if self._embeddings is None else np.vstack([self._embeddings, arr])
        self._documents.extend(documents)
        self._metadatas.extend(metadatas or [{} for _ in documents])

    def query(self, query_embeddings, n_results=5):
        q = np.array(query_embeddings, dtype=np.float32)
        sims = cosine_similarity(q, self._embeddings)[0]
        top = np.argsort(sims)[::-1][:n_results]
        return {
            "documents": [[self._documents[i] for i in top]],
            "metadatas": [[self._metadatas[i] for i in top]],
            "distances": [[float(1 - sims[i]) for i in top]],
        }

    def __len__(self):
        return len(self._documents)


def _new_collection(name, chroma_dir=None):
    """Returns a ChromaDB collection when available, otherwise in-memory."""
    if chroma_dir is not None:
        try:
            import chromadb
            import os
            os.makedirs(chroma_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=chroma_dir)
            try:
                client.delete_collection(name)
            except Exception:
                pass
            return client.create_collection(name)
        except Exception:
            pass
    return _InMemoryCollection(name)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

def get_embedder(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------

def build_cuad_collection(df, embedder, batch_size=256, chroma_dir=None):
    """Embeds all rows in `df` (text + label) and adds them to a collection."""
    collection = _new_collection("cuad_clauses", chroma_dir)
    texts = df["text"].tolist()
    metas = [{"label": lbl} for lbl in df["label"].tolist()]
    for start in range(0, len(texts), batch_size):
        emb = embedder.encode(texts[start:start + batch_size])
        collection.add(texts[start:start + batch_size], emb, metas[start:start + batch_size])
    return collection


def extract_contract_relevant_excerpts(ildc_texts, max_excerpt_chars=800, max_excerpts=5000):
    """
    Keyword-filters ILDC judgment text down to paragraphs that mention
    contract-relevant concepts. ILDC is case-judgment text, so most of each
    document is court procedure rather than contract language; this keeps the
    signal-to-noise ratio manageable.
    """
    sentence_split = re.compile(r"(?<=[.!?])\s+")
    excerpts = []
    for doc in ildc_texts:
        chunk = ""
        for sent in sentence_split.split(doc):
            if len(chunk) + len(sent) > max_excerpt_chars:
                if len(chunk) > 40 and CONTRACT_KEYWORD_PATTERN.search(chunk):
                    excerpts.append(chunk.strip())
                chunk = sent
            else:
                chunk = (chunk + " " + sent).lstrip()
        if len(chunk) > 40 and CONTRACT_KEYWORD_PATTERN.search(chunk):
            excerpts.append(chunk.strip())
        if len(excerpts) >= max_excerpts:
            break
    return excerpts[:max_excerpts]


def build_ildc_collection(ildc_texts, embedder, batch_size=256, chroma_dir=None):
    excerpts = extract_contract_relevant_excerpts(ildc_texts)
    collection = _new_collection("ildc_excerpts", chroma_dir)
    for start in range(0, len(excerpts), batch_size):
        emb = embedder.encode(excerpts[start:start + batch_size])
        collection.add(excerpts[start:start + batch_size], emb)
    return collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_dual_corpus(query, embedder, cuad_collection, ildc_collection, k_cuad=3, k_ildc=3):
    qe = embedder.encode([query])
    cuad_res = cuad_collection.query(qe, n_results=k_cuad)
    ildc_res = ildc_collection.query(qe, n_results=k_ildc)
    return {
        "cuad": cuad_res["documents"][0] if cuad_res["documents"] else [],
        "ildc": ildc_res["documents"][0] if ildc_res["documents"] else [],
    }


def build_rag_context_block(retrieved):
    lines = ["Relevant CUAD contract clauses:"]
    for i, doc in enumerate(retrieved["cuad"], 1):
        lines.append(f"{i}. {doc}")
    lines.append("\nRelevant Indian legal excerpts:")
    for i, doc in enumerate(retrieved["ildc"], 1):
        lines.append(f"{i}. {doc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_labels, true_label, k):
    return 1.0 if true_label in retrieved_labels[:k] else 0.0


def mean_reciprocal_rank(retrieved_labels, true_label):
    for rank, lbl in enumerate(retrieved_labels, 1):
        if lbl == true_label:
            return 1.0 / rank
    return 0.0


def majority_label_baseline(train_df, test_df, k_values=(5, 10)):
    """
    Retrieves the k most frequent training labels for every query,
    regardless of the query text. The difference between this and the
    embedder's numbers is the signal the embedder is actually adding.
    """
    counts = train_df["label"].value_counts()
    majority_sequence = list(counts.index)
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    for _, row in test_df.iterrows():
        n = max(k_values)
        retrieved = majority_sequence[:n]
        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved, row["label"], k))
        results["mrr"].append(mean_reciprocal_rank(retrieved, row["label"]))
    return {m: float(np.mean(v)) for m, v in results.items()}


def evaluate_retrieval(test_df, embedder, cuad_collection, k_values=(5, 10)):
    """
    Label-based relevance on the CUAD side. A retrieved clause counts as
    relevant if it shares the query's ground-truth category.
    """
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    n = max(k_values)
    for _, row in test_df.iterrows():
        qe = embedder.encode([row["text"]])
        res = cuad_collection.query(qe, n_results=n)
        retrieved_labels = [m["label"] for m in res["metadatas"][0]] if res["metadatas"] else []
        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved_labels, row["label"], k))
        results["mrr"].append(mean_reciprocal_rank(retrieved_labels, row["label"]))
    return {m: float(np.mean(v)) for m, v in results.items()}
