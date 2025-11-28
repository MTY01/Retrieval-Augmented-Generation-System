from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Plus
import numpy as np
import nltk
import faiss
import torch
from typing import List, Tuple


# =====================================
# Sparse Retriever Module
# Model: BM25Plus
# =====================================

# First time you have to download this
# nltk.download("punkt")
# nltk.download("punkt_tab")

class SparseRetriever:
    def __init__(self, documents):
        """
        Initialize BM25 retriever with a list of documents.
        """
        self.documents = documents
        # Tokenize each document into words
        self.tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Plus(self.tokenized_docs)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a query.
        Returns list of (doc, score).
        """
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        # Sort by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self.documents[i], float(scores[i])) for i in top_indices]
        return results


# =====================================
# Static Embedding Retrieval
# Model: model2vec
# =====================================

class StaticRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True):
        """
        Static retriever using SentenceTransformer (Model2Vec) + FAISS.
        Embeddings are precomputed once and reused.
        """
        device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

        self.index = None
        self.documents = None
        self.embeddings = None
        self.use_gpu = use_gpu

    def build_index(self, documents, batch_size=64):
        """
        Build FAISS index from documents.
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]

        cpu_index = faiss.IndexFlatIP(dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a query.
        """
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        results = [(self.documents[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
        return results


# =====================================
# Dense Retriever Module
# Model: E5-large-v2
# =====================================

class DenseRetriever:
    def __init__(self, model_name="intfloat/e5-large-v2", use_gpu=True):
        """
        Dense retriever using SentenceTransformer + FAISS (E5-large-v2).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.index = None
        self.documents = None
        self.embeddings = None
        self.use_gpu = use_gpu

    def build_index(self, documents, batch_size=128):
        """
        Build FAISS index from documents with batch encoding.
        Adds 'passage:' prefix for E5.
        """
        self.documents = documents
        prefixed_docs = [f"{doc['text']}" for doc in documents]

        self.embeddings = self.model.encode(
            prefixed_docs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=True
        )
        dim = self.embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)

        # Inner product index (cosine similarity if normalized)
        if self.use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a query.
        Adds 'query:' prefix for E5.
        """
        query_emb = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for j, i in enumerate(indices[0]):
            doc = self.documents[i]   # dict: {"id":..., "text":...}
            results.append((doc, float(scores[0][j])))
        return results


# =====================================
# Dense Retriever Module with instruction
# Model: Qwen3-Embedding-0.6B
# =====================================

class DenseRetrieverIns:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", 
                 show_progress_bar: bool = True, use_gpu: bool = True, 
                 use_fp16: bool = True):
        device = "cuda" if use_gpu else "cpu"
        dtype = torch.float16 if use_fp16 else torch.float32
        
        self.show_progress_bar = show_progress_bar
        self.use_gpu = use_gpu
        self.model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": dtype})
        self.index = None
        self.documents = []

    def build_index(self, docs: List[str]):
        self.documents = docs
        embeddings = self.model.encode(
            docs,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dim)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 10):
        query_emb = self.model.encode(
            [query],
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        return [(self.documents[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    
    def rerank(self, query: str, candidates: List[Tuple[str, float]]):
        """
        Rerank a candidate set of documents using Qwen embeddings.
        candidates: List of (doc, score_from_other_retriever)
        Returns: List of (doc, rerank_score) sorted by rerank_score desc
        """
        # Encode query
        query_emb = self.model.encode(
            [query],
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")[0]

        # Encode candidate docs
        docs = [doc["text"] for doc, _ in candidates]
        doc_embs = self.model.encode(
            docs,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # Cosine similarity
        scores = np.dot(doc_embs, query_emb)  # since embeddings are normalized
        reranked = sorted(
            [(candidates[i][0], float(scores[i])) for i in range(len(candidates))],
            key=lambda x: x[1],
            reverse=True
        )

        return reranked


# =====================================
# Dense Multi-vector Retrieval
# Model: 
# =====================================

# class MultiVectorRetrieval:
