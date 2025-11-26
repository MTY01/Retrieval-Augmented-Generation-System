from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Plus
import numpy as np
import nltk
import faiss
import torch
from typing import List


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
# Model: E5-base-v2
# =====================================

class StaticRetriever:
    def __init__(self, model_name="intfloat/e5-base-v2", use_gpu=True):
        """
        Dense retriever using SentenceTransformer + FAISS (E5-base-v2).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.index = None
        self.documents = None
        self.embeddings = None
        self.use_gpu = use_gpu

    def build_index(self, documents, batch_size=512):
        """
        Build FAISS index from documents with batch encoding.
        Adds 'passage:' prefix for E5.
        """
        self.documents = documents
        prefixed_docs = [f"passage: {doc}" for doc in documents]

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
        results = [(self.documents[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
        return results


# =====================================
# Dense Retriever Module with instruction
# Model: E5-Mistral
# =====================================

# class DenseRetrieverIns:
    


# =====================================
# Dense Multi-vector Retrieval
# Model: 
# =====================================

# class MultiVectorRetrieval:
