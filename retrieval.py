from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Plus
import nltk
# import faiss
import torch


# =====================================
# Dense Retriever Module
# Model: BGE-m3
# =====================================

class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-m3"):
        """
        Initialize the retriever with a BGE-m3 model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.documents = None
        self.embeddings = None

    def build_index(self, documents):
        """
        Build FAISS index from documents.
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            documents, 
            batch_size=256, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            device=self.device)
        dim = self.embeddings.shape[1]

        # Inner product index (cosine similarity if normalized)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a query.
        Returns list of (doc, score).
        """
        query_emb = self.model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            device=self.device)
        distances, indices = self.index.search(query_emb, n_neighbors=top_k)
        results = [(self.documents[i], 1 - float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results


# =====================================
# Sparse Retriever Module
# Model: BM25
# =====================================

# First time you have to download this
# nltk.download("punkt")

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


