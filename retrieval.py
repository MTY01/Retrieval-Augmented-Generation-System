from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

import torch


class DenseRetriever:
    def __init__(self, model_name="intfloat/e5-base-v2"):
        """
        Initialize the retriever with an E5 model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.nn = None
        self.documents = None
        self.embeddings = None

    def build_index(self, documents):
        """
        Build FAISS index from documents.
        """
        self.documents = documents
        self.embeddings = self.model.encode(documents, convert_to_numpy=True, normalize_embeddings=True, device=self.device)
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.nn.fit(self.embeddings)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a query.
        Returns list of (doc, score).
        """
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, device=self.device)
        distances, indices = self.nn.kneighbors(query_emb, n_neighbors=top_k)
        results = [(self.documents[i], 1 - float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results
