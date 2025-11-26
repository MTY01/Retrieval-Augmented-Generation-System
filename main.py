from dataset_loader import load_collection
from retrieval import SparseRetriever, StaticRetriever, DenseRetriever, DenseRetrieverIns
from generation import RAGGenerator

import time


def rag_answer(question: str, retriever: DenseRetriever, generator: RAGGenerator, top_k: int = 5):
    results = retriever.retrieve(question, top_k=top_k)
    contexts = [doc for doc, _ in results]
    answer = generator.generate(question, contexts)

    # return docs for inspection/eval
    return answer, results

def rag_answer_dual(
    question: str,
    retriever_e5: DenseRetriever,
    retriever_qwen: DenseRetrieverIns,
    generator: RAGGenerator,
    top_k: int = 5,
    candidate_k: int = 20
):
    """
    Dual-retriever RAG pipeline:
    1. Use E5 for initial recall (candidate_k docs).
    2. Use Qwen3-Embedding to rerank candidates.
    3. Pass top_k reranked docs into generator.
    """

    # Step 1: E5 initial recall
    candidates = retriever_e5.retrieve(question, top_k=candidate_k)

    # Step 2: Qwen3 rerank
    reranked = retriever_qwen.rerank(question, candidates)

    # Step 3: Select top_k docs
    top_docs = [doc for doc, _ in reranked[:top_k]]

    # Step 4: Generate answer
    answer = generator.generate(question, top_docs)

    return answer, reranked[:top_k]

def main():
    time_start = time.time()

    # Load documents
    docs = load_collection("data/collection.jsonl")
    
    # Build retriever for BM25
    # time_build_retriever = time.time()
    # print("Start to build doc index!")
    # retriever = SparseRetriever(docs)
    # print(f"Time spent in retriever: {time.time() - time_build_retriever:.2f} s")
    
    # Build retriever for model2vec
    # retriever = StaticRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # print("Start to build doc index!")
    # time_build_retriever = time.time()
    # retriever.build_index(docs, batch_size=64)
    # print(f"Time spent in retriever: {time.time() - time_build_retriever:.2f} s")

    # Build retriever for E5
    # retriever = DenseRetriever(model_name="intfloat/e5-base-v2")
    # time_build_retriever = time.time()
    # print("Start to build doc index!")
    # retriever.build_index(docs)
    # print(f"Time spent in retriever: {time.time() - time_build_retriever:.2f} s")

    # Build retriever for Qwen3-0.6B
    # retriever = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-0.6B")
    # time_build_retriever = time.time()
    # print("Start to build doc index!")
    # retriever.build_index(docs)
    # print(f"Time spent in retriever: {time.time() - time_build_retriever:.2f} s")

    # Build retriever for ColBERT
    # TODO

    # Initialize generator
    # generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)

    # # Single-turn example
    # question = "Who wrote The Old Man and the Sea?"

    # time_generate_answer = time.time()
    # print("Start to generate answer!")

    # answer, doc = rag_answer(question, retriever, generator, top_k=5)

    # print(f"Time spent in generator: {time.time() - time_generate_answer:.2f} s")

    # print("Answer:", answer)

    # Track the doc
    # print("\nTop contexts (scores):")
    # for i, (doc, score) in enumerate(doc, 1):
    #     processed_doc = doc[:160].replace('\\n', ' ')
    #     print(f"{i}. {score:.4f} | {processed_doc}...")
    
    # -------------------------- Initialization --------------------------
    retriever_e5 = DenseRetriever(model_name="intfloat/e5-base-v2")
    retriever_qwen = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-0.6B")
    generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)

    # Read in json file
    question = ?
    answer, docs = rag_answer_dual(question, retriever_e5, retriever_qwen, generator, top_k=5)
    
    # This should be json file
    answer = ?
    
    print(f"Total time spent: {time.time() - time_start:.2f} s")

if __name__ == "__main__":
    main()
