from dataset_loader import load_collection, load_split
from retrieval import SparseRetriever, StaticRetriever, DenseRetriever, DenseRetrieverIns
from generation import RAGGenerator
import json
import time
from collections import defaultdict


# This is for testing single model
def rag_answer(question: str, retriever: DenseRetriever, generator: RAGGenerator, top_k: int = 5):
    results = retriever.retrieve(question, top_k=top_k)
    contexts = [doc for doc, _ in results]
    answer = generator.generate(question, contexts)

    # return docs for inspection/eval
    return answer, results

# This is for testing combine model
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
    time_e5_retrieve = time.time()
    print("Start to retrieve doc index")

    candidates = retriever_e5.retrieve(question, top_k=candidate_k)

    print(f"Time spent in e5 retrieve: {time.time() - time_e5_retrieve:.2f} s")

    time_qwen_rerank = time.time()
    print("Start to rerank doc index")

    # Step 2: Qwen3 rerank
    reranked = retriever_qwen.rerank(question, candidates)

    print(f"Time spent in qwen rerank: {time.time() - time_qwen_rerank:.2f} s")

    # Step 3: Select top_k docs
    top_docs = [doc["text"] for doc, _ in reranked[:top_k]]

    print("Start to generate answer!")
    # Step 4: Generate answer
    answer = generator.generate(question, top_docs)

    return answer, reranked[:top_k]

def fuse_results(bm25_results, e5_results, top_k=50, k=60, alpha=0.7):
    """
    Fuse BM25 and E5 retrieval results using Reciprocal Rank Fusion (RRF)
    with weighted bias toward E5 for higher precision.

    Parameters
    ----------
    bm25_results : list of dict
        Each dict must have {"doc": <doc>, "score": <float>}
    e5_results : list of dict
        Same format as bm25_results
    top_k : int
        Number of fused results to return
    k : int
        RRF constant (default 60)
    alpha : float
        Weight for E5 contribution (0.0–1.0). Higher alpha = more precision bias.

    Returns
    -------
    fused : list of dict
        Ranked list of fused results [{"doc": ..., "score": ...}]
    """

    scores = defaultdict(float)

    # Index BM25 ranks
    for rank, (doc, _) in enumerate(bm25_results, start=1):
        scores[doc["id"]] += (1 - alpha) * (1.0 / (k + rank))

    # Index E5 ranks
    for rank, (doc, _) in enumerate(e5_results, start=1):
        scores[doc["id"]] += alpha * (1.0 / (k + rank))
    
    # Merge metadata (keep highest score doc object)
    doc_map = {doc["id"]: doc for doc, _ in bm25_results + e5_results}

    # Sort by fused score
    fused = sorted(
        [(doc_map[doc_id], score) for doc_id, score in scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    return fused[:top_k]

# Write jsonl file
def write_rag_answers(
    input_path: str,
    output_path: str,
    retriever_e5,
    retriever_bm25, 
    retriever_qwen,
    generator,
    top_k: int = 10,
    candidate_k: int = 50
):
    """
    Generate RAG answers for a dataset split and write to JSONL.
    - input_path: train/validation/test split file (jsonl)
    - output_path: output jsonl file
    - retriever_e5: E5 retriever (for initial recall)
    - retriever_qwen: Qwen3 retriever (for rerank)
    - generator: RAGGenerator
    - top_k: number of final docs used for answer generation
    - candidate_k: number of docs recalled by E5 before rerank
    """

    examples = load_split(input_path)

    with open(output_path, "w", encoding="utf-8") as fout:
        for ex in examples:
            qid = ex["id"]
            query = ex["text"]

            # Step 1: BM initial recall
            bm25_candidates = retriever_bm25.retrieve(query, top_k=candidate_k)

            # Step 2: E5 initial recall
            e5_candidates = retriever_e5.retrieve(query, top_k=candidate_k)

            # Step 3: Fusion (union + score normalization)
            fused_candidates = fuse_results(bm25_candidates, e5_candidates, top_k=candidate_k)

            # Step 4: Qwen rerank
            reranked = retriever_qwen.rerank(query, fused_candidates)

            # Step 5: Select top_k docs
            top_docs = [doc["text"] for doc, _ in reranked[:top_k]]
            supporting_ids = [[doc["id"], score] for doc, score in reranked[:10]]

            # Step 6: Generate answer
            answer = generator.generate(query, top_docs)

            # Step 7: supporting_ids = first 10 doc id
            supporting_ids = [[doc["id"], score] for doc, score in reranked[:10]]

            # Step 8: write in JSONL
            out_obj = {
                "id": qid,
                "text": query,
                "answer": answer,
                "retrieved_docs": supporting_ids
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

def main():
    time_start = time.time()

    # Load documents
    # Previous output [[text], [text], ...]
    # Now output [{doc_id, text}, {doc_id, text}, ...]
    # TODO: modify all the build function inside model.
    docs = load_collection("data/collection.jsonl")
    

    # ----------------------- Test single model -----------------------
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
    # retriever = DenseRetriever(model_name="intfloat/e5-large-v2")
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

    # # Initialize generator
    # generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)

    # # Single-turn example
    # question = "Who wrote The Old Man and the Sea?"

    # time_generate_answer = time.time()
    # print("Start to generate answer!")

    # answer, doc = rag_answer(question, retriever, generator, top_k=5)

    # print(f"Time spent in generator: {time.time() - time_generate_answer:.2f} s")

    # print("Answer:", answer)

    # ----------------------------- End of test -----------------------------
    
    
    # -------------------------- Test combine model --------------------------
    # retriever_e5 = DenseRetriever(model_name="intfloat/e5-large-v2")
    
    # print("Start to build doc index e5!")
    # retriever_e5.build_index(docs)
    # print(f"Time spent in e5: {time.time() - time_start:.2f} s")

    # retriever_qwen = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-0.6B")
    # generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)

    # question = "Who wrote The Old Man and the Sea?"
    # answer, docs = rag_answer_dual(question, retriever_e5, retriever_qwen, generator, top_k=5)
    # print("Answer:", answer)

    # print(f"Total time spent: {time.time() - time_start:.2f} s")
    # ----------------------------- End of test -----------------------------


    # ------------------------ Output jsonl file ------------------------
    input_file = "data/validation.jsonl"
    output_file = "data/rag_answer.jsonl"
    mode = 0
    if mode:
        docs = load_collection("data/tiny_collection.jsonl")
        input_file = "data/tiny_test.jsonl"
        output_file = "data/tiny_answer.jsonl"
        
    docs = load_collection("data/tiny_collection.jsonl")
    retriever_e5 = DenseRetriever(model_name="intfloat/e5-large-v2")
    
    print("Start to build doc index e5!")
    time_e5 = time.time()
    retriever_e5.build_index(docs)
    print(f"Time spent in e5: {time.time() - time_e5:.2f} s")

    time_bm25 = time.time()
    print("Start to build doc index bm25!")
    retriever_bm25 = SparseRetriever(docs)
    print(f"Time spent in e5: {time.time() - time_bm25:.2f} s")

    retriever_qwen = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-4B")
    generator = RAGGenerator(model_name="Qwen/Qwen2.5-3B-Instruct", max_new_tokens=128, temperature=0.0)
    write_rag_answers(input_file, 
                      output_file, 
                      retriever_e5, 
                      retriever_bm25, 
                      retriever_qwen, 
                      generator
                      )
    
    print(f"Total time spent: {time.time() - time_start:.2f} s")

if __name__ == "__main__":
    main()
