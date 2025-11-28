from dataset_loader import load_collection, load_split
from retrieval import SparseRetriever, StaticRetriever, DenseRetriever, DenseRetrieverIns, MultiVectorRetrieval
from generation import RAGGenerator
import json
import time


# This is for testing single model
def rag_answer(question: str, retriever, generator, top_k: int = 5):
    """
    General RAG pipeline:
    - Pass any retriever (DenseRetriever, ColBERTRetriever, BM25Retriever, etc.)
    - Collect contexts from retrieved docs
    - Generate answer with RAGGenerator
    """
    results = retriever.retrieve(question, top_k=top_k)

    # Normalize results: always extract doc["text"]
    contexts = []
    for doc, score in results:
        # If retriever returns dict directly, handle gracefully
        if isinstance(doc, dict) and "text" in doc:
            contexts.append(doc["text"])
        else:
            contexts.append(str(doc))

    answer = generator.generate(question, contexts)

    # Return both answer and retrieved docs for inspection/eval
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

def write_rag_answers(
    input_path: str,
    output_path: str,
    retriever_e5,
    retriever_qwen,
    generator,
    top_k: int = 5,
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

            # Step 1: E5 initial recall
            candidates = retriever_e5.retrieve(query, top_k=candidate_k)

            # Step 2: Qwen rerank
            reranked = retriever_qwen.rerank(query, candidates)

            # Step 3: Select top_k docs
            top_docs = [doc["text"] for doc, _ in reranked[:top_k]]

            # Step 4: Generate answer
            answer = generator.generate(query, top_docs)

            # Step 5: supporting_ids = first 10 id
            supporting_ids = [[doc["id"], score] for doc, score in reranked[:10]]

            # Step 6: Write JSONL
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
    # Output [[doc_id, text], [doc_id, text], ...]
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
    # docs = load_collection("data/tiny_collection.jsonl")
    # retriever = MultiVectorRetrieval(model_name="colbert-ir/colbertv2.0", use_gpu=True)
    # time_build_retriever = time.time()
    # print("Start to build doc index!")
    # retriever.build_index(docs)
    # print(f"Time spent in retriever: {time.time() - time_build_retriever:.2f} s")

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
    docs = load_collection("data/tiny_collection.jsonl")
    retriever_e5 = DenseRetriever(model_name="intfloat/e5-large-v2")
    
    print("Start to build doc index e5!")
    retriever_e5.build_index(docs)
    print(f"Time spent in e5: {time.time() - time_start:.2f} s")

    retriever_qwen = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-4B")
    generator = RAGGenerator(model_name="Qwen/Qwen2.5-3B-Instruct", max_new_tokens=128, temperature=0.0)
    write_rag_answers("data/test.jsonl", 
                      "data/test_predict.jsonl", 
                      retriever_e5, 
                      retriever_qwen, 
                      generator
                      )
    
    print(f"Total time spent: {time.time() - time_start:.2f} s")

if __name__ == "__main__":
    main()

# eval_retrieval.py
# {
#   "map_at_2": 0.552,
#   "map_at_5": 0.6704055555555556,
#   "map_at_10": 0.6919238095238094,
#   "ndcg_at_2": 0.6266158664801237,
#   "ndcg_at_5": 0.7542217637174936,
#   "ndcg_at_10": 0.7876044429284017,
#   "recall_at_2": 0.5903333333333334,
#   "recall_at_5": 0.8183333333333334,
#   "recall_at_10": 0.9016666666666666,
#   "precision_at_2": 0.5903333333333334,
#   "precision_at_5": 0.3273333333333333,
#   "precision_at_10": 0.18033333333333335
# }

# eval_hotpotqa.py
# {
#   "em": 0.366,
#   "f1": 0.4653739647244111,
#   "prec": 0.47359432243720057,
#   "recall": 0.5114029581529582,
#   "sp_em": 0.0,
#   "sp_f1": 0.46799999999999614,
#   "sp_prec": 0.3275999999999917,
#   "sp_recall": 0.819,
#   "joint_em": 0.0,
#   "joint_f1": 0.23560524173862615,
#   "joint_prec": 0.16789462689177898,
#   "joint_recall": 0.4551616883116881
# }

# New
# {
#   "em": 0.376,
#   "f1": 0.4982857125500713,
#   "prec": 0.5044795222186222,
#   "recall": 0.5535126003626004,
#   "sp_em": 0.0,
#   "sp_f1": 0.46799999999999614,
#   "sp_prec": 0.3275999999999917,
#   "sp_recall": 0.819,
#   "joint_em": 0.0,
#   "joint_f1": 0.24998344259964184,
#   "joint_prec": 0.17755366175041476,
#   "joint_recall": 0.4906165686165683
# }