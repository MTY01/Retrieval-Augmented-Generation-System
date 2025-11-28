from dataset_loader import load_collection, load_split
from retrieval import SparseRetriever, StaticRetriever, DenseRetriever, DenseRetrieverIns
from generation import RAGGenerator
import json
import time


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

def write_rag_answers(
    input_path: str,
    output_path: str,
    retriever,
    generator,
    top_k: int = 10,
):
    """
    Generate RAG answers for a dataset split and write to JSONL.
    - retriever: Qwen3 retriever (既负责召回也负责 rerank)
    - generator: RAGGenerator
    - top_k: number of final docs used for answer generation
    """

    examples = load_split(input_path)

    with open(output_path, "w", encoding="utf-8") as fout:
        for ex in examples:
            qid = ex["id"]
            query = ex["text"]

            candidates = retriever.retrieve(query, top_k=top_k)

            contexts = [doc["text"] for doc, _ in candidates]

            answer = generator.generate(query, contexts)

            supporting_ids = [[doc["id"], score] for doc, score in candidates]

            # Step 5: 写入 JSONL
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
    # Now output [[doc_id, text], [doc_id, text], ...]
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

    # Initialize generator
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
        input_file = "data/tiny_validation.jsonl"
        output_file = "data/tiny_rag_answer.jsonl"

    retriever_qwen = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-0.6B")
    print("Start to build doc index Qwen3!")
    retriever_qwen.build_index(docs)
    print(f"Time spent in Qwen3 index build: {time.time() - time_start:.2f} s")

    generator = RAGGenerator(model_name="Qwen/Qwen2.5-3B-Instruct", max_new_tokens=128, temperature=0.0)
    write_rag_answers(input_file, 
                      output_file, 
                      retriever_qwen, 
                      generator
                      )
    
    print(f"Total time spent: {time.time() - time_start:.2f} s")

if __name__ == "__main__":
    main()

# eval_retrieval.py
# {
#   "map_at_2": 0.5153333333333333,
#   "map_at_5": 0.6193166666666666,
#   "map_at_10": 0.6369358465608466,
#   "ndcg_at_2": 0.5913414830741336,
#   "ndcg_at_5": 0.7071959694045842,
#   "ndcg_at_10": 0.7347611783104457,
#   "recall_at_2": 0.5476666666666666,
#   "recall_at_5": 0.754,
#   "recall_at_10": 0.823,
#   "precision_at_2": 0.5476666666666666,
#   "precision_at_5": 0.3016,
#   "precision_at_10": 0.1646
# }

# # eval_hotpotqa.py
# {
#   "em": 0.14333333333333334,
#   "f1": 0.20776486412819511,
#   "prec": 0.19955357567500887,
#   "recall": 0.2693722222222224,
#   "sp_em": 0.0,
#   "sp_f1": 0.42990476190475907,
#   "sp_prec": 0.30093333333332617,
#   "sp_recall": 0.7523333333333333,
#   "joint_em": 0.0,
#   "joint_f1": 0.09684338244528023,
#   "joint_prec": 0.06673463842670826,
#   "joint_recall": 0.2290674603174603
# }
# # New
# {
#   "em": 0.32866666666666666,
#   "f1": 0.42896923090344863,
#   "prec": 0.4368431004077861,
#   "recall": 0.4775508713508714,
#   "sp_em": 0.0,
#   "sp_f1": 0.42990476190475907,
#   "sp_prec": 0.30093333333332617,
#   "sp_recall": 0.7523333333333333,
#   "joint_em": 0.0,
#   "joint_f1": 0.20306789333963282,
#   "joint_prec": 0.14496353398087233,
#   "joint_recall": 0.39955281662781666
# }