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
    retriever_e5,
    retriever_qwen,
    generator,
    top_k: int = 10,
    candidate_k: int = 20
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

            # Step 5: supporting_ids = 前两个文档的 id
            supporting_ids = [[doc["id"], score] for doc, score in reranked[:10]]

            # Step 6: 写入 JSONL
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
    retriever_e5 = DenseRetriever(model_name="intfloat/e5-large-v2")
    
    print("Start to build doc index e5!")
    retriever_e5.build_index(docs)
    print(f"Time spent in e5: {time.time() - time_start:.2f} s")

    retriever_qwen = DenseRetrieverIns(model_name="Qwen/Qwen3-Embedding-0.6B")
    generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)
    write_rag_answers("data/validation.jsonl", 
                      "data/rag_answer.jsonl", 
                      retriever_e5, 
                      retriever_qwen, 
                      generator
                      )
    
    print(f"Total time spent: {time.time() - time_start:.2f} s")

if __name__ == "__main__":
    main()
