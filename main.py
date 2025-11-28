from dataset_loader import load_collection, load_split
from retrieval import SparseRetriever, StaticRetriever, DenseRetriever, DenseRetrieverIns, MultiVectorRetrieval
from generation import RAGGenerator
import json
import time

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
