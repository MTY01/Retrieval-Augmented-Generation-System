from dataset_loader import load_collection
from retrieval import DenseRetriever
from generation import RAGGenerator

import time


def rag_answer(question: str, retriever: DenseRetriever, generator: RAGGenerator, top_k: int = 5):
    results = retriever.retrieve(question, top_k=top_k)
    contexts = [doc for doc, _ in results]
    answer = generator.generate(question, contexts)

    # return docs for inspection/eval
    return answer, results


def main():
    time_start = time.time()

    # Load documents
    docs = load_collection("data/collection.jsonl")

    # Build retriever
    retriever = DenseRetriever(model_name="intfloat/e5-base-v2")

    time_build_retriever = time.time()
    print("Start to build doc index!")
    retriever.build_index(docs)
    print(f"Time spent in retriever: {time.time() - time_build_retriever:.2f} s")

    # Initialize generator
    generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)

    # Single-turn example
    question = "Who wrote The Old Man and the Sea?"

    time_generate_answer = time.time()
    print("Start to generate answer!")
    answer, doc = rag_answer(question, retriever, generator, top_k=5)

    print(f"Time spent in generator: {time.time() - time_generate_answer:.2f} s")

    print("Answer:", answer)

    print(f"Total time spent: {time.time() - time_start:.2f} s")

    # Track the doc
    # print("\nTop contexts (scores):")
    # for i, (doc, score) in enumerate(doc, 1):
    #     processed_doc = doc[:160].replace('\\n', ' ')
    #     print(f"{i}. {score:.4f} | {processed_doc}...")


if __name__ == "__main__":
    main()
