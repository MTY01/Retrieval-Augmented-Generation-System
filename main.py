from dataset_loader import load_collection
from retrieval import DenseRetriever
from generation import RAGGenerator


def rag_answer(question: str, retriever: DenseRetriever, generator: RAGGenerator, top_k: int = 5):
    results = retriever.retrieve(question, top_k=top_k)
    contexts = [doc for doc, _ in results]
    answer = generator.generate(question, contexts)

    # return docs for inspection/eval
    return answer, results


def main():
    # Load documents
    docs = load_collection("data/collection.jsonl")

    # Build retriever
    retriever = DenseRetriever(model_name="intfloat/e5-base-v2")
    retriever.build_index(docs)

    # # Example query
    # query = "Where was Barack Obama born?"
    # results = retriever.retrieve(query, top_k=5)
    # # Test the retrieval module
    # for doc, score in results:
    #     print(f"{score:.4f} | {doc[:100]}...")

    # Initialize generator
    generator = RAGGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128, temperature=0.0)

    # Single-turn example
    question = "Who wrote The Old Man and the Sea?"
    answer, ctxs = rag_answer(question, retriever, generator, top_k=5)
    print("Answer:", answer)
    print("\nTop contexts (scores):")
    for i, (doc, score) in enumerate(ctxs, 1):
        processed_doc = doc[:160].replace('\\n', ' ')
        print(f"{i}. {score:.4f} | {processed_doc}...")


if __name__ == "__main__":
    main()
