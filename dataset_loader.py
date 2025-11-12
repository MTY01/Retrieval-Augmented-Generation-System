import json


def load_collection(path):
    """
    Load the collection.jsonl file into a list of documents.
    Each document is represented as its text string.

    Args:
        path (str): path to collection.jsonl

    Returns:
        list[str]: list of document texts
    """
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Some datasets may use "text" or "document" as the key
            text = obj.get("text") or obj.get("document")
            if text:
                documents.append(text)
    return documents


def load_split(path):
    """
    Load train/validation/test split files.
    Each line has: {"id":..., "text":..., "answer":..., "supporting_ids":[...]}

    Args:
        path (str): path to split.jsonl

    Returns:
        list[dict]: list of QA examples
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            examples.append(obj)
    return examples
