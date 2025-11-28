# Retrieval-Augmented Generation (RAG) System

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline.  
It leverages a knowledge base to answer multiple questions in batch mode, producing structured outputs with retrieved document scores.

---

## 📂 Data Format

- **Knowledge Base (`data/collection.jsonl`)**
  ```json
  {"id": "<str>", "text": "<str>"}
  ```

- **Input Questions (`data/test.jsonl`)**
  ```json
  {"id": "<str>", "text": "<str>"}
  ```

- **Output Answers (`data/test_predict.jsonl`)**
  ```text
  {
    "id": "<str>",
    "question": "<str>",
    "answer": "<str>",
    "retrieved_docs": [
      ["id_1", score_1],
      ["id_2", score_2],
      ...,
      ["id_10", score_10]
    ]
  }
  ```

---

## ⚙️ Environment Setup

This project is designed to run on **WSL2 Ubuntu Linux** with a **GPU (24GB VRAM)**.

1. **Install Conda** (if not already installed)  
   Follow the [Miniconda installation guide](https://docs.conda.io/en/latest/miniconda.html).

2. **Create Environment from `environment.yml`**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate Environment**
   ```bash
   conda activate rag-env
   ```

---

## 🚀 Running the Project

Once the environment is set up, simply run:

```bash
python main.py
```

This will:
- Load the knowledge base from `data/collection.jsonl`
- Process the batch of questions from `data/test.jsonl`
- Generate answers with retrieved document scores
- Save results to `data/test_predict.jsonl`

---

## 📁 Output Location

All generated `.jsonl` files will be saved automatically in the **`data/` folder**.

---

## ✅ Example Workflow

1. Prepare your knowledge base in `data/collection.jsonl`
2. Add your questions in `data/test.jsonl`
3. Run:
   ```bash
   python main.py
   ```
4. Check results in:
   ```
   data/test_predict.jsonl
   ```

---

## 🖥️ Notes

- Ensure your GPU drivers and CUDA toolkit are properly installed for optimal performance.
- The environment file (`environment.yml`) includes all dependencies required for retrieval, generation, and evaluation.

