# Retrieval-Augmented Generation (RAG) System

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline.  
This RAG system using **E5 + Qwen3-Embedding rerank** as hybrid retriever and **Qwen2.5-instruct** as generator.
It leverages a knowledge base to answer multiple questions in batch mode, producing structured outputs with retrieved document scores.

This system includes **five** retrieval modules in total: 
- Sparse retrieval (BM25Plus)
- Static embedding retrieval (model2vec MiniLM-L6)
- Dense retrieval (e5-large-v2)
- Dense retrieval with instruction (Qwen3-Embedding)
- Multi-vector retrieval (colbert)

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
It requires **CUDA Toolkit 11.8** for GPU acceleration.

1. **Install Conda** (if not already installed)  
   Follow the [Miniconda installation guide](https://docs.conda.io/en/latest/miniconda.html).

2. Make sure your system has **CUDA Toolkit 11.8** installed and available in PATH:  
   ```bash
   nvcc --version   # should show release 11.8
   ```

2. **Create Environment from `environment.yml`**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate Environment**
   ```bash
   conda activate rag_env
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
5. Test results in:
   ```bash
   python eval_retrieval.py --gold data/validation.jsonl --pred test_predict.jsonl
   ```

   ```bash
   python eval_hotpotqa.py --gold data/validation.jsonl --pred test_predict.jsonl
   ```
6. Evaluation results：

   Output by eval_retrieval.py:
   ```
   {
     "map_at_2": 0.552,
     "map_at_5": 0.6704055555555556,
     "map_at_10": 0.6919238095238094,
     "ndcg_at_2": 0.6266158664801237,
     "ndcg_at_5": 0.7542217637174936,
     "ndcg_at_10": 0.7876044429284017,
     "recall_at_2": 0.5903333333333334,
     "recall_at_5": 0.8183333333333334,
     "recall_at_10": 0.9016666666666666,
     "precision_at_2": 0.5903333333333334,
     "precision_at_5": 0.3273333333333333,
     "precision_at_10": 0.18033333333333335
   }
   ```

   Output by eval_hotpotqa.py:
   ```
   {
     "em": 0.376,
     "f1": 0.4982857125500713,
     "prec": 0.5044795222186222,
     "recall": 0.5535126003626004,
     "sp_em": 0.0,
     "sp_f1": 0.46799999999999614,
     "sp_prec": 0.3275999999999917,
     "sp_recall": 0.819,
     "joint_em": 0.0,
     "joint_f1": 0.24998344259964184,
     "joint_prec": 0.17755366175041476,
     "joint_recall": 0.4906165686165683
   }
   ```
---

## 🖥️ Notes

- Ensure your GPU drivers and CUDA toolkit are properly installed for optimal performance.
- The environment file (`environment.yml`) includes all dependencies required for retrieval, generation, and evaluation.

