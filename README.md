# Dify Knowledge Base Retrieval Evaluation (RAG)

This repo provides a simple toolchain to evaluate Dify Knowledge Base retrieval quality across different configurations (chunking strategy, TopK, reranking on/off).

Chinese docs:
- `README.zh-CN.md`
- `docs/FAQ.zh-CN.md`

Core scripts (pipeline):
1. `build_evaluation_set.py`: build candidate questions from an existing Dify dataset (Knowledge Base)
2. Manual review: filter candidates and save as `evaluation_set.xlsx`
3. `rag_evaluator.py`: run evaluation for one dataset
4. `batch_evaluation.py`: compare multiple datasets/configs in batch
5. `visualization.py`: generate charts/reports from summary JSON
6. `run_evaluation.py`: one-click batch evaluation + visualization

## Quickstart

### 0) Install dependencies

This project is pure Python. Install the usual data stack:

```bash
python3 -m pip install -U pandas numpy openpyxl requests python-dotenv tqdm matplotlib seaborn jieba
```

### 1) Configure `.env`

Required:
- `DIFY_API_KEY=...`

Optional:
- `DIFY_API_BASE=https://api.dify.ai/v1` (default)

For batch comparison (3 chunking strategies):
- `DATASET_ID_GENERAL=...`
- `DATASET_ID_PARENT_CHILD=...`
- `DATASET_ID_QA=...`

For reranking (must match Dify "System Model Settings"):
- `RERANK_PROVIDER_NAME=local` or `siliconflow` ...
- `RERANK_MODEL_NAME=bge-reranker-base` or `BAAI/bge-reranker-v2-m3` ...

For multi-dataset evaluation correctness:
- `GOLD_MATCH_MODE=doc_name` (recommended when comparing different datasets)

### 2) Build candidates (optional)

```bash
python3 build_evaluation_set.py --action build --dataset-id <ONE_DATASET_ID> --output candidates.xlsx
```

Then manually review `candidates.xlsx`:
- mark `is_valid=Y` for good rows
- fill `category` / `difficulty` if you want grouped analysis
- save as `evaluation_set.xlsx`

### 3) Run single evaluation

Recommended for comparing multiple datasets: match gold by **document name**.

```bash
python3 rag_evaluator.py \
  --dataset-id <DATASET_ID> \
  --eval-set evaluation_set.xlsx \
  --top-k 5 \
  --gold-match doc_name
```

With reranking:

```bash
python3 rag_evaluator.py \
  --dataset-id <DATASET_ID> \
  --eval-set evaluation_set.xlsx \
  --top-k 5 \
  --gold-match doc_name \
  --use-rerank \
  --rerank-provider siliconflow \
  --rerank-model BAAI/bge-reranker-v2-m3
```

### 4) Batch compare + visualization

```bash
python3 run_evaluation.py
```

Outputs:
- `results_<timestamp>/summary_*.json|.xlsx`
- charts: `*.png`

## Why `gold_doc_name` matters

`gold_doc_id` is **dataset-scoped** in Dify: the same file uploaded to different Knowledge Bases usually gets different `document_id`.
So for comparing chunk strategies (general vs parent-child vs QA), use:
- `gold_doc_name` in the evaluation set, and
- `--gold-match doc_name` when evaluating.

## Examples

See `examples/` for a tiny demo corpus and a sample `evaluation_set_example.xlsx` you can use for smoke-testing.

## Submodule

`CRUD_RAG/` is included as a git submodule pointing to `https://github.com/IAAR-Shanghai/CRUD_RAG.git`.
It is not required for running the Dify evaluation scripts.
