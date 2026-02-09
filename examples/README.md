# Examples

This folder contains a tiny, self-contained demo corpus you can upload into Dify to verify the evaluation toolchain end-to-end (build candidates -> manual review -> evaluate -> visualize).

## 1) Upload docs to Dify

Upload all files in `examples/docs/` into your Dify Knowledge Base.

Suggested setup for comparing chunk strategies:
- Create 3 Knowledge Bases (datasets) with the same documents:
  - general chunk
  - parent-child chunk
  - QA chunk
- Keep document names consistent across the 3 datasets (the default file name is usually fine).

## 2) Build candidate questions (optional)

```bash
python3 build_evaluation_set.py --action build --dataset-id <ONE_DATASET_ID> --output candidates.xlsx
```

Manually review `candidates.xlsx`, filter `is_valid=Y`, and save as `evaluation_set.xlsx`.
Keep `gold_doc_name` if you want to evaluate across multiple datasets.

## 3) Run evaluation (recommended: match by doc name)

```bash
python3 rag_evaluator.py \\
  --dataset-id <DATASET_ID> \\
  --eval-set evaluation_set.xlsx \\
  --top-k 5 \\
  --gold-match doc_name
```

If you use reranking via a third-party provider configured in Dify (for example `siliconflow`), pass:

```bash
python3 rag_evaluator.py ... --use-rerank --rerank-provider siliconflow --rerank-model BAAI/bge-reranker-v2-m3
```

## Notes

- The exact metrics depend on your Dify embedding model, reranker, indexing state, etc.
- These example docs are intentionally short; they are meant for smoke-testing the scripts rather than benchmarking.

