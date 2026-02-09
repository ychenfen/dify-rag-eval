# AGENTS.md (RAG repo)

## Scope
Applies to `/Users/yuchenxu/Desktop/RAG` and all subdirectories unless overridden by a nested `AGENTS.md`.

## Project Map (Module Map)
- `build_evaluation_set.py`, `rag_evaluator.py`, `batch_evaluation.py`, `visualization.py`, `run_evaluation.py`: Dify 知识库检索评测工具链（评测集构建 -> 评测 -> 汇总 -> 可视化）。
- `CRUD_RAG/`: CRUD-RAG 数据集与评测框架（见 `CRUD_RAG/AGENTS.md`）。
- `CRUD_RAG使用说明.md`, `CRUD_RAG快速入门教程.md`: 数据集与使用说明文档。
- `rag_evaluation_guide.*`: 评测指南源码与生成文件（`.tex` 为源，`.pdf` 为产物）。

## Global Rules
- 默认不要修改大规模数据文件（如 `CRUD_RAG/data/`）或生成文件（如 `rag_evaluation_guide.pdf`），除非明确要求。
- 新脚本优先放在仓库根目录；模块级代码放在对应模块目录中。
- 若涉及 Dify API 调用，必须通过 `.env` 提供 `DIFY_API_KEY` 和可选 `DIFY_API_BASE`。

## Cross-Domain Workflows
- 评测集构建流程：`build_evaluation_set.py` 生成 `candidates.xlsx` -> 人工审核筛选 -> 保存为最终 `evaluation_set.xlsx`。
- 评测执行流程：`rag_evaluator.py`（单次）或 `batch_evaluation.py`（批量） -> 输出 `results/` 或 `results_*` -> `visualization.py` 生成图表与报告。
- 一键流程：`run_evaluation.py` 依赖 `.env` 与 `evaluation_set.xlsx`，内部调用批量评测与可视化。

## Verification Guidance
- 单次评测（需已准备 `.env` 与 `evaluation_set.xlsx`）：`python rag_evaluator.py --dataset-id <ID> --eval-set evaluation_set.xlsx`。
- 批量评测：`python batch_evaluation.py`。
- 一键评测：`python run_evaluation.py`。

## Nested Guidance
- `CRUD_RAG/AGENTS.md`
