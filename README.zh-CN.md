# Dify 知识库检索评测工具（RAG Evaluation）

本仓库用于评测 Dify Knowledge Base（知识库）的检索效果，便于对比不同配置下的召回率、精确率、F1、MRR、延迟等指标。

典型用途：
- 对比三种分块策略：通用 / 父子 / QA
- 对比 TopK（Top3/5/10）
- 对比是否启用重排（rerank）

## 工具链脚本

流程从左到右：
1. `build_evaluation_set.py`：从某个知识库生成候选问题集 `candidates.xlsx`
2. 人工审核：筛选可用问题，另存为 `evaluation_set.xlsx`
3. `rag_evaluator.py`：单知识库评测
4. `batch_evaluation.py`：批量对比评测（多知识库、多 TopK、重排开关）
5. `visualization.py`：生成可视化图表与报告
6. `run_evaluation.py`：一键跑完批量评测 + 可视化

## 快速开始

### 1) 安装依赖

```bash
python3 -m pip install -U pandas numpy openpyxl requests python-dotenv tqdm matplotlib seaborn jieba
```

### 2) 配置 `.env`

把 `.env.example` 复制为 `.env`，填入你自己的配置：

必填：
- `DIFY_API_KEY`

可选：
- `DIFY_API_BASE`（默认 `https://api.dify.ai/v1`）

用于“通用/父子/QA”三套知识库对比（可选，但 `run_evaluation.py` 需要）：
- `DATASET_ID_GENERAL`
- `DATASET_ID_PARENT_CHILD`
- `DATASET_ID_QA`

重排配置（可选，但必须与 Dify「系统模型设置」一致）：
- `RERANK_PROVIDER_NAME`：例如 `local` / `siliconflow`
- `RERANK_MODEL_NAME`：例如 `bge-reranker-base` / `BAAI/bge-reranker-v2-m3`

强烈建议（多知识库对比时）：
- `GOLD_MATCH_MODE=doc_name`

### 3) 构建候选评测集（可选）

```bash
python3 build_evaluation_set.py --action build --dataset-id <任意一个知识库ID> --output candidates.xlsx
```

然后在 `candidates.xlsx` 里人工审核：
- 把可用问题标 `is_valid=Y`
- 可选填 `category` / `difficulty`
- 另存为最终评测集 `evaluation_set.xlsx`

说明：
- 若 `evaluation_set.xlsx` 里存在 `is_valid` 列，并且至少有一行是 `Y`，评测时会自动只评测 `Y` 的行。

### 4) 单知识库评测

多知识库对比（通用/父子/QA）时，推荐用 **文档名** 做 gold：

```bash
python3 rag_evaluator.py \
  --dataset-id <知识库ID> \
  --eval-set evaluation_set.xlsx \
  --top-k 5 \
  --gold-match doc_name
```

启用重排：

```bash
python3 rag_evaluator.py \
  --dataset-id <知识库ID> \
  --eval-set evaluation_set.xlsx \
  --top-k 5 \
  --gold-match doc_name \
  --use-rerank \
  --rerank-provider siliconflow \
  --rerank-model BAAI/bge-reranker-v2-m3
```

### 5) 一键批量对比 + 可视化

```bash
python3 run_evaluation.py
```

输出目录示例：
- `results_20260209_235959/summary_*.json`（可视化输入）
- `results_20260209_235959/summary_*.xlsx`
- `results_20260209_235959/*.png`

## 为什么多知识库对比必须用 `gold_doc_name`

Dify 的 `document_id` 是“知识库内”唯一的：同一份文件上传到不同知识库，通常会得到不同的 `document_id`。

所以：
- `gold_doc_id` 适用于“只评测一个知识库”
- 对比“通用/父子/QA”多个知识库时，应使用 `gold_doc_name`，并在评测时加 `--gold-match doc_name`

## 示例数据

`examples/` 提供了一个很小的 demo 文档集，方便你先把工具链跑通：
- `examples/docs/`：可上传到 Dify 的 demo 文档
- `examples/evaluation_set_example.xlsx`：示例评测集（用 `gold_doc_name`）
- `examples/README.md`：使用说明

## 常见问题

请看：`docs/FAQ.zh-CN.md`
