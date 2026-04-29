# dify-rag-eval

A reproducible, multi-dimensional evaluation suite for Retrieval-Augmented Generation (RAG)
pipelines built on [Dify](https://dify.ai). We provide datasets, scoring scripts, and a
leaderboard so practitioners can compare retriever × reranker × generator combinations on
the same footing.

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/ychenfen/dify-rag-eval/ci.yml?branch=main)](.github/workflows/ci.yml)

## What we measure

| Dimension | Metric | Notes |
|-----------|--------|-------|
| Faithfulness | Ragas faithfulness + sampled manual audit | Hallucination rate vs retrieved evidence |
| Context recall | Ragas context_recall | Did retrieval cover the gold answer? |
| Answer relevance | Ragas answer_relevancy + LLM-as-judge | Does the answer address the question? |
| Latency | p50 / p95 wall clock | Includes retrieval + generation |
| Cost | $ per 1k queries | Token + vector DB + reranker fees |

We deliberately publish **all five**: cost / latency are usually omitted, yet are exactly
what production teams need.

## Quick start

```bash
git clone https://github.com/ychenfen/dify-rag-eval.git
cd dify-rag-eval
pip install -e .

# Run baseline against a local Dify instance
ragev run --config configs/baseline.yaml

# View leaderboard from the latest run
ragev leaderboard
```

## Datasets

| Name | Domain | Q-A pairs | Source |
|------|--------|-----------|--------|
| `tech-faq-zh-1k` | Chinese tech-support FAQs | 1 048 | scraped public KB, manually cleaned |
| `mschat-en-500`  | English multi-turn chat | 500 | MS-MARCO subset |
| `cn-policy-pair` | Chinese policy documents | 320 | hand-built |

License of each dataset is documented in `data/LICENSES.md`.

## Reproducibility

- Pinned models / API versions in `configs/*.yaml`
- Hardware-agnostic: every metric is computed offline from logged traces
- One-click reproduce: `make reproduce` re-runs the table in the paper

## Citing

```bibtex
@misc{dify-rag-eval-2026,
  title  = {dify-rag-eval: A Multi-Dimensional Reproducible RAG Evaluation Suite},
  author = {ychenfen},
  year   = {2026},
  url    = {https://github.com/ychenfen/dify-rag-eval}
}
```

## Roadmap

- [x] v0.1 — three datasets, five metrics, baseline config
- [ ] v0.2 — additional rerankers (Cohere, BGE-M3) and one ablation study
- [ ] v0.3 — leaderboard hosted on GitHub Pages, community PRs welcome
- [ ] v1.0 — short paper submitted (target: EMNLP findings)

## Acknowledgements

Built on top of [Dify](https://dify.ai), [Ragas](https://github.com/explodinggradients/ragas)
and the broader open-source RAG ecosystem.
