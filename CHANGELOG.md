# Changelog — dify-rag-eval

## [0.1.0] — 2026-04-28
### Added
- Drop-in English README that frames the repo as a paper-grade evaluation suite.
- Five-dimension scorer (`evaluation/scorer.py`): faithfulness, context recall,
  answer relevancy, latency p50/p95, cost per 1k.
- Leaderboard generator (`evaluation/leaderboard.py`).
- Baseline run config (`configs/baseline.yaml`) with pinned models / cost table.
- Sample dataset (`data/sample.jsonl`, 3 rows zh + en) so `make reproduce` works
  out of the box without external services.
- GitHub Actions CI that scores the sample on every push.
- `Makefile` with `score`, `leaderboard`, `reproduce` targets.
- `data/LICENSES.md` declaring per-dataset licenses.

### Notes
- Heuristic scoring is a deterministic baseline; plug in a real LLM judge via
  `--llm-judge` (protocol in `docs/judges.md`, coming v0.2).
- Targeted submission venue: EMNLP 2026 findings / ACL short.
