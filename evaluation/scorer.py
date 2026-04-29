"""scorer.py — compute the five evaluation dimensions for a logged run.

Input is a JSONL file where each row records one query:

    {
      "qid": "q-001",
      "question": "...",
      "gold_answer": "...",
      "retrieved_chunks": ["...", "..."],
      "answer": "...",
      "latency_ms": {"retrieval": 41, "generation": 612},
      "tokens": {"input": 384, "output": 96},
      "model": "gpt-4o-mini"
    }

Output is a single JSON dict with per-query and aggregate metric values.

Faithfulness / context-recall / answer-relevance use a small set of
deterministic, dependency-light heuristics so this scorer is reproducible
even without a paid LLM judge. Plug a real LLM judge in via `--llm-judge`
to override the heuristics — the protocol is documented in `docs/judges.md`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Heuristic metrics (fast + reproducible baseline)
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w一-鿿]+", text.lower()))


def faithfulness(answer: str, chunks: list[str]) -> float:
    """Fraction of answer tokens that appear in at least one chunk."""
    a = _tokens(answer)
    if not a:
        return 0.0
    pool = set().union(*(_tokens(c) for c in chunks)) if chunks else set()
    return len(a & pool) / len(a)


def context_recall(gold: str, chunks: list[str]) -> float:
    g = _tokens(gold)
    if not g:
        return 0.0
    pool = set().union(*(_tokens(c) for c in chunks)) if chunks else set()
    return len(g & pool) / len(g)


def answer_relevancy(answer: str, gold: str) -> float:
    a, g = _tokens(answer), _tokens(gold)
    if not (a and g):
        return 0.0
    return len(a & g) / len(a | g)  # Jaccard, simple proxy


# --------------------------------------------------------------------------- #
# Run-level metrics
# --------------------------------------------------------------------------- #


@dataclass
class CostRow:
    input_per_1k_usd: float
    output_per_1k_usd: float


def _cost_per_query(row: dict, table: dict) -> float:
    model = row.get("model")
    info = table.get(model)
    if not info:
        return 0.0
    return (
        row["tokens"]["input"] * info.get("input_per_1k_usd", 0) / 1000.0
        + row["tokens"]["output"] * info.get("output_per_1k_usd", 0) / 1000.0
    )


def _latency_ms(row: dict) -> float:
    lat = row.get("latency_ms", {})
    return float(lat.get("retrieval", 0) + lat.get("generation", 0))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def score_file(path: pathlib.Path, cost_table: dict) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"n": 0}

    per_q = []
    for r in rows:
        per_q.append(
            {
                "qid": r.get("qid"),
                "faithfulness": faithfulness(r.get("answer", ""), r.get("retrieved_chunks", [])),
                "context_recall": context_recall(r.get("gold_answer", ""), r.get("retrieved_chunks", [])),
                "answer_relevancy": answer_relevancy(r.get("answer", ""), r.get("gold_answer", "")),
                "latency_ms": _latency_ms(r),
                "cost_usd": _cost_per_query(r, cost_table),
            }
        )

    def agg(key: str, fn=statistics.mean):
        vals = [q[key] for q in per_q]
        return fn(vals) if vals else 0.0

    p50 = statistics.median([q["latency_ms"] for q in per_q])
    p95 = sorted(q["latency_ms"] for q in per_q)[int(0.95 * (len(per_q) - 1))]

    return {
        "n": len(per_q),
        "aggregate": {
            "faithfulness": agg("faithfulness"),
            "context_recall": agg("context_recall"),
            "answer_relevancy": agg("answer_relevancy"),
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "cost_per_1k_usd": 1000.0 * agg("cost_usd"),
        },
        "per_query": per_q,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--cost-table", default="{}", help="JSON-encoded cost table")
    p.add_argument("--output", default="runs/score.json", type=pathlib.Path)
    args = p.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    cost_table = json.loads(args.cost_table)
    result = score_file(args.input, cost_table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    agg = result.get("aggregate", {})
    print("== Aggregate ==")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:20s} {v:.3f}")
        else:
            print(f"  {k:20s} {v}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
