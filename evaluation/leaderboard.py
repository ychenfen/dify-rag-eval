"""leaderboard.py — collect runs/*/score.json into a markdown table."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs", type=pathlib.Path)
    p.add_argument("--out", default="LEADERBOARD.md", type=pathlib.Path)
    args = p.parse_args()

    rows = []
    if args.runs_dir.exists():
        for d in sorted(args.runs_dir.iterdir()):
            score = d / "score.json"
            if score.exists():
                blob = json.loads(score.read_text())
                agg = blob.get("aggregate", {})
                rows.append(
                    {
                        "run": d.name,
                        "n": blob.get("n"),
                        **agg,
                    }
                )

    rows.sort(key=lambda r: -r.get("faithfulness", 0))

    header = (
        "| Run | n | Faith | C-Recall | A-Rel | p50 ms | p95 ms | $/1k |\n"
        "|-----|---|------:|---------:|------:|-------:|-------:|-----:|"
    )
    body = "\n".join(
        f"| {r['run']} | {r['n']} | {r.get('faithfulness',0):.2f} | "
        f"{r.get('context_recall',0):.2f} | {r.get('answer_relevancy',0):.2f} | "
        f"{r.get('latency_ms_p50',0):.0f} | {r.get('latency_ms_p95',0):.0f} | "
        f"{r.get('cost_per_1k_usd',0):.3f} |"
        for r in rows
    )

    today = datetime.date.today().isoformat()
    text = (
        f"# RAG-Eval Leaderboard (auto-generated {today})\n\n"
        f"{header}\n{body}\n\n"
        "Sorted by faithfulness. Lower latency / cost is better.\n"
    )
    args.out.write_text(text)
    print(f"wrote {args.out} with {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
