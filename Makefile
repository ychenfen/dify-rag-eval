.PHONY: install lint score leaderboard reproduce clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m ruff check evaluation/ scripts/ || true

score:
	$(PYTHON) evaluation/scorer.py \
		--input data/sample.jsonl \
		--cost-table '{"gpt-4o-mini":{"input_per_1k_usd":0.00015,"output_per_1k_usd":0.00060}}' \
		--output runs/sample/score.json

leaderboard:
	$(PYTHON) evaluation/leaderboard.py --runs-dir runs --out LEADERBOARD.md

reproduce: score leaderboard
	@echo "Reproduced sample run. For full eval, run \`ragev run --config configs/baseline.yaml\`."

clean:
	rm -rf runs/
