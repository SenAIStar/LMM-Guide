@echo off
setlocal
python -m compileall -q src scripts tests || exit /b 1
python -m unittest discover -s tests -v || exit /b 1
python scripts\validate_data.py --input data\sample\audit_examples.jsonl --policy configs\policy.v1.json || exit /b 1
python scripts\score_rollouts.py --input data\sample\rollouts.jsonl --policy configs\policy.v1.json || exit /b 1
python scripts\evaluate_predictions.py --gold data\sample\audit_examples.jsonl --predictions data\sample\predictions.jsonl --policy configs\policy.v1.json || exit /b 1

