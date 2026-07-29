@echo off
setlocal

python -X utf8 -m unittest discover -s tests -v || exit /b 1
python -X utf8 scripts\validate_data.py data\sample\train.jsonl || exit /b 1
python -X utf8 scripts\validate_data.py data\sample\eval_sft.jsonl || exit /b 1
python -X utf8 scripts\evaluate_predictions.py data\sample\eval_gold.jsonl data\sample\eval_predictions.jsonl || exit /b 1
python -X utf8 -m compileall -q src scripts tests || exit /b 1

echo validation_complete
