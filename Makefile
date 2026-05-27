.PHONY: setup repro tune train score evaluate test api dashboard update-deps update-readme-assets

# ── Setup ─────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	pip install -e .

# ── Pipeline ──────────────────────────────────────────────
repro:
	dvc repro

tune:
	python src/models/tune.py

train:
	python src/models/train.py

evaluate:
	python src/models/evaluate.py

score:
	python src/models/predict.py

# ── Services ──────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

# ── Tests ─────────────────────────────────────────────────
test:
	pytest -v

test-api:
	pytest api/tests/test_api.py -v

test-preprocessing:
	pytest src/tests/test_preprocessing.py -v

# ── Dependencies ──────────────────────────────────────────
update-deps:
	pip-compile requirements.in --output-file requirements.txt

# ── README assets ─────────────────────────────────────────
update-readme-assets:
	cp artifacts/shap_summary.png images/shap_summary.png
	cp artifacts/shap_beeswarm.png images/shap_beeswarm.png