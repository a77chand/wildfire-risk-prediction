PY ?= python

.PHONY: install data train report export app test lint all
install:
	$(PY) -m pip install -e ".[app,dev]"

data:            ## download boundary, ignitions, weather and build the dataset (~5-10 min first time)
	$(PY) -m wildfire.cli fetch
	$(PY) -m wildfire.cli build

train:           ## fit models, evaluate on held-out years (--quick skips bootstrap / CV / ablation)
	$(PY) -m wildfire.cli train

report:          ## figures, SHAP, maps
	$(PY) -m wildfire.cli report

export:          ## bundle the small artefacts the app needs
	$(PY) -m wildfire.cli export

app:
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

all: data train report export
