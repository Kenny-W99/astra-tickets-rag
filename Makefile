.PHONY: install ingest api ui eval format

install:
	python -m pip install -r requirements.txt

ingest:
	python -m src.ingest

api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run src/ui.py --server.port 8501

eval:
	python -m src.eval
