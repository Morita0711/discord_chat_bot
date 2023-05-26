.PHONY: start
start:
	uvicorn main:app --reload --port 9000

.PHONY: format
format:
	black .
	isort .
image: 
	@make ingest
	@gcloud builds submit --tag gcr.io/mineonlium/partyllm

ingest:
	@python ingest.py