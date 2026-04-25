.PHONY: install test train build up-airflow

install:
	uv sync

test:
	set CI=true && uv run pytest

train:
	uv run python -m src.training.script

build:
	docker build -f docker/Dockerfile.api -t course-completion-api .

up-airflow:
	cd airflow && docker-compose up -d