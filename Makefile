# Variables
IMAGE_NAME=course-completion-api
REGISTRY=685251218327.dkr.ecr.eu-north-1.amazonaws.com

.PHONY: install test build train push up-airflow down-airflow mlflow-ui

# 1. Install local dependencies
install:
	pip install -r requirements.txt
	pip install pytest httpx

# 2. Run unit tests
test:
	pytest

# 3. Run Training with Hyperparameter Tuning & MLflow logging
train:
	python -m src.training.script

# 4. Start MLflow UI
mlflow-ui:
	mlflow ui --port 5000

# 5. Build the Docker image locally
build:
	docker build -f docker/Dockerfile.api -t $(IMAGE_NAME) .

# 6. Tag and Push to ECR
push:
	docker tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

# 7. Start Airflow
up-airflow:
	cd airflow && docker-compose up -d

down-airflow:
	cd airflow && docker-compose down