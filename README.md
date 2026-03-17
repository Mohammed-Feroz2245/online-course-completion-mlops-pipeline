# Course Completion Prediction System (MLOps Project)

An end-to-end **machine learning system with production-oriented MLOps practices** to predict whether a student will complete an online course based on engagement data.

This project focuses on **deployability, automation, and scalability**, simulating a real-world ML pipeline used in industry.

---

## 🚀 Key Highlights

* End-to-end ML pipeline (data → training → evaluation → deployment)
* REST API for real-time predictions using FastAPI
* Automated CI/CD pipeline using GitHub Actions
* Containerized services using Docker
* Cloud integration with AWS (S3, ECR)
* Workflow orchestration using Apache Airflow
* Modular and maintainable code structure

---

## 🏗️ Architecture Overview

```
Dataset (CSV)
     │
     ▼
AWS S3 (Storage)
     │
     ▼
Airflow Pipeline (Training)
     │
     ▼
Model Stored in S3
     │
     ▼
FastAPI Service (Inference)
     │
     ▼
Docker Container
     │
     ▼
AWS ECR → (Deployable to ECS)
```

---

## 🧰 Tech Stack

**Languages & ML**

* Python
* Pandas, Scikit-learn

**Backend**

* FastAPI
* Pydantic

**MLOps & DevOps**

* Docker
* GitHub Actions (CI/CD)
* Apache Airflow

**Cloud (AWS)**

* S3 (data + model storage)
* ECR (container registry)
* Lambda (optional retraining trigger)

---

## ⚙️ ML Pipeline

* Data ingestion from AWS S3
* Data cleaning and preprocessing
* Feature engineering (one-hot encoding)
* Model training (Random Forest)
* Model evaluation
* Model serialization and upload to S3

---

## 🔌 API Endpoints

### GET /

Health check endpoint

### POST /predict

Predict course completion

**Input:**

```json
{
  "age": 25,
  "hours_per_week": 10,
  "assignments_submitted": 5,
  "desktop": 1,
  "mobile": 0,
  "pager": 0,
  "smart_tv": 0,
  "tablet": 0
}
```

**Output:**

```json
{
  "prediction": "Completed"
}
```

---

## 🐳 Running Locally

### Using Docker

```bash
docker build -t course-completion-api -f docker/Dockerfile.api .
docker run -p 8000:8000 course-completion-api
```

Access API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow:

* Install dependencies
* Run tests using pytest
* Build Docker image
* Push image to AWS ECR

---

## 📊 Airflow Orchestration

* DAG: `ml_training_pipeline`
* Automates model training
* Fetches data from S3
* Uploads trained model back to S3

---

## 📌 Project Goals

* Build a production-style ML system
* Practice MLOps workflows
* Demonstrate deployment-ready ML engineering skills

---

## 📈 Future Improvements

* Model versioning (MLflow)
* Monitoring & logging
* Automated deployment to ECS
* Feature store integration

---

## 👨‍💻 Author

Mohammed Feroz
