# 🚀 Course Completion Prediction System (MLOps Project)

An end-to-end **production-grade machine learning system** that predicts whether a student will complete an online course based on engagement data.

This project demonstrates **real-world MLOps practices** including automation, containerization, CI/CD, and cloud deployment.

---

## 📌 Project Overview

This system simulates an industry-level ML pipeline:

- Data ingestion from AWS S3
- Automated training using Airflow
- Model storage & versioning
- Real-time inference via FastAPI
- CI/CD pipeline with GitHub Actions
- Dockerized deployment ready for AWS ECS

---

## 🗂️ Project Structure
```
ML_MAIN_PROJECT/
├── .github/workflows/
│ └── ci.yml
├── airflow/
│ ├── dags/
│ │ └── ml_pipeline.py
│ ├── docker-compose.yaml
│ └── Dockerfile
├── api/
│ └── main.py
├── docker/
│ └── Dockerfile.api
├── src/
│ ├── training/
│ │ └── script.py
│ └── model_class.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🚀 Key Highlights

- End-to-end ML pipeline (**data → training → evaluation → deployment**)
- REST API for real-time predictions using FastAPI
- Automated CI/CD pipeline using GitHub Actions
- Containerized services using Docker
- Cloud integration with AWS (S3, ECR)
- Workflow orchestration using Apache Airflow
- Modular and scalable code structure

---

## 📸 Preview

### 🔹 API (Swagger UI)
<img width="1916" height="1092" alt="swagger png" src="https://github.com/user-attachments/assets/2b45e568-8cfc-407c-95e3-6303990b991c" />


### 🔹 AWS ECS Deployment
<img width="1916" height="913" alt="ecs png" src="https://github.com/user-attachments/assets/644e3542-87f7-4a64-8ad1-0f79205ccb67" />

### 🔹 CI/CD Pipeline
<img width="1908" height="1097" alt="ci png" src="https://github.com/user-attachments/assets/ae9ecc8e-d3d9-4fdc-981c-322218e35684" />

---

## 🏗️ Architecture Overview
![kweEo](https://github.com/user-attachments/assets/6a159188-f2fc-422c-9415-85c9a64bf072)
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
Trained Model (Stored in S3)
│
▼
FastAPI Service (Inference)
│
▼
Docker Container
│
▼
AWS ECR → ECS Deployment
```

---

## 🧰 Tech Stack

### 🔹 Languages & ML
- Python
- Pandas, Scikit-learn

### 🔹 Backend
- FastAPI
- Pydantic

### 🔹 MLOps & DevOps
- Docker
- GitHub Actions (CI/CD)
- Apache Airflow

### 🔹 Cloud (AWS)
- S3 (data & model storage)
- ECR (container registry)
- ECS (deployment-ready)
- Lambda (optional retraining trigger)

---

## ⚙️ ML Pipeline

- Data ingestion from AWS S3
- Data preprocessing & cleaning
- Feature engineering (one-hot encoding)
- Model training (Random Forest)
- Model evaluation
- Model serialization & upload to S3

---

## 🔌 API Endpoints

### ✅ GET /
Health check endpoint

### ✅ POST /predict

#### Input:
```
json
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
Output:
```
{
  "prediction": "Completed"
}

```
🐳 Running Locally
🔹 Using Docker

```
docker build -t course-completion-api -f docker/Dockerfile.api .
docker run -p 8000:8000 course-completion-api

```
Access API:
```
http://127.0.0.1:8000/docs
```
---

🔄 CI/CD Pipeline

Automated workflow using GitHub Actions:
Install dependencies
Run tests (pytest)
Build Docker image
Push image to AWS ECR
---

📊 Airflow Orchestration
DAG: ml_training_pipeline
Automates training workflow
Fetches data from S3
Stores trained model back to S3
---

🎯 Project Goals
Build a production-ready ML system
Apply real-world MLOps practices
Demonstrate deployment-ready ML engineering skills
---

📈 Future Improvements
Model versioning (MLflow)
Monitoring & logging (Prometheus/Grafana)
Full ECS deployment automation
Feature store integration
---

👨‍💻 Author
Mohammed Feroz Shaik

