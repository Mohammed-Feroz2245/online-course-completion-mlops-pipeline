from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Adds the project root to path so 'src' can be found
sys.path.append(os.getenv("PYTHONPATH", "/opt/project"))

def run_training():
    from src.training.script import train_and_upload
    return train_and_upload()

with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops"],
    default_args={
        "retries": 2,
    },
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )
    