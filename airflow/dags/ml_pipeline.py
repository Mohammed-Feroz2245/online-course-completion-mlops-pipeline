from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Ensure the src directory is in the python path
sys.path.append("/opt/project")

def run_training():
    # This imports the function we just updated in Step 1
    from src.training.script import train_and_upload
    return train_and_upload()

with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )