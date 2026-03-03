from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# Add project path
sys.path.append("/opt/project")

def run_training():
    from src.training.script import train_and_upload
    return train_and_upload()

with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )