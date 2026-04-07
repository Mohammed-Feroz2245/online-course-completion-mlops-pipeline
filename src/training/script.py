import pandas as pd
import boto3
import pickle
import os
import tempfile
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
BUCKET_NAME = os.getenv("S3_BUCKET", "course-completion-ml-artifacts")
DATA_KEY = "data/online_course_completion(3).csv"
MODEL_KEY = "artifacts/model.pkl"

def fetch_data_and_preprocess(s3_client, local_data_path):
    """
    Downloads raw data from S3, cleans it, and prepares features.
    """
    print(f"📥 Downloading data from S3: s3://{BUCKET_NAME}/{DATA_KEY}")
    s3_client.download_file(BUCKET_NAME, DATA_KEY, local_data_path)
    
    df = pd.read_csv(local_data_path)
    df.drop_duplicates(inplace=True)

    # Filter to numeric columns + target + categorical feature
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_keep = list(set(numeric_cols + ['preferred_device', 'completed_course']))
    df = df[cols_to_keep].dropna()

    # One-Hot Encoding: preferred_device
    # Note: 'drop_first=True' drops 'Desktop' to avoid the Dummy Variable Trap.
    # This matches the mapping logic in model_class.py.
    dummies = pd.get_dummies(df["preferred_device"], drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    # Define Features (X) and Target (y)
    X = df.drop(["completed_course", "preferred_device"], axis=1)
    y = df["completed_course"]
    
    print(f"✅ Preprocessing complete. Features: {list(X.columns)}")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_upload():
    """
    Main pipeline: Fetch data, Tune Hyperparameters, Log to MLflow, and Upload to S3.
    """
    mlflow.set_experiment("Course_Completion_Training")
    s3 = boto3.client("s3", region_name="eu-north-1")
    
    # Use temporary directory for local file handling
    tmp_dir = tempfile.gettempdir()
    data_path = os.path.join(tmp_dir, "raw_data.csv")
    model_path = os.path.join(tmp_dir, "model.pkl")

    with mlflow.start_run():
        try:
            # 1. Data Phase
            X_train, X_test, y_train, y_test = fetch_data_and_preprocess(s3, data_path)
        except Exception as e:
            print(f"❌ Error during data phase: {e}")
            return None

        # 2. Hyperparameter Tuning
        print("🚀 Starting Hyperparameter Tuning...")
        rf = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        # Optimized search (n_iter=2 for speed, increase for better accuracy)
        random_search = RandomizedSearchCV(
            estimator=rf, 
            param_distributions=param_dist, 
            n_iter=2, 
            cv=3, 
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # 3. Evaluation and Logging
        predictions = best_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        print(f"📊 Model Accuracy: {acc:.4f}")
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(best_model, "random-forest-model")

        # 4. Persistence (S3 Upload)
        print(f"📤 Uploading model to S3: s3://{BUCKET_NAME}/{MODEL_KEY}")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        
        s3.upload_file(model_path, BUCKET_NAME, MODEL_KEY)
        print("✨ Pipeline Finished Successfully!")
        
        return acc

if __name__ == "__main__":
    train_and_upload()