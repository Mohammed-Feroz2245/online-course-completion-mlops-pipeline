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

BUCKET_NAME = "course-completion-ml-artifacts"
DATA_KEY = "data/online_course_completion(3).csv"
MODEL_KEY = "artifacts/model.pkl"

def train_and_upload():
    # 1. Initialize MLflow Experiment
    mlflow.set_experiment("Course_Completion_Tuning")

    # Initialize S3 client
    s3 = boto3.client("s3", region_name="eu-north-1")
    
    tmp_dir = tempfile.gettempdir()
    data_path = os.path.join(tmp_dir, "data.csv")
    model_path = os.path.join(tmp_dir, "model.pkl")

    with mlflow.start_run():
        print(f"Downloading data from S3 to {data_path}...")
        try:
            s3.download_file(BUCKET_NAME, DATA_KEY, data_path)
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return None

        # Load and clean data
        df = pd.read_csv(data_path)
        df.drop_duplicates(inplace=True)

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols_to_keep = list(set(numeric_cols + ['preferred_device', 'completed_course']))
        df = df[cols_to_keep]
        df.dropna(inplace=True)

        dummies = pd.get_dummies(df["preferred_device"], drop_first=True)
        df = pd.concat([df, dummies], axis=1)

        X = df.drop(["completed_course", "preferred_device"], axis=1)
        y = df["completed_course"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- HYPERPARAMETER TUNING ---
        rf = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }

        random_search = RandomizedSearchCV(
            estimator=rf, 
            param_distributions=param_dist, 
            n_iter=3,   # Reduced for speed as per your request
            cv=5, 
            scoring='accuracy', 
            n_jobs=1,      
            random_state=42,
            verbose=2     
        )

        print("Searching for the best model parameters (Cross-Validation)...")
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        
        # 2. Log Parameters & Metrics to MLflow
        mlflow.log_params(random_search.best_params_)
        acc = accuracy_score(y_test, best_model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        
        print(f"Best Parameters Found: {random_search.best_params_}")
        print(f"Final Accuracy on Test Set: {acc}")

        # 3. Log the Model to MLflow Tracking
        mlflow.sklearn.log_model(best_model, "random-forest-model")

        # Save locally and Upload to S3 (Your existing production path)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Uploading model to S3: {MODEL_KEY}")
        s3.upload_file(model_path, BUCKET_NAME, MODEL_KEY)
        
        print("Process complete!")
        return acc

if __name__ == "__main__":
    train_and_upload()