import pandas as pd
import boto3
import pickle
import os
import tempfile
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
BUCKET_NAME = os.getenv("S3_BUCKET", "course-completion-ml-artifacts")
DATA_KEY = "data/online_course_completion(3).csv"
MODEL_KEY = "artifacts/model.pkl"

# This list defines the exact final feature order (very important!)
FEATURES_LIST = ['age', 'hours_per_week', 'assignments_submitted', 
                 'Desktop', 'Mobile', 'Pager', 'Smart TV', 'Tablet']

def fetch_data_and_preprocess(s3_client, local_data_path):
    """Download data from S3 and prepare features for training"""
    try:
        s3_client.download_file(BUCKET_NAME, DATA_KEY, local_data_path)
        print(f"✅ Downloaded data from S3: s3://{BUCKET_NAME}/{DATA_KEY}")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        raise

    df = pd.read_csv(local_data_path).drop_duplicates()
    df = df.dropna()

    # Target
    y = df["completed_course"]
    
    # One-hot encoding for preferred_device (includes Desktop now)
    dummies = pd.get_dummies(df["preferred_device"], drop_first=False) 
    
    # Combine numeric features + dummies
    X = df[['age', 'hours_per_week', 'assignments_submitted']].join(dummies)
    
    # Ensure exact column order and fill missing with 0 (safety)
    X = X.reindex(columns=FEATURES_LIST, fill_value=0)
    
    print(f"✅ Preprocessing complete. Final features: {list(X.columns)}")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_upload():
    """Main function: train model and upload to S3"""
    mlflow.set_experiment("Course_Completion_Training")
    
    s3 = boto3.client("s3", region_name="eu-north-1")
    
    tmp_dir = tempfile.gettempdir()
    data_path = os.path.join(tmp_dir, "data.csv")
    model_path = os.path.join(tmp_dir, "model.pkl")

    with mlflow.start_run():
        try:
            X_train, X_test, y_train, y_test = fetch_data_and_preprocess(s3, data_path)
        except Exception as e:
            print(f"❌ Error during data phase: {e}")
            return None

        # XGBoost model (usually gives better accuracy than RandomForest)
        xgb = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)

        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        random_search = RandomizedSearchCV(
            xgb, param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Calculate both train and test accuracy
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        test_acc = accuracy_score(y_test, best_model.predict(X_test))
        
        print(f"📊 Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

        # Log to MLflow
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        # Save model locally then upload to S3
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        
        s3.upload_file(model_path, BUCKET_NAME, MODEL_KEY)
        
        print("✨ Pipeline Finished Successfully!")
        return test_acc


if __name__ == "__main__":
    train_and_upload()