import boto3
import pickle
import os
import pandas as pd

class CourseCompletionModel:
    def __init__(self):
        self.bucket_name = "course-completion-ml-artifacts"
        self.model_key = "artifacts/model.pkl"
        self.local_model_path = "/tmp/model.pkl"
        self.model = None

    def load_model(self):
        # Skip model loading in CI environment
        if os.getenv("CI") == "true":
            return

        if self.model is not None:
            return

        # FIXED: Added region_name to match your infrastructure
        s3 = boto3.client("s3", region_name="eu-north-1")

        try:
            if not os.path.exists(self.local_model_path):
                print(f"Downloading model from {self.bucket_name}...")
                s3.download_file(
                    self.bucket_name,
                    self.model_key,
                    self.local_model_path
                )

            with open(self.local_model_path, "rb") as f:
                self.model = pickle.load(f)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, input_data: dict):
        if self.model is None:
            self.load_model()
        
        df = pd.DataFrame([input_data])
        # Ensure the columns in df match what the model saw during training
        return int(self.model.predict(df)[0])