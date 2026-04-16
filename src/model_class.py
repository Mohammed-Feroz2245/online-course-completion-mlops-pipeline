import os
import boto3
import pickle
import tempfile
import pandas as pd
from pathlib import Path

class CourseCompletionModel:
    def __init__(self, skip_loading: bool = False):
        self.bucket_name = os.getenv("S3_BUCKET", "course-completion-ml-artifacts")
        self.model_key = "artifacts/model.pkl"
        self.local_model_path = Path(tempfile.gettempdir()) / "model.pkl"
        self.model = None
        self.skip_loading = skip_loading 

    def load_model(self):
        if self.skip_loading or self.model is not None:
            return
        try:
            s3 = boto3.client("s3", region_name="eu-north-1")
            if not self.local_model_path.exists():
                s3.download_file(self.bucket_name, self.model_key, str(self.local_model_path))
            with open(self.local_model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✅ Model loaded successfully from S3")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, input_data: dict):
        if self.model is None:
            if self.skip_loading: 
                return 0
            self.load_model()
        
        # MUST exactly match FEATURES_LIST in training script
        expected_order = ['age', 'hours_per_week', 'assignments_submitted', 
                         'Desktop', 'Mobile', 'Pager', 'Smart TV', 'Tablet', 'device_count']
        
        df = pd.DataFrame([{
            'age': input_data['age'],
            'hours_per_week': input_data['hours_per_week'],
            'assignments_submitted': input_data['assignments_submitted'],
            'Desktop': input_data.get('desktop', 0),   # added Desktop (default 0)
            'Mobile': input_data['mobile'],
            'Pager': input_data['pager'],
            'Smart TV': input_data['smart_tv'],
            'Tablet': input_data['tablet'],
            'device_count': (input_data.get('desktop', 0) + input_data['mobile'] + 
                            input_data['pager'] + input_data['smart_tv'] + input_data['tablet'])
        }])

        return int(self.model.predict(df[expected_order])[0])