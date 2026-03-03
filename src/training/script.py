import pandas as pd
import boto3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BUCKET_NAME = "course-completion-ml-artifacts"
DATA_KEY = "data/online_course_completion(3).csv"
MODEL_KEY = "artifacts/model.pkl"

def train_and_upload():
    s3 = boto3.client("s3", region_name="eu-north-1")

    data_path = "/tmp/data.csv"
    model_path = "/tmp/model.pkl"

    s3.download_file(BUCKET_NAME, DATA_KEY, data_path)
    df = pd.read_csv(data_path)

    df.drop_duplicates(inplace=True)
    
    df.drop(
        [
            'height_cm', 'weight_kg', 'num_siblings', 'has_pet',
            'favorite_color', 'birth_month'
        ],
        axis=1,
        inplace=True,
        errors="ignore"
    )
    df.dropna(inplace=True)

    dummies = pd.get_dummies(df["preferred_device"], drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    X = df.drop(["completed_course", "preferred_device"], axis=1)
    y = df["completed_course"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy:", acc)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    s3.upload_file(model_path, BUCKET_NAME, MODEL_KEY)
    return acc
