from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch
import pytest

client = TestClient(app)

def test_read_main():
    """Test the home endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}

def test_predict_endpoint_structure():
    """Test if the predict endpoint accepts correct input structure using a Mock"""
    payload = {
        "age": 25,
        "hours_per_week": 10,
        "assignments_submitted": 5,
        "desktop": 1,
        "mobile": 0,
        "pager": 0,
        "smart_tv": 0,
        "tablet": 0
    }
    
    # We "patch" the predict method so it doesn't actually run the model
    with patch("src.model_class.CourseCompletionModel.predict") as mock_predict:
        mock_predict.return_value = 1  # Pretend the model predicted 'Completed'
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"prediction": "Completed"}
        # Verify the mock was actually called
        mock_predict.assert_called_once()