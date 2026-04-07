import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.model_class import CourseCompletionModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

model = CourseCompletionModel(skip_loading=(os.getenv("CI") == "true"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model()
    yield

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

class PredictionInput(BaseModel):
    age: int
    hours_per_week: int
    assignments_submitted: int
    desktop: int # User sends this, but model_class filters it
    mobile: int
    pager: int
    smart_tv: int
    tablet: int

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict")
def predict_course(data: PredictionInput):
    result = model.predict(data.model_dump())
    return {"prediction": "Completed" if result == 1 else "Not Completed"}