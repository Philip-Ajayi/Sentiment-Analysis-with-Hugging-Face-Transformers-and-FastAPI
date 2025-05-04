from fastapi import FastAPI
from pydantic import BaseModel
from app.model import SentimentModel

app = FastAPI()
model = SentimentModel()

class Review(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(review: Review):
    prediction = model.predict(review.text)
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}
