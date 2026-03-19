from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Input schema
class MessageInput(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {"message": "FastAPI Spam Detection API is running!"}

# Prediction route
@app.post("/predict")
def predict(data: MessageInput):
    transformed = vectorizer.transform([data.text])

    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0]

    spam_prob = float(probability[1])

    result = "Spam" if prediction == 1 else "Not Spam"

    return {
        "prediction": result,
        "spam_probability": round(spam_prob, 4)
    }