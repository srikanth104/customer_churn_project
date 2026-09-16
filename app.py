from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load("model/churn_model.pkl")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AverageMonthlySpend: float
    IsNewCustomer: int


@app.get("/")
def home():
    # Return a simple message to confirm that the API is running.
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    customer_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(customer_data)[0]

    churn_probability = probability[list(model.classes_).index("Yes")]

    return {
        "prediction": prediction,
        "churn_probability": round(float(churn_probability), 4)
    }