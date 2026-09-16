# Customer Churn Prediction

## 1. Project Overview

This project predicts whether a telecom customer is likely to churn using the IBM Telco Customer Churn dataset.

A Decision Tree Classifier is used to predict:

- `Yes` → Customer is likely to churn
- `No` → Customer is likely to stay

The trained model is exposed through a REST API using FastAPI.

---

## 2. Project Structure

```text
customer_churn_project/
│
├── data/
│   └── telco_customer_churn.csv
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── model/
│   └── churn_model.pkl
│
├── app.py
├── requirements.txt
├── sample_request.json
└── README.md
```

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Jupyter Notebook

---

## 4. Machine Learning Workflow

The project follows these steps:

1. Data loading and understanding
2. Missing-value analysis
3. Duplicate analysis
4. Data cleaning
5. Numerical and categorical feature identification
6. Exploratory Data Analysis
7. Feature engineering
8. Train/test split
9. Categorical encoding
10. Decision Tree model training
11. Model evaluation
12. Feature importance analysis
13. Decision Tree visualization
14. Model serialization
15. REST API deployment

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

- `customerID` was removed because it is an identifier and does not provide useful predictive information.
- `TotalCharges` was converted from string to numeric.
- Invalid `TotalCharges` values were handled as zero because they corresponded to customers with zero tenure.
- Categorical features were converted using OneHotEncoder.
- `handle_unknown="ignore"` was used so unseen categorical values do not cause the preprocessing pipeline to fail.
- The target variable is `Churn`.

The dataset was divided into:

- 70% training data
- 30% testing data

The split used:

```text
random_state = 42
stratify = y
```

---

## 6. Exploratory Data Analysis

Five major visualizations were created:

1. Customer churn distribution
2. Churn by contract type
3. Customer tenure by churn
4. Monthly charges by churn
5. Churn by internet service

### Key observations

- Approximately 26.5% of customers churned.
- Month-to-month contract customers showed substantially higher churn counts.
- Churned customers generally had shorter tenure.
- Churned customers tended to have higher monthly charges.
- Fiber-optic customers showed relatively high churn counts.

---

## 7. Feature Engineering

Two additional features were created.

### AverageMonthlySpend

Calculated as:

```text
TotalCharges / tenure
```

For customers with zero tenure, zero tenure was replaced with 1 during calculation to avoid division by zero.

### IsNewCustomer

A binary feature was created:

```text
1 → tenure <= 12 months
0 → tenure > 12 months
```

This identifies relatively new customers.

---

## 8. Model Development

Two Decision Tree configurations were evaluated.

### Configuration 1

Default Decision Tree:

```text
DecisionTreeClassifier(random_state=42)
```

Results:

| Metric | Result |
|---|---:|
| Accuracy | 72.12% |
| Precision | 47.50% |
| Recall | 47.42% |
| F1-score | 47.46% |

### Configuration 2

Decision Tree with maximum depth of 5:

```text
DecisionTreeClassifier(max_depth=5, random_state=42)
```

Results:

| Metric | Result |
|---|---:|
| Accuracy | 79.37% |
| Precision | 61.38% |
| Recall | 60.07% |
| F1-score | 60.72% |

Configuration 2 was selected as the final model based on its stronger test-set performance across all four evaluated metrics.

---

## 9. Feature Importance

The most important transformed features included:

1. Contract — Month-to-month
2. Tenure
3. Internet Service — Fiber optic
4. Payment Method — Electronic check
5. Tech Support — No
6. Average Monthly Spend
7. Total Charges
8. Monthly Charges

Feature importance represents the contribution of features to the Decision Tree's splitting decisions. It does not by itself establish that a feature causes churn.

---

## 10. API

The project provides a REST API using FastAPI.

### Start the API

From the project directory:

```bash
python -m uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### API Documentation

FastAPI provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## 11. Prediction Endpoint

### Endpoint

```text
POST /predict
```

The endpoint accepts customer information as JSON and returns:

- Churn prediction
- Churn probability

### Example Response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.5837
}
```

The probability represents the model's estimated probability that the customer belongs to the churn (`Yes`) class.

---

## 12. Input Validation

The API uses Pydantic to validate incoming request data.

Invalid data types or missing required fields are rejected by the API before the request reaches the machine-learning model.

---

## 13. Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use `POST /predict` to test a prediction.

---

## 14. Model File

The final trained pipeline is saved as:

```text
model/churn_model.pkl
```

The saved pipeline contains:

- Feature preprocessing
- One-hot encoding
- Final Decision Tree model

This allows the API to apply the same preprocessing used during model training.

---

## 15. Limitations

- The model is trained on a single telecom customer churn dataset.
- Decision Tree performance may vary on other datasets or customer populations.
- Feature importance shows model contribution, not causation.
- The current model uses a fixed decision threshold for classification.
- The API expects the engineered features `AverageMonthlySpend` and `IsNewCustomer` as inputs.
- Further model tuning and validation could potentially improve performance.
