# Customer Churn Prediction

## Demo 

▶️ **[Watch the Customer Churn Prediction Project Demo](./demo/Demo.mp4)**

## 1. Project Overview

This project builds an end-to-end **Customer Churn Prediction** solution for a telecom company using the IBM Telco Customer Churn dataset.

The objective is to identify customers who are more likely to leave the telecom service so that the business can take proactive retention actions.

The project covers:

**Business Problem → Data Understanding → Data Preparation → EDA → Feature Engineering → Model Training → Evaluation → Interpretation → Model Saving → REST API**

---


## 2. Project Structure

```text
customer_churn_project/
│
├── data/
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── model/
│   └── churn_model.pkl
│
├── app.py
├── sample_request.json
├── requirements.txt
└── README.md
```

> The dataset was worked with in the notebook. The README does not assume that a separate CSV file is present in the `data/` directory.

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Jupyter Notebook

---

# 4. Data Understanding & Preparation

## Dataset

The project uses the **IBM Telco Customer Churn Dataset**.

The target variable is:

```text
Churn
```

where:

- `Yes` = customer churned
- `No` = customer stayed

### Original Features

The dataset contains customer demographic, service, contract, billing, and usage-related information such as:

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

`customerID` was removed before model training because it uniquely identifies a customer and does not represent customer behaviour.

---

## Missing-Value Analysis

A standard missing-value check was performed using:

```python
df.isnull().sum()
```

The dataset did not contain standard missing values across the columns.

However, `TotalCharges` was initially stored as a string. After converting it to numeric using `errors="coerce"`, **11 invalid values** were identified.

All 11 invalid values belonged to customers with:

- `tenure = 0`
- `Churn = No`

These values were therefore converted to `0` for `TotalCharges`.

This decision is appropriate for the dataset because customers with zero tenure have no accumulated tenure-based charges.

---

## Duplicate Analysis

Duplicate records were checked using:

```python
df.duplicated().sum()
```

Result:

```text
0
```

Therefore, no duplicate rows were removed.

---

## Data Types

The dataset contained both numerical and categorical features.

### Numerical Features

Before feature engineering:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

After feature engineering:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `AverageMonthlySpend`
- `IsNewCustomer`

### Categorical Features

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

---

# 5. Target Variable Analysis

The target distribution was:

| Churn | Count | Percentage |
|---|---:|---:|
| No | 5174 | 73.46% |
| Yes | 1869 | 26.54% |

Approximately **26.5% of customers churned**, while approximately **73.5% stayed**.

This indicates a moderately imbalanced target variable, with non-churn customers forming the majority.

The class distribution should therefore be considered when interpreting model performance, particularly recall and precision.

---

# 6. Exploratory Data Analysis

At least five meaningful visualizations were created.

## Visualization 1 — Churn Distribution

The churn distribution shows that most customers remained with the company, while approximately one quarter churned.

### Business Insight

The company is dealing with a meaningful churn population. Even though churn is the minority class, identifying these customers can provide opportunities for proactive retention.

---

## Visualization 2 — Churn by Contract Type

The analysis shows a substantially higher number of churned customers among customers on **month-to-month contracts** compared with customers on longer-term contracts.

### Business Insight

Contract type is strongly associated with churn in this dataset. Customers on month-to-month contracts represent an important group for retention analysis.

---

## Visualization 3 — Customer Tenure by Churn

Customers who churned generally had shorter tenure than customers who stayed.

The median tenure for churned customers was approximately **10 months**, compared with approximately **38 months** for customers who stayed.

### Business Insight

Newer customers appear to represent an important churn-risk segment. Early customer engagement and retention strategies may therefore be relevant.

---

## Visualization 4 — Monthly Charges by Churn

Customers who churned generally had higher monthly charges.

The median monthly charge was approximately:

- Churned customers: **$80**
- Customers who stayed: **$64**

There is considerable overlap between the groups, so monthly charges alone do not explain churn.

### Business Insight

Pricing or service-cost characteristics may be associated with churn, but they should be considered together with contract, tenure, and service characteristics.

---

## Visualization 5 — Churn by Internet Service

Customers using **fiber optic internet** showed a relatively high number of churn cases, while customers without internet service showed relatively few churn cases.

### Business Insight

Internet service type is associated with churn in the dataset. This feature can help the model distinguish different customer/service segments.

> These EDA findings describe associations in the dataset and should not be interpreted as proof that a particular feature directly causes churn.

---

# 7. Feature Engineering

Two additional features were created.

## Feature 1 — AverageMonthlySpend

```text
AverageMonthlySpend = TotalCharges / tenure
```

Because customers with zero tenure could cause division by zero, zero tenure was temporarily replaced with `1` during the calculation.

### Why it was created

`MonthlyCharges` represents the current monthly charge, while `TotalCharges` represents accumulated charges.

Average monthly spend provides an additional representation of the customer's historical spending level.

---

## Feature 2 — IsNewCustomer

```text
IsNewCustomer = 1 if tenure <= 12 else 0
```

### Why it was created

Customers with shorter tenure showed higher churn levels during EDA.

This binary feature explicitly identifies customers who have been with the company for one year or less, allowing the model to use this customer lifecycle information directly.

---

# 8. Train/Test Split

The target was separated from the input features.

The data was split into:

- **70% training data**
- **30% testing data**

The split used:

```text
random_state = 42
```

and stratification using the target variable.

Stratification helps maintain a similar churn/non-churn distribution in both training and testing datasets.

The resulting shapes were:

```text
X shape: (7043, 20)
y shape: (7043,)
```

---

# 9. Data Preprocessing

A `ColumnTransformer` was used to apply preprocessing to numerical and categorical features.

Numerical features were passed through without scaling.

Categorical features were transformed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

This converts categorical values into numerical indicator columns.

`handle_unknown="ignore"` ensures that an unseen categorical value in future data does not cause the preprocessing step to fail.

---

# 10. Preventing Data Leakage

The preprocessing and model were combined into a single Scikit-learn `Pipeline`.

Conceptually:

```text
Raw Customer Data
       ↓
Preprocessing
       ↓
One-Hot Encoding
       ↓
Decision Tree
       ↓
Prediction
```

This approach ensures that the same preprocessing logic is applied during:

- training
- testing
- future API predictions

The preprocessing is fitted as part of the training pipeline rather than separately using the complete dataset.

This helps avoid data leakage and ensures consistent treatment of unseen customer data.

---

# 11. Model Development

The assignment required a `DecisionTreeClassifier` with at least two configurations.

Two configurations were evaluated.

## Configuration 1 — Default Decision Tree

```text
DecisionTreeClassifier(random_state=42)
```

Results:

| Metric | Score |
|---|---:|
| Accuracy | 72.12% |
| Precision | 47.50% |
| Recall | 47.42% |
| F1 Score | 47.46% |

Confusion Matrix:

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 1258 | 294 |
| Yes | 295 | 266 |

---

## Configuration 2 — Limited Tree Depth

```text
DecisionTreeClassifier(max_depth=5, random_state=42)
```

Results:

| Metric | Score |
|---|---:|
| Accuracy | 79.37% |
| Precision | 61.38% |
| Recall | 60.07% |
| F1 Score | 60.72% |

Confusion Matrix:

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 1340 | 212 |
| Yes | 224 | 337 |

---

## Model Comparison

Both models were evaluated on the same test dataset.

The `max_depth=5` configuration produced higher:

- Accuracy
- Precision
- Recall
- F1 Score

Therefore, the `max_depth=5` Decision Tree was selected as the final model used by the API.

Limiting tree depth also produces a simpler model than an unrestricted Decision Tree, making the resulting decision structure easier to inspect.

---

# 12. Model Evaluation

The final model achieved:

| Metric | Final Score |
|---|---:|
| Accuracy | **79.37%** |
| Precision | **61.38%** |
| Recall | **60.07%** |
| F1 Score | **60.72%** |

## Confusion Matrix

For the final model:

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 1340 | 212 |
| Yes | 224 | 337 |

This means:

- **1340** customers who stayed were correctly predicted as `No`.
- **212** customers who stayed were incorrectly predicted as `Yes`.
- **224** customers who churned were incorrectly predicted as `No`.
- **337** customers who churned were correctly predicted as `Yes`.

---

# 13. Precision vs Recall — Business Interpretation

For a telecom company trying to identify customers who may churn, **recall is particularly important** because it measures how many actual churners are successfully identified.

A false negative occurs when a customer who actually churns is predicted as a non-churner. This may result in a lost opportunity for proactive retention.

However, precision is also important because retention campaigns have associated costs.

Therefore, the business should consider the relative cost of:

- False negatives — missing customers who are likely to churn
- False positives — targeting customers who were not actually going to churn

For this churn-identification use case, **recall is an important consideration because the objective is to identify as many potential churners as possible, while precision should also be monitored to control unnecessary retention efforts.**

---

# 14. Model Interpretation

Feature importance was extracted from the final Decision Tree after preprocessing.

The most important transformed features included:

| Feature | Importance |
|---|---:|
| Contract — Month-to-month | 0.510351 |
| tenure | 0.177449 |
| InternetService — Fiber optic | 0.152518 |
| PaymentMethod — Electronic check | 0.031905 |
| TechSupport — No | 0.029829 |
| AverageMonthlySpend | 0.029001 |
| TotalCharges | 0.025496 |
| MonthlyCharges | 0.018583 |
| OnlineSecurity — No | 0.010108 |
| StreamingMovies — No | 0.007738 |

### Key Interpretation

The model relied heavily on:

1. Contract type
2. Customer tenure
3. Internet service type
4. Payment method
5. Technical support/service characteristics
6. Customer spending characteristics

The root split of the final tree was based on:

```text
Contract_Month-to-month
```

Other important decision splits included monthly charges, fiber-optic service, tenure, electronic check payment, technical support, online security, average monthly spend, and total charges.

> Feature importance indicates how much a feature contributes to the tree's splitting decisions. It should not be interpreted as proof that the feature causes churn.

---

# 15. Key Findings

The analysis produced several important observations:

- Approximately **26.5%** of customers in the dataset churned.
- Month-to-month contract customers had substantially more churn.
- Churned customers generally had shorter tenure.
- Churned customers generally had higher monthly charges.
- Fiber optic internet customers showed a relatively high number of churn cases.
- Contract type and tenure were among the strongest features used by the Decision Tree.
- The engineered features `AverageMonthlySpend` and `IsNewCustomer` provided additional customer behaviour/lifecycle information.
- The final Decision Tree achieved approximately **79.37% accuracy** and **60.07% recall** on the test set.

These findings can help a telecom business identify customer segments for further retention analysis.

---

# 16. Model Saving

The complete preprocessing and Decision Tree pipeline was saved using Joblib:

```text
model/churn_model.pkl
```

The saved object contains:

```text
Preprocessing
     +
One-Hot Encoding
     +
Decision Tree
```

This means the API can load a single saved object and apply the same processing used during training.

---

# 17. REST API

A REST API was implemented using **FastAPI**.

The API loads:

```text
model/churn_model.pkl
```

when the application starts.

## Start the API

Run the following command from the project root:

```bash
python -m uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# 18. API Endpoints

## GET /

Used to confirm that the API is running.

Example response:

```json
{
  "message": "Customer Churn Prediction API is running"
}
```

---

## POST /predict

Accepts customer information as JSON and returns:

- churn prediction
- churn probability

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.5837
}
```

---

# 19. API Input Validation

The API uses Pydantic to define the expected request structure.

It validates:

- required fields
- basic data types such as integers, floats, and strings

Invalid or missing fields are rejected by FastAPI/Pydantic with an appropriate validation response.

The API currently expects the engineered features:

```text
AverageMonthlySpend
IsNewCustomer
```

as part of the request payload, matching the current API implementation.

---

# 20. Sample API Request

A sample request is available in:

```text
sample_request.json
```

Example:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85,
  "AverageMonthlySpend": 29.85,
  "IsNewCustomer": 1
}
```

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.5837
}
```

---

# 21. Example Second Prediction

A longer-tenure customer with a one-year contract was also tested.

Example response:

```json
{
  "prediction": "No",
  "churn_probability": 0.0107
}
```

This demonstrates how the API can return both the predicted class and the model's estimated probability for churn.

---

# 22. How to Run the Project

## Step 1 — Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

## Step 2 — Run the notebook

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebook/churn_analysis.ipynb
```

Run the notebook from top to bottom to reproduce the analysis and model training.

## Step 3 — Ensure the model exists

After running the model training/saving section, verify:

```text
model/churn_model.pkl
```

exists.

## Step 4 — Start the API

From the project root:

```bash
python -m uvicorn app:app --reload
```

## Step 5 — Test the API

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint and provide the sample JSON payload.

---


# 23. Conclusion

This project demonstrates an end-to-end machine learning workflow for telecom customer churn prediction.

The final solution:

1. Understands and cleans the customer data.
2. Performs exploratory analysis to identify churn-related patterns.
3. Creates additional customer behaviour and lifecycle features.
4. Applies consistent preprocessing using a Scikit-learn pipeline.
5. Trains and compares two Decision Tree configurations.
6. Evaluates the selected model using accuracy, precision, recall, F1 score, and a confusion matrix.
7. Interprets the model using feature importance and tree structure.
8. Saves the complete preprocessing/model pipeline.
9. Exposes the model through a FastAPI REST endpoint.

The final model achieved **79.37% accuracy, 61.38% precision, 60.07% recall, and 60.72% F1 score** on the 30% held-out test set.
