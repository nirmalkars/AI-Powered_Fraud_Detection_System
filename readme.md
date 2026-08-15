# PyTorch Fraud Detection System

A machine learning application that uses **PyTorch** to detect potentially fraudulent credit card transactions. The system performs data preprocessing, handles class imbalance, trains a neural network, evaluates model performance, and exposes fraud prediction through a **FastAPI REST API**.

The project is designed as a practical **AI/ML portfolio project** demonstrating deep learning, imbalanced classification, model evaluation, and API deployment.

---

## 1. Project Overview

Fraud detection is a binary classification problem where the goal is to determine whether a financial transaction is:

* **Normal**
* **Fraudulent**

One of the main challenges is **class imbalance**, because fraudulent transactions typically represent a very small percentage of all transactions.

This project addresses the problem using a **PyTorch feed-forward neural network** with class-weighted binary cross-entropy loss.

---

## 2. Key Features

* PyTorch-based fraud classification model
* Binary transaction classification
* Handling of highly imbalanced data
* StandardScaler-based feature normalization
* Class-weighted `BCEWithLogitsLoss`
* Adam optimizer
* Dropout and Batch Normalization
* Precision, Recall, F1-score evaluation
* ROC-AUC evaluation
* Confusion matrix
* Saved PyTorch model
* Saved preprocessing scaler
* FastAPI REST API
* Interactive Swagger API documentation
* Local model inference
* No OpenAI or paid API required

---

## 3. Technology Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Application development        |
| PyTorch      | Deep learning model            |
| Pandas       | Data processing                |
| NumPy        | Numerical computation          |
| Scikit-learn | Preprocessing and evaluation   |
| FastAPI      | REST API                       |
| Pydantic     | Request validation             |
| Joblib       | Saving preprocessing artifacts |
| Pytest       | Unit testing                   |

---

## 4. Architecture

```text
                  Transaction Dataset
                         |
                         v
                Data Preprocessing
                         |
                         v
                  Train/Test Split
                         |
                         v
                   StandardScaler
                         |
                         v
                  PyTorch Dataset
                         |
                         v
                    DataLoader
                         |
                         v
                FraudDetector Model
                         |
                         v
             BCEWithLogitsLoss
                         |
                         v
                   Backpropagation
                         |
                         v
                      Adam
                         |
                         v
                    Evaluation
                         |
                +--------+--------+
                |                 |
                v                 v
             Metrics          Model
                                  |
                                  v
                             FastAPI
                                  |
                                  v
                         Fraud Prediction
```

---

## 5. Project Structure

```text
fraud-detection-pytorch/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── fraud_detector.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── transaction.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│
├── models/
│   ├── fraud_detector.pt
│   ├── scaler.pkl
│   └── feature_names.json
│
├── training/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── __init__.py
│   └── test_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Dataset

This project expects a credit card transaction dataset with the following structure:

```text
Time
V1
V2
...
V28
Amount
Class
```

The `Class` column represents the target:

```text
0 → Normal transaction
1 → Fraudulent transaction
```

Place the dataset at:

```text
data/raw/creditcard.csv
```

### Important

The dataset is intentionally not included in the Git repository.

Do not commit sensitive financial data or datasets whose license does not permit redistribution.

---

## 7. Machine Learning Pipeline

The complete ML pipeline is:

```text
Raw Transaction Data
        |
        v
Data Validation
        |
        v
Feature / Target Separation
        |
        v
Train/Test Split
        |
        v
Feature Scaling
        |
        v
PyTorch Dataset
        |
        v
DataLoader
        |
        v
Neural Network Training
        |
        v
Model Evaluation
        |
        v
Model Serialization
```

---

## 8. Data Preprocessing

The dataset is separated into:

```text
Features → Time, V1...V28, Amount

Target → Class
```

The feature scaler is fitted only on the training data to avoid data leakage.

```python
scaler.fit(X_train)
```

The same scaler is then used for:

```text
Training data
Test data
API inference
```

This ensures that the model receives data in the same representation during training and prediction.

---

## 9. Handling Class Imbalance

Fraud detection datasets are usually highly imbalanced.

For example:

```text
Normal transactions → Majority
Fraud transactions  → Minority
```

Simply optimizing for accuracy can produce a misleading model.

Therefore, this project calculates a positive-class weight:

```text
positive_weight =
number_of_normal_transactions /
number_of_fraud_transactions
```

The weight is passed to:

```python
BCEWithLogitsLoss(
    pos_weight=pos_weight
)
```

This increases the penalty for incorrectly classifying fraudulent transactions.

---

## 10. PyTorch Model

The fraud detection model is a feed-forward neural network.

Architecture:

```text
Input
  |
  v
Linear
  |
  v
ReLU
  |
  v
BatchNorm
  |
  v
Dropout
  |
  v
Linear
  |
  v
ReLU
  |
  v
BatchNorm
  |
  v
Dropout
  |
  v
Linear
  |
  v
ReLU
  |
  v
Linear
  |
  v
Fraud Logit
```

The model uses:

* ReLU activation
* Batch Normalization
* Dropout
* Linear layers

The final layer produces a logit rather than applying sigmoid directly.

Sigmoid is applied during prediction:

```python
probability = torch.sigmoid(logit)
```

---

## 11. Loss Function

The project uses:

```python
BCEWithLogitsLoss
```

instead of manually applying sigmoid followed by binary cross-entropy.

This combines the sigmoid operation and binary cross-entropy calculation in a numerically stable implementation.

Class imbalance is handled using:

```python
pos_weight
```

---

## 12. Optimizer

The model uses the Adam optimizer:

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam was selected because it provides adaptive learning rates and generally works well for neural-network optimization.

---

## 13. Model Training

Activate your virtual environment.

### Windows

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Make sure the dataset exists:

```text
data/raw/creditcard.csv
```

Run training:

```powershell
python -m training.train
```

During training, the application displays:

```text
Epoch 1/20 | Loss: ...
Epoch 2/20 | Loss: ...
...
Epoch 20/20 | Loss: ...
```

After training, the following files are created:

```text
models/
├── fraud_detector.pt
├── scaler.pkl
└── feature_names.json
```

---

## 14. Model Evaluation

The training process evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Example:

```text
Classification Report:

              precision    recall    f1-score

Normal          ...
Fraud           ...

ROC-AUC: ...
```

### Why not only accuracy?

In fraud detection, a model could achieve high accuracy by predicting almost every transaction as normal.

Therefore, Precision and Recall are more informative.

---

## 15. Evaluation Metrics

### Precision

Precision answers:

> Of the transactions predicted as fraud, how many were actually fraudulent?

```text
Precision =
True Positives /
(True Positives + False Positives)
```

### Recall

Recall answers:

> Of all actual fraudulent transactions, how many did the model detect?

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

### F1-score

F1-score balances Precision and Recall:

```text
F1 =
2 × Precision × Recall /
(Precision + Recall)
```

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between fraudulent and normal transactions across classification thresholds.

---

## 16. Start FastAPI

After training the model:

```powershell
uvicorn app.main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

---

## 17. API Documentation

FastAPI provides interactive Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the fraud prediction endpoint directly from the browser.

---

## 18. API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
    "status": "healthy",
    "service": "PyTorch Fraud Detection API"
}
```

---

### Fraud Prediction

```http
POST /api/v1/predict
```

The endpoint accepts 30 numerical transaction features in the same order used during training.

Example request:

```json
{
    "features": [
        0.0,
        -1.35,
        -0.07,
        2.53,
        1.38,
        -0.33,
        0.46,
        0.24,
        0.10,
        -0.18,
        0.05,
        -0.12,
        0.21,
        -0.31,
        0.14,
        -0.10,
        0.03,
        -0.08,
        0.17,
        -0.05,
        0.11,
        -0.09,
        0.04,
        -0.02,
        0.06,
        -0.01,
        0.03,
        -0.04,
        0.02,
        149.50
    ]
}
```

Example response:

```json
{
    "fraud_probability": 0.924,
    "prediction": "FRAUD",
    "risk_level": "HIGH"
}
```

---

## 19. Risk Classification

The API converts the fraud probability into a simple risk level.

```text
Probability       Risk
-------------------------
< 0.30            LOW
0.30 - 0.69       MEDIUM
>= 0.70           HIGH
```

The classification threshold can be changed depending on the business requirements.

---

## 20. Testing

Run:

```powershell
pytest -v
```

The tests verify core model behavior such as the expected input and output dimensions.

---

## 21. Running the Complete Project

### Step 1

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 2

Install dependencies:

```powershell
pip install -r requirements.txt
```

### Step 3

Place the dataset:

```text
data/raw/creditcard.csv
```

### Step 4

Train the model:

```powershell
python -m training.train
```

### Step 5

Start the API:

```powershell
uvicorn app.main:app --reload
```

### Step 6

Open:

```text
http://127.0.0.1:8000/docs
```

### Step 7

Test:

```text
POST /api/v1/predict
```

---

## 22. Example Workflow

```text
User
 |
 | Transaction Data
 v
FastAPI
 |
 v
Input Validation
 |
 v
Saved StandardScaler
 |
 v
PyTorch Fraud Detector
 |
 v
Sigmoid Probability
 |
 v
Risk Classification
 |
 v
JSON Response
```

Example:

```text
Transaction
     |
     v
Fraud Probability: 0.924
     |
     v
Prediction: FRAUD
     |
     v
Risk Level: HIGH
```

---

## 23. Why PyTorch?

PyTorch was selected because it allows the project to implement:

* Custom neural-network architecture
* Forward propagation
* Automatic differentiation
* Backpropagation
* Optimizer-based training
* GPU acceleration when available
* Model serialization
* Local inference

This makes the project a genuine deep-learning application rather than simply using a pre-trained API.

---

## 24. Why Use a Neural Network?

The transaction dataset contains numerical features that can have complex relationships.

A feed-forward neural network can learn nonlinear relationships between these features and the fraud label.

The model is intentionally kept relatively small so that it is:

* Easy to train
* Easy to deploy
* Easy to explain
* Suitable for a portfolio project

---

### Technologies

```text
Python | PyTorch | Pandas | NumPy | Scikit-learn |
FastAPI | Pydantic | Machine Learning | Deep Learning | REST API
```

---