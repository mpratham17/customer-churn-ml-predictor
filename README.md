# Customer Churn Prediction

A machine learning project that predicts customer churn using customer profile, subscription, usage, engagement, and satisfaction data. The project includes exploratory data analysis, data preprocessing, model evaluation, classification threshold optimization, and an interactive Streamlit application.

## Overview

Customer churn occurs when customers stop using a company's services. Identifying customers who are at risk of leaving can help businesses prioritize retention efforts and make data-driven decisions.

This project builds a binary classification model to predict whether a customer will churn within the next 30 days.

## Objectives

- Explore customer data and identify patterns associated with churn.
- Prepare numerical, binary, and categorical features for machine learning.
- Train and evaluate multiple classification models.
- Address class imbalance through evaluation and decision threshold selection.
- Build an interactive web interface for generating churn predictions.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Persistence:** Pickle
- **Development Environment:** Google Colab / Jupyter Notebook

## Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   Examined the target distribution, customer characteristics, and relationships between features and churn.

2. **Data Preprocessing**  
   - Median imputation and standardization for numerical features.
   - Most-frequent imputation for binary features.
   - Most-frequent imputation and one-hot encoding for categorical features.
   - Stratified training and test split.

3. **Model Training**  
   Evaluated Logistic Regression, Decision Tree, and Random Forest classifiers.

4. **Model Evaluation**  
   Used confusion matrices, classification reports, ROC-AUC, and Precision-Recall AUC to evaluate performance.

5. **Classification Threshold Optimization**  
   Evaluated multiple decision thresholds using validation data and selected a threshold of `0.25` based on the highest F1-score among the tested values.

6. **Final Evaluation**  
   Evaluated the selected threshold on an independent test set.

7. **Interactive Application**  
   Built a Streamlit interface that accepts customer details and displays the predicted churn probability and classification.

## Dataset

- Source: https://www.kaggle.com/datasets/mrityu98/customer-churn-telecomott

The dataset contains 34,000 customer records and the target variable `churn_next_30d`.

- **Target:** `churn_next_30d`
- `0`: Customer does not churn within the target period.
- `1`: Customer churns within the next 30 days.

The target distribution is approximately 88.49% non-churn and 11.51% churn.

The model uses customer tenure, monthly fee, recent usage, support tickets, competing offers, engagement score, customer satisfaction, estimated lifetime value, billing and discount indicators, plan type, region, device type, customer segment, and preferred contact channel.

## Model Results

The final Logistic Regression model was evaluated using a decision threshold of `0.25`.

| Metric | Test Result |
|---|---:|
| Accuracy | 86.15% |
| Churn Precision | 41.07% |
| Churn Recall | 46.74% |
| Churn F1-score | 43.88% |
| ROC-AUC | 0.8285 |
| PR-AUC (Average Precision) | 0.4337 |

The lower decision threshold improves churn detection compared with the default threshold of `0.50`, at the cost of more false-positive predictions.

These results are specific to the dataset and evaluation split used in this project. Predictions should be interpreted as estimates rather than guarantees of future customer behavior.

## Streamlit Application

The application allows users to enter customer details and obtain:

- Estimated churn probability.
- Predicted churn-risk classification.
- Classification threshold used by the model.

The saved model, preprocessing pipeline, and threshold are loaded from `models/churn_model.pkl`.

## Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.


### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

**Compatibility note:** The saved pickle artifact was created with scikit-learn `1.6.1`. Use a compatible scikit-learn environment when loading it.

## Repository Structure

```text
Customer-Churn-Prediction/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── churn_model.pkl
└── notebooks/
    ├── 01_eda.ipynb
    └── 02_preprocessing.ipynb
|
└──data
    └──raw_customer_churn.csv
    └──cleaned_customer_churn.csv
```

## Author

**Pratham**

