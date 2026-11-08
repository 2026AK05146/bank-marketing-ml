# Bank Marketing - Term Deposit Classification

ML Assignment 2 - M.Tech (AIML), BITS Pilani WILP

## a. Problem statement

A Portuguese bank runs phone marketing campaigns to sell term deposits. Calling
every customer is expensive, so the goal is to predict from the client's
details and the campaign history whether a client will subscribe to a term
deposit (`yes`) or not (`no`). This is a binary classification problem. Five
classifiers are trained on the same data, compared on six metrics and shown in
a Streamlit app.

## b. Dataset description

- Source: UCI Machine Learning Repository - Bank Marketing (`bank-full.csv`)
- Instances: 45,211
- Features: 16 (7 numeric, 9 categorical) + target `y`
- Target: `y` - has the client subscribed to a term deposit (yes/no)
- Class balance: about 88% `no` and 12% `yes`, so the data is imbalanced
- Missing values: none (unknown values are stored as the text "unknown")

| Type | Columns |
|------|---------|
| Numeric | age, balance, day, duration, campaign, pdays, previous |
| Categorical | job, marital, education, default, housing, loan, contact, month, poutcome |

Preprocessing: stratified 80/20 train-test split (`random_state=42`), numeric
columns scaled with StandardScaler, categorical columns one-hot encoded. The
preprocessor is fit on the training data only and saved as
`model/preprocessor.pkl`.

## c. GitHub repository link

<PASTE GITHUB REPO LINK HERE>

Live app: <PASTE STREAMLIT APP LINK HERE>

## d. Models used

Comparison on the 20% hold-out test set (9,043 rows, same as `test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall, winner for your dataset?** | |

## Project structure

```
bank-marketing-ml/
|-- app.py               streamlit app
|-- train_models.py      training + evaluation script
|-- requirements.txt
|-- README.md
|-- test_data.csv        hold-out test rows (with target column)
|-- model_results.csv    metrics table produced by train_models.py
|-- data/                original UCI files
|-- model/               saved preprocessor and the 5 trained models
```

## How to run

```
pip install -r requirements.txt
python train_models.py        # trains and saves models in model/
streamlit run app.py
```

## Streamlit app features

- Upload a test csv (use `test_data.csv`)
- Dropdown to pick one of the 5 models
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix and classification report
- Table with all 5 models on the uploaded data

The app loads the saved models and does not retrain.
