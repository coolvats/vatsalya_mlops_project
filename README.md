# Cancer Type Prediction MLOps Project

This project demonstrates an end-to-end machine learning workflow for predicting cancer type (Malignant or Benign) using the Wisconsin Breast Cancer dataset. The workflow includes data validation, preprocessing, model training, experiment tracking with MLflow, and model serialization for deployment in Streamlit.

## Project Structure

```
├── data
│   ├── preprocess
│   └── raw
│       └── cancer-data.csv
├── mlprocess.ipynb
├── data.dvc
```

## Steps

### 1. Data Validation
- The raw data is loaded from `data/raw/cancer-data.csv`.
- The notebook checks for missing values and duplicate rows.

### 2. Data Preprocessing
- Duplicates are removed.
- Missing values in numeric columns are filled with the median.
- The cleaned data is saved to `data/preprocess/cancer-data-preprocessed.csv`.

### 3. Model Training & Experiment Tracking
- Features and target (`diagnosis`: M=malignant, B=benign) are prepared.
- Data is split into train and test sets.
- A RandomForestClassifier is trained.
- Model performance is evaluated using accuracy and F1-score.
- MLflow is used to track experiments, metrics, and model artifacts.

### 4. Model Serialization
- The trained model is saved as `data/preprocess/cancer_rf_model.pkl` for use in Streamlit.

### 5. Running MLflow Locally
To start the MLflow UI locally and view experiment results, run:

```
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, mlflow, joblib

Install dependencies with:

```
pip install pandas numpy scikit-learn mlflow joblib
```

## Streamlit Deployment
You can use the generated `.pkl` model in a Streamlit app for cancer type prediction and experimentation.

---

For details, see the step-by-step workflow in `mlprocess.ipynb`.
