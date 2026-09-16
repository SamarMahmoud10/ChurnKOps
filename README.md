# ChurnOps

ChurnOps is a machine learning project for predicting customer churn, designed with an MLOps-oriented architecture.

The machine learning part is intentionally simple. The main focus of the project is building a reproducible and automated workflow around the model, including data versioning, model evaluation, testing, CI/CD, and containerization.

---

## What Does It Do?

The project uses customer information such as:

- Tenure
- Contract type
- Internet service
- Monthly charges
- Total charges
- Other customer attributes

to predict whether a customer is likely to churn.

The ML model is based on a classification approach, starting with Logistic Regression as a simple baseline.

The model is exposed through a FastAPI service so it can be used to make predictions through an API.

---

## Why MLOps?

Training a machine learning model is only one part of a real ML project.

When the project changes, we need to be able to:

- Keep track of dataset versions
- Reproduce training results
- Test changes automatically
- Evaluate new model versions
- Detect models that do not meet the required performance
- Build and run the application consistently
- Automate deployment

ChurnOps is designed to demonstrate this workflow instead of focusing on building a complex ML model.

---

## Architecture

```text
                     GitHub
                        │
              ┌─────────┴─────────┐
              │                   │
           Code CI              ML CI
              │                   │
        Tests & Linting     DVC Pipeline
                                  │
                           Train / Evaluate
                                  │
                               CML Report
                                  │
                           Model Validation
                                  │
                                Merge
                                  │
                            Docker CI
                                  │
                           Docker Image
                                  │
                                GHCR
                                  │
                                 CD
                                  │
                              Deployment

The project is divided into several main components:

## Machine Learning

Raw Data
   ↓
Preprocessing
   ↓
Training
   ↓
Evaluation
   ↓
Model Validation

## API

The trained model is served using FastAPI.

Client
   ↓
FastAPI
   ↓
Trained Model
   ↓
Prediction

## API

The trained model is served using FastAPI.

Client
   ↓
FastAPI
   ↓
Trained Model
   ↓
Prediction


## Project Structure

churnops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── ml-pipeline.yml
│       ├── docker.yml
│       └── cd.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── validate_model.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_api.py
│
├── models/
├── reports/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .dvcignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
