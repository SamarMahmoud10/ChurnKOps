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

## Main Components

src/

Contains the machine learning pipeline:

preprocess.py — prepares the dataset for training.
train.py — trains and saves the model.
evaluate.py — evaluates model performance.
validate_model.py — checks whether the model meets the required performance level.
tests/

Contains automated tests for:

Data preprocessing
Model training
API behavior
api/

Contains the FastAPI application used to serve the trained model.

data/

Stores the raw and processed data used by the ML pipeline.

models/

Stores generated model artifacts locally and through the appropriate versioning workflow.

reports/

Stores evaluation results and ML reports.

MLOps Workflow

The project is developed incrementally.

## The intended workflow is:

Data
 ↓
Preprocessing
 ↓
Training
 ↓
Evaluation
 ↓
Validation
 ↓
DVC
 ↓
CML
 ↓
CI
 ↓
Docker
 ↓
CD

Each stage adds automation and reproducibility around the same ML project.

Technology Stack

Machine Learning

Python
pandas
NumPy
scikit-learn

API

FastAPI
Uvicorn

MLOps

DVC
CML
GitHub Actions

Containerization

Docker
GitHub Container Registry

Testing & Code Quality

pytest
Ruff
Getting Started

The project is currently being developed incrementally.

Once the required environment is available, the application will provide:

A reproducible ML pipeline
A trained churn prediction model
A FastAPI prediction endpoint
Automated tests
Automated ML evaluation and reporting
Containerized deployment
Project Goal

## The goal of ChurnOps is to demonstrate how a simple machine learning model can be transformed into a reproducible, testable, and automated ML system using modern MLOps practices.

## The ML model is kept intentionally simple so the project can focus on the engineering and automation surrounding it.