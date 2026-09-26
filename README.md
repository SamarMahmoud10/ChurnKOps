# ChurnKops 

ChurnKops is an end-to-end Machine Learning project designed to predict customer churn, heavily focused on a production-ready MLOps architecture. While the underlying Machine Learning model is intentionally straightforward, the primary emphasis is on automated workflows, reproducible pipelines, continuous integration, containerization, and orchestration.

---

##  Architecture & Project Structure

The project is structured to decouple the core Machine Learning experiment/logic from the operational and infrastructure layers (DevOps/MLOps):

```text
ChurnKops/
│
├── ml/                      # Machine Learning Core 
│   ├── src/                 # Training, preprocessing, and inference scripts
│   ├── tests/               # Unit tests for ML code
│   ├── models/              # Serialized model artifacts (.pkl / .onnx)
│   └── requirements.txt     # ML-specific dependencies
│
├── tests/                   # Integration Testing Layer
│   └── integration/         # API and end-to-end integration tests
│
├── Dockerfile               # Multi-stage production Docker image
├── docker-compose.yml       # Local orchestration for API and tools
├── .dockerignore            # Docker exclusion rules
├── .env.example             # Template for environment variables
├── README.md                # Project documentation
│
├── .github/                 # CI/CD Automation Layer 
│   └── workflows/
│       ├── ci.yml           # Runs unit/integration tests and linting
│       ├── security.yml     # Code and dependency vulnerability scanning
│       └── docker.yml       # Builds and pushes Docker images to registry
│
└── k8s/                     # Infrastructure Deployment (Optional)
    ├── deployment.yaml      # Kubernetes deployment specs
    └── service.yaml         # Kubernetes service definitions
```

### Key Attributes Used for Prediction:
* **Tenure:** Number of months the customer has stayed with the company.
* **Contract Type:** Month-to-month, one year, or two years.
* **Internet Service:** Fiber optic, DSL, or no internet.
* **Monthly Charges & Total Charges:** Financial metrics per user.
* **Other Demographics:** Additional customer features.

---

##  Running Locally

### Prerequisites
* Python 3.10+
* Docker & Docker Compose

### Setup
1. Clone the repository and navigate to the root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On WSL/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r ml/requirements.txt
   ```

---

## 🐳 Docker

The project uses a production-ready `Dockerfile` to containerize the inference API:
```bash
# Build the image
docker build -t churnkops-api .

# Run the container
docker run -p 8000:8000 --env-file .env churnkops-api
```

---

##  Environment Variables

Copy the template to create your local configurations:
```bash
cp .env.example .env
```
Key configurations include:
* `MODEL_PATH`: Location of the serialized model artifact.
* `API_PORT`: Port for exposing the inference service.
* `ENV`: Environment switch (`development`, `production`).

---

##  CI/CD Pipeline

Automated with GitHub Actions under `.github/workflows/`:
* **CI (`ci.yml`):** Automatically triggers on PRs to run `pytest` covering both `ml/tests/` and `tests/integration/`.
* **Docker (`docker.yml`):** Automatically builds the container on successful merges to the `main` branch.

---

##  Security Scanning

Managed via `security.yml` in GitHub Actions:
* Utilizes tools like **Bandit** for Python static code analysis.
* Employs **Trivy** or GitHub's native Dependabot for scanning Docker layer vulnerabilities and dependency leaks.

---

##  Deployment

Deployment manifests are located in the `k8s/` folder:
* Supports smooth rollouts on Kubernetes clusters using rolling updates.
* Exposes the inference service securely using K8s Services/Ingress.

---

##  Monitoring

*(Future Scope)* 
Designed to monitor data drift and model performance metrics in production, feeding logs into tools like Prometheus or specialized MLOps monitoring frameworks.
