# SecureML Pipeline

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)
[![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)](#)

> DevSecOps learning project: ML anomaly-detection pipeline with automated security gates

---

## Overview

**SecureML Pipeline** is a portfolio project demonstrating DevSecOps best practices applied to a production-grade machine learning system. It takes an existing ML anomaly-detection pipeline (Kafka + AutoML + PyOD) and wraps it in automated security gates across three layers: CI/CD pipeline hardening, ML-specific security (MLSecOps), and secure coding practices.

The project is designed for **security-conscious engineers, DevSecOps practitioners, and ML platform teams** looking to understand how to integrate security validation into ML workflows at scale.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Architecture](#architecture)
- [Security Goals](#security-goals)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Status](#status)
- [Contact](#contact)

---

## Background

This project originated as an **MSc thesis** on predicting user reputation from behaviour logs using advanced AutoML techniques. The core pipeline combines:

- **TPOT** (Tree-based Pipeline Optimization Tool) for genetic-algorithm-driven pipeline search
- **FLAML** (Fast Library for AutoML) for cost-efficient hyperparameter tuning
- **PyOD** for anomaly detection and outlier removal
- **Kafka** for optional real-time streaming and scoring
- **Dask** for parallelized data processing

The thesis demonstrated how to automate feature engineering, model selection, and validation on large-scale behaviour datasets. For details on the original research, see `Thesis.pdf` in this repository.

**Key ML components:**
- Automatic pipeline optimization with TPOT & FLAML
- Parallelized data processing with Dask
- Outlier detection and misbehaviour classification
- Full reproducibility (all models and pipelines versioned)

---

## Architecture

```
┌─────────────┐      ┌──────────┐      ┌─────────────────────┐      ┌─────────────┐
│  Producer   │ ──→  │  Kafka   │  ──→ │  ML Consumer        │  ──→ │  Results    │
│  (Events)   │      │  Broker  │      │ (TPOT/PyOD/AutoML)  │      │  (Anomalies)│
└─────────────┘      └──────────┘      └─────────────────────┘      └─────────────┘
                            ▲                       ▲                       ▲
                            │                       │                       │
                     ┌──────────────────────────────────────────────────────┐
                     │        GitHub Actions Security Gates                 │
                     │  • Secrets scanning (gitleaks)                       │
                     │  • SAST (bandit, semgrep)                           │
                     │  • Dependency scanning (pip-audit)                   │
                     │  • Linting & code quality                            │
                     │  • Container scanning (trivy)                        │
                     │  • Data validation gates (pandera)                   │
                     └──────────────────────────────────────────────────────┘
```

The pipeline is wrapped by three layers of automated security validation that ensure code, dependencies, and model inputs meet security standards before deployment.

---

## Security Goals

Our DevSecOps approach is organized into three complementary security layers:

| Layer | Focus | Controls |
|-------|-------|----------|
| **Layer 1: CI/CD Pipeline** | Secure the deployment infrastructure | GitHub Actions gates: secrets scanning (gitleaks), SAST analysis, dependency vulnerability scanning, linting, container image scanning (trivy) |
| **Layer 2: ML Process** | MLSecOps—secure data and models | Data validation (pandera schemas), adversarial input testing, model artifact integrity checks, automated retraining gates |
| **Layer 3: Code Security** | Secure development practices | SAST with bandit & semgrep, secure coding standards enforcement, pre-commit hooks, dependency pinning |

---

## Roadmap

Development is organized into five phases:

- [ ] **Phase 1: Devcontainer & Repo Hygiene**  
      Set up reproducible development environment; add gitleaks and pre-commit hooks to prevent credential leaks

- [ ] **Phase 2: SAST & Dependency Scanning**  
      Integrate bandit & semgrep for code security; add pip-audit for dependency vulnerabilities in GitHub Actions

- [ ] **Phase 3: Container Hardening**  
      Apply hadolint for Dockerfile linting; integrate trivy for container image scanning

- [ ] **Phase 4: Data & Model Validation**  
      Implement pandera for input data validation; add adversarial input tests; verify model artifact integrity

- [ ] **Phase 5: Documentation & SonarQube Integration**  
      Complete security documentation; integrate SonarQube/SonarCloud; publish security write-up and architecture guide

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Data Pipeline** | Kafka, Dask, PyOD |
| **ML/AutoML** | TPOT, FLAML, Scikit-learn |
| **CI/CD** | GitHub Actions |
| **Security Scanning** | gitleaks, bandit, semgrep, trivy, pip-audit |
| **Data Validation** | Pandera |
| **Containerization** | Docker, Dockerfile linting (hadolint) |
| **Development** | Devcontainer, pre-commit |

---

## Quick Start

```bash
# 1. Create / activate environment
conda create -n MATRIX python=3.10
conda activate MATRIX

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

**Outputs:**  
All results are saved under `outputs/` (models, predictions, plots, and logs).

**Visualization:**  
Open `outputs/.../plots/` for TPOT vs FLAML comparisons and reputation scatterplots.

---

### Docker Compose (Alternative Quick Start)

You can run all components—**Kafka, Dask, Kafdrop**, and all pipeline steps—using Docker Compose.

```bash
# 1. Build and start all services in detached mode
docker-compose up --build -d

# 2. Check the status of all containers
docker-compose ps

# 3. View logs for any service (example: main pipeline)
docker-compose logs -f main

# 4. (Optional) Run a specific service step manually, for example:
docker-compose run --rm data
docker-compose run --rm label
docker-compose run --rm main
```

**Service overview:**
- `main` runs the main pipeline (`python main.py`)
- `data` generates or processes datasets (`python data_generator/dataset_creation.py`)
- `label` runs ground truth labeling
- `smoke` executes tests (`python tests/smoke_test.py`)
- `dask` provides the Dask dashboard ([http://localhost:8787](http://localhost:8787))
- `kafka`, `zookeeper` power the Kafka broker
- `kafdrop` provides a Kafka web UI at [http://localhost:9000](http://localhost:9000)

| Service   | Description                                | Web UI                    |
|-----------|--------------------------------------------|---------------------------|
| main      | Runs the main pipeline                      | -                         |
| data      | Dataset creation / preprocessing            | -                         |
| label     | Ground truth labeling                       | -                         |
| smoke     | Pipeline smoke tests                        | -                         |
| dask      | Dask distributed cluster                    | [localhost:8787](http://localhost:8787) |
| kafka     | Kafka broker for data streaming             | -                         |
| zookeeper | Zookeeper for Kafka                         | -                         |
| kafdrop   | Kafka web-based monitoring                  | [localhost:9000](http://localhost:9000) |

**Dashboards:**
- Dask dashboard: [http://localhost:8787](http://localhost:8787)
- Kafdrop (Kafka Web UI): [http://localhost:9000](http://localhost:9000)

**Stopping everything:**
```bash
docker-compose down
```

**Volumes:**  
- Data and output folders are mounted: changes persist on your host under `./data` and `./outputs`.

**Tips:**
- To execute arbitrary commands inside a running service:
  ```bash
  docker-compose exec main bash
  ```
- The pipeline can stream to Kafka (topic: `reputation`), and you can monitor messages in Kafdrop UI.

**Requirements:**  
- Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) first.

---

## Project Structure

```
Project/
 ├─ docker-compose.yaml
 ├─ dockerfile
 ├─ full_pipeline.py        # Optional all-in-one run
 ├─ main.py                 # Main pipeline entry
 ├─ dataset_creation/       # Dataset creation scripts + scenarios
 ├─ ground_truth_labeling/  # Ground truth labeling scripts + scenarios
 ├─ modules/                # All pipeline code (see below)
 ├─ outputs/                # Saved models, predictions, plots, logs
 ├─ tests/
 ├─ requirements.txt
 └─ README.md
```
**Key modules:**  
- `config.py` — central constants, budgets
- `system_guard.py` — Dask client + logging
- `data_processor.py` — data cleaning, encoding
- `outlier_methods.py`, `outlier_processor.py` — outlier detection (HBOS, iForest, etc.)
- `misbehaving_data_splitter.py` — train/test split, classifier
- `final_reputation_predictor.py` — TPOT & FLAML regressors
- `visualizer.py` — plot dashboards

---

## Configurable Parameters

| Purpose        | Argument (`budget`) | Details                                      | Default (`config.py`)      |
| -------------- | ------------------- | -------------------------------------------- | -------------------------- |
| TPOT depth     | `budget=N`          | `population_size=N`, `generations=N//2`      | 20 pop / 5 gens            |
| FLAML time     | `budget=N`          | `time_budget=N` seconds                      | 120 s                      |

**Example usage:**
```python
tpot_f.tpot_model_runner(X, y, "experiment", budget=30)
flaml_f.flaml_model_runner(X, y, "experiment", budget=600)
```

---

## Pipeline Overview

1. **Data Preparation:**  
   Raw logs are cleaned, encoded, and split.

2. **Outlier Detection & Removal:**  
   Techniques like HBOS, iForest (PyOD) identify and exclude anomalies.

3. **Misbehaviour Detection:**  
   Automatic classifiers split data into “normal” and “misbehaving” subsets.

4. **Reputation Prediction:**  
   - **TPOT** explores full ML pipelines via genetic programming.
   - **FLAML** finds fast, strong baselines with low resource usage.

5. **Result Saving & Visualization:**  
   All models, predictions, and plots are saved for analysis and reproducibility.

---

## Dask Parallelism

- The pipeline leverages Dask for efficient parallel processing—ideal for large datasets and AutoML.
- Local Dask cluster: `http://localhost:8787` (web UI).
- Outlier detection, CSV I/O, and TPOT search are all parallelized.

---

## Kafka Streaming (Optional)

For online/real-time prediction demo:
```bash
cd kafka
docker-compose up -d
python kafka_streamer.py
```
Streams predictions to Kafka topic `reputation`.

---

## Results & Outputs

- All fitted models are versioned and saved under `outputs/`.
- All prediction results are available as CSV.
- Interactive visualizations help compare model quality and analyze reputation prediction accuracy.


---

## Status

🚧 **Work in Progress** — The project is transitioning from an MSc thesis demonstration to a production-focused DevSecOps portfolio project. Security gates are being implemented incrementally across all three layers.

Current focus: Foundation setup (Phases 1–2)

---

## Citation (Original Thesis)

If this project's ML components contribute to your research, please cite:

> Olson et al. "TPOT: A Tree‑based Pipeline Optimization Tool for AutoML." *Bioinformatics* 2020.  
> Zheng et al. "FLAML: A Fast Library for AutoML & Hyperparameter Tuning." *MLSys* 2022.

---

## Contributing

Contributions, bug reports, and suggestions are welcome!  
Please open an issue or submit a pull request.

---

## License

This project is open-source. See [LICENSE](LICENSE) for details.

---

## Contact

Created by Spiros Karagilanis | [GitHub](https://github.com/BlackPhoenixr) | Questions? Open an issue.

---

✨ **Secure ML at scale—one gate at a time.** 🔒