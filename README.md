# User‑Reputation‑Prediction Thesis Project

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)
<!-- Add real links as appropriate -->

> **Academic Thesis: Predicting User Reputation from Behaviour Logs with AutoML and Parallelism**

---

## About

This project presents an end-to-end machine learning pipeline for predicting user reputation from raw behaviour logs. It combines advanced AutoML techniques with scalable parallel processing to automate feature engineering, model selection, and result validation.  
**Main technologies:** TPOT, FLAML, Dask, Scikit-learn, PyOD.

**Key features:**
- **Automatic pipeline optimization** with TPOT (genetic search) & FLAML (cost-efficient tuning)
- **Parallelized data processing** with Dask for efficient execution on large datasets
- **Outlier removal** and **misbehaviour detection** to improve data quality
- **Fully reproducible**: every model and pipeline is saved for auditing and further research
- Optional **Kafka streaming** for real-time scoring (demo-ready stub)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configurable Parameters](#configurable-parameters)
- [Pipeline Overview](#pipeline-overview)
- [Dask Parallelism](#dask-parallelism)
- [Kafka Streaming (Optional)](#kafka-streaming-optional)
- [Results & Outputs](#results--outputs)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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

## Citation

If this project contributes to your research, please cite:
> Olson et al. “TPOT: A Tree‑based Pipeline Optimization Tool for AutoML.” *Bioinformatics* 2020.  
> Zheng et al. “FLAML: A Fast Library for AutoML & Hyperparameter Tuning.” *MLSys* 2022.

---

## Contributing

Contributions, bug reports, and suggestions are welcome!  
Please open an issue or submit a pull request.

---

## License

This project is open-source. See [LICENSE](LICENSE) for details.  
<!-- Change or remove if you use a different license -->

---

## Contact

Created by Spiros Karagilanis (contact: [GitHub](https://github.com/BlackPhoenixr) or by email upon request).  
Questions or suggestions? Open an issue or contact via [GitHub](#).

---

✨ **Happy researching—and may your reputations always converge!** 🎓