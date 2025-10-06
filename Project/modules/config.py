"""
This module centralizes all important file and folder paths used across the project.
It defines constants for accessing dataset locations, output directories, log storage,
and other utility folders to ensure consistency and ease of maintenance throughout the pipeline.
"""

from pathlib import Path

from sqlalchemy.dialects.postgresql.operators import (
    OVERLAP,
)  # Standard library for handling file paths.

import modules.outlier_methods as outlier  # For outlier detection methods.

# === Project Base Path ===
BASE_DIR = Path(__file__).resolve().parent.parent

# === Scenario Settings ===
RUN_SCENARIO_ID = 3

# === Dataset File Names ===
DATA_FOLDER = BASE_DIR / "dataset_creation" / f"scenario_{RUN_SCENARIO_ID}"
DATASET_FILE_NAME = "Dataset.csv"

# === Ground Truth Folder & Dataset ===
GROUND_TRUTH_FOLDER = BASE_DIR / "ground_truth_labeling" / f"scenario_{RUN_SCENARIO_ID}"
GROUND_TRUTH_FILE_NAME = "Ground_Truth_Labels.csv"


# === File Names ===
MODEL_RESULTS = "model_results"

# === Output Paths ===
OUTPUT_FOLDER = BASE_DIR / "outputs" / f"scenario_{RUN_SCENARIO_ID}"
EXPORTED_REPUTATIONS = OUTPUT_FOLDER / "exported_reputations"
EXPORTED_MODELS = OUTPUT_FOLDER / "exported_models"
SPLITTED_DATA = OUTPUT_FOLDER / "splitted_data"
OUTLIER_DATA = OUTPUT_FOLDER / "outlier_data"
PLOTS = OUTPUT_FOLDER / "plots"
LOG_FOLDER = BASE_DIR / "logs"

# === Model Configuration ===
ENCODER_MODEL_PATH = OUTPUT_FOLDER / "encoder_model.pkl"

# === Dataset Settings ===
TARGET_COLUMN_NAME = "User_Reputation"

# === Scenario Configurations ===
SCENARIO_CONFIGS = {
    1: {
        "INTENTIONAL_PERC": 0.05,
        "UNINTENTIONAL_PERC": 0.1,
        "OVERLAP_PERC": 0,
        "MISSING_PERC": 0,
        "GROUND_SAMPLE_SIZE": 500,
        "CONTAMINATION_RATE": 0.01,
        "THRESHOLD": 3.0,
        "NORMAL_INCREMENT": 0.005,
        "UNINTENTIONAL_PENALTY": 0.005,
        "INTENTIONAL_PENALTY": 0.015,
        "AUTOML_BUDGET_DEFAULT": 30,
        "AUTOML_SAMPLE_SIZE": 1000,
    },
    2: {
        "INTENTIONAL_PERC": 0.1,
        "UNINTENTIONAL_PERC": 0.2,
        "OVERLAP_PERC": 0.05,
        "MISSING_PERC": 0.02,
        "GROUND_SAMPLE_SIZE": 1000,
        "CONTAMINATION_RATE": 0.02,
        "THRESHOLD": 2.5,
        "NORMAL_INCREMENT": 0.0075,
        "UNINTENTIONAL_PENALTY": 0.0075,
        "INTENTIONAL_PENALTY": 0.02,
        "AUTOML_BUDGET_DEFAULT": 30,
        "AUTOML_SAMPLE_SIZE": 1200,
    },
    3: {
        "INTENTIONAL_PERC": 0.25,
        "UNINTENTIONAL_PERC": 0.3,
        "OVERLAP_PERC": 0.10,
        "MISSING_PERC": 0.15,
        "GROUND_SAMPLE_SIZE": 3000,
        "CONTAMINATION_RATE": 0.08,
        "THRESHOLD": 4.0,
        "NORMAL_INCREMENT": 0.005,
        "UNINTENTIONAL_PENALTY": 0.001,
        "INTENTIONAL_PENALTY": 0.03,
        "AUTOML_BUDGET_DEFAULT": 30,
        "AUTOML_SAMPLE_SIZE": 2000,
    },
}


# === Get Scenario Parameters ===
def get_scenario_param(key):
    """
    Function which retrieves the scenario parameter based on the RUN_SCENARIO_ID.
    """
    return SCENARIO_CONFIGS[RUN_SCENARIO_ID][key]


# === Dataset Settings ===
INTENTIONAL_PERC = get_scenario_param("INTENTIONAL_PERC")
UNINTENTIONAL_PERC = get_scenario_param("UNINTENTIONAL_PERC")
OVERLAP_PERC = get_scenario_param("OVERLAP_PERC")
MISSING_PERC = get_scenario_param("MISSING_PERC")

# === Ground Truth Settings ===
GROUND_SAMPLE_SIZE = get_scenario_param("GROUND_SAMPLE_SIZE")

# === Outlier Detection Settings ===
CONTAMINATION_RATE = get_scenario_param("CONTAMINATION_RATE")
THRESHOLD = get_scenario_param("THRESHOLD")

# === Reputation Adjustment Values ===
NORMAL_INCREMENT = get_scenario_param("NORMAL_INCREMENT")
UNINTENTIONAL_PENALTY = get_scenario_param("UNINTENTIONAL_PENALTY")
INTENTIONAL_PENALTY = get_scenario_param("INTENTIONAL_PENALTY")

# === AutoML default budgets ===
AUTOML_BUDGET_DEFAULT = get_scenario_param("AUTOML_BUDGET_DEFAULT")
AUTOML_SAMPLE_SIZE = get_scenario_param("AUTOML_SAMPLE_SIZE")

# === Outlier processing ===
CHUNK_PERCENT = 0.1

# === Feature Selection ===
FEATURES = [
    "User_Experience",
    "User_Activity",
    "University_Area",
    "Road_Accident",
    "Road_Type",
    "Speed_Limit",
    "Violation_Type",
    "Battery_Percentage",
    "Internet_Connection",
    "Internet_Signal_Strength",
    "Accelerometer_Range",
    "Accelerometer_Resolution",
    "Latitude",
    "Longitude",
]
INTENTIONAL_OUTLIER_COLUMNS = [
    "User_Activity",
    "University_Area",
    "Road_Accident",
    "Road_Type",
    "Speed_Limit",
    "Violation_Type",
]


# === Outlier Methods Dictionary ===
OUTLIER_METHODS = {
    "HBOS": outlier.histogram_based_method,
    "kNN": outlier.k_nearest_neighbors_method,
    "IForest": outlier.isolation_forest_method,
    "LODA": outlier.lightweight_on_line_detector_method,
    "OC-SVM": outlier.one_class_svm_method,
    "Z-Score": outlier.z_score_method,
    "IQR": outlier.iqr_method,
    "MAD": outlier.mad_method,
}

# === Final Outlier Summary ===
FINAL_OUTLIER_SUMMARY = {
    "original_size": None,
    "cleaned_size": None,
    "unique_outliers_removed": None,
    "removal_percentage": None,
}


# ===== Ensure Directories  =====
def ensure_directories():
    """
    Function which ensures that all necessary directories exist.
    """
    for folder in [
        BASE_DIR,  # Base directory of the project
        DATA_FOLDER,
        GROUND_TRUTH_FOLDER,
        OUTPUT_FOLDER,
        EXPORTED_REPUTATIONS,
        EXPORTED_MODELS,
        SPLITTED_DATA,
        OUTLIER_DATA,
        PLOTS,
        LOG_FOLDER,
    ]:  # List of folders to ensure exist
        folder.mkdir(
            parents=True, exist_ok=True
        )  # Create directories if they don't exist


ensure_directories()  # Ensure all directories are created at the start of the script
