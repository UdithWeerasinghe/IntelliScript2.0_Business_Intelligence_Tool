import os
from datetime import datetime

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for dir_path in [DATA_DIR, GRAPHS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Test data parameters
TRAIN_START = "2007-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# Prophet optimization parameters
PROPHET_PARAM_GRID = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'yearly_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'daily_seasonality': [True, False]
}

# LLM Model comparison parameters
LLM_MODELS = {
    'llama2-7b': {
        'name': 'llama2-7b',
        'parameters': '7B',
        'latency': 150,  # Simulated latency in ms
        'context_window': 4096,
        'description': 'Ollama model - Balanced performance'
    },
    'mistral-7b': {
        'name': 'mistral-7b',
        'parameters': '7B',
        'latency': 120,  # Simulated latency in ms
        'context_window': 8192,
        'description': 'Ollama model - Fast inference'
    },
    'phi-2': {
        'name': 'phi-2',
        'parameters': '2.7B',
        'latency': 80,  # Simulated latency in ms
        'context_window': 2048,
        'description': 'Reference model - Lightweight'
    }
}

# Fine-tuning parameters
FINE_TUNING_PARAMS = {
    'base_model': 'llama2-7b',
    'training_data_path': os.path.join(DATA_DIR, "fine_tuning_data.json"),
    'output_dir': os.path.join(BASE_DIR, "fine_tuned_models"),
    'num_epochs': 3,
    'batch_size': 4,
    'learning_rate': 2e-5
} 