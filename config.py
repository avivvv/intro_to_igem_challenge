"""Configuration file for the promoter expression prediction project."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Data parameters
SEQUENCE_LENGTH = 1000  # Length of promoter sequences
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# Model parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
TREE_DEPTH = 5

# Feature extraction
K_MER_SIZE = 3  # For k-mer feature extraction
