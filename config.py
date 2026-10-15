# Configuration — model, paths, thresholds

import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-70b-versatile"

# Paths
DATA_RAW = "data/raw/sncf_tgv_raw.csv"
DATA_CLEANED = "data/cleaned/sncf_tgv_cleaned.csv"
RESULTS_DIR = "results/"

# Profiling thresholds
MISSING_THRESHOLD = 0.05      # flag column if >5% missing
OUTLIER_IQR_FACTOR = 1.5      # standard IQR multiplier
HIGH_CARDINALITY_THRESHOLD = 0.95  # flag if unique ratio > 95%
