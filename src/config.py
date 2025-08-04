import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Summarization settings
MAX_SUMMARY_LENGTH = 500  # Maximum words in summary
MIN_SUMMARY_LENGTH = 100  # Minimum words in summary
SUMMARY_RATIO = 0.3       # Proportion of original text to keep

# Model settings
TRANSFORMER_MODEL = "facebook/bart-large-cnn"
SPACY_MODEL = "en_core_web_sm"

# PDF output settings
PDF_MARGIN = 72  # Points (1 inch)
PDF_FONT_SIZE = 11
PDF_LINE_SPACING = 1.2