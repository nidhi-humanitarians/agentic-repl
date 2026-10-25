import os
from pathlib import Path

# Try loading .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
EVALS_DIR = PROJECT_ROOT / "evals"

# Particle AI API Configuration
PARTICLE_BASE_URL = os.getenv("PARTICLE_BASE_URL", "https://api.particle.ai/v1")
PARTICLE_API_KEY = os.getenv("PARTICLE_API_KEY", "")

# Model IDs available on Particle AI
MODEL_DEEPSEEK = "deepseek-v4.1-flash"
MODEL_GLM = "glm-5.3-flash"

# Default Model Selection
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", MODEL_DEEPSEEK)
