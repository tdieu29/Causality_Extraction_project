# Configurations
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(BASE_DIR, 'data')
INDEX_DIR = Path(DATA_DIR, 'index')
EMBEDDING_DIR = Path(DATA_DIR, 'embedding')
TRAIN_DIR = Path(DATA_DIR, 'train')
TEST_DIR = Path(DATA_DIR, 'test')
PREDICT_DIR = Path(DATA_DIR, 'predict')
LOGS_DIR = Path(DATA_DIR, 'logs')


