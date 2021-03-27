# Setup installation for the application

from pathlib import Path
from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, 'requirements.txt'), 'r') as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    python_requires = '<=3.7',
    install_requires=[required_packages],
    entry_points = {
        'console_scripts': [
            'causalityExtraction = app.api:app'
        ]
    }
)