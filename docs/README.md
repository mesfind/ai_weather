AI Weather 
==========================

ai-weather/
├── .github/workflows/      # GitHub Actions for automated testing (CI/CD)
├── docs/                   # Documentation (e.g., Sphinx or MkDocs)
├── notebooks/              # Jupyter notebooks for examples and tutorials
├── src/                    # Source code root
│   └── ai_weather/         # Your main Python package
│       ├── __init__.py     # Package initialization
│       ├── utils/          # Shared functions (data loading, grids, metrics)
│       │   ├── __init__.py
│       │   └── loaders.py
│       ├── onset/          # Functionality 1: Rainfall Onset
│       │   ├── __init__.py
│       │   └── core.py
│       ├── cessation/      # Functionality 2: Rainfall Cessation
│       │   ├── __init__.py
│       │   └── core.py
│       └── temperature/    # Functionality 3: Temperature Analysis
│           ├── __init__.py
│           └── core.py
├── tests/                  # Unit tests for each module
│   ├── test_onset.py
│   ├── test_cessation.py
│   └── test_temperature.py
├── environment.yml         # Conda environment file
├── pyproject.toml          # Modern Python packaging and build configuration
├── README.md
└── LICENSE