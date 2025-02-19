# Chess Player Type Analysis Thesis

Research on chess player types and playing styles using machine learning techniques.

## Project Overview
This project analyzes chess games to identify and classify different player types and playing styles. It processes chess game data, extracts relevant features, and uses machine learning to cluster players based on their playing characteristics.

## Project Structure

- `src/`: Source code
  - `data_processing/`: Scripts for data collection and preprocessing
  - `feature_engineering/`: Feature extraction and transformation
  - `model/`: ML model implementation
  - `visualization/`: Data visualization tools
  - `utils/`: Utility functions and helpers

- `data/`: Data files
  - `raw/`: Original data
  - `processed/`: Cleaned and processed data
  - `interim/`: Intermediate data
  - `external/`: External data sources

- `docs/`: Documentation
  - `references/`: Papers, manuals, and references
  - `analysis/`: Analysis documents
  - `meetings/`: Meeting notes

- `tests/`: Test files
  - `unit/`: Unit tests
  - `integration/`: Integration tests

- `notebooks/`: Jupyter notebooks
  - `exploratory/`: EDA notebooks
  - `reports/`: Analysis reports

- `models/`: Model files
  - `trained/`: Trained model files
  - `evaluation/`: Model evaluation results

- `reports/`: Generated analysis
  - `figures/`: Generated graphics
  - `tables/`: Generated tables

- `config/`: Configuration files

- `logs/`: Log files

## Requirements
- Python 3.8+
- Required packages listed in `requirements.txt`

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
python src/data_processing/csv_to_sql.py
```

## Data Processing Pipeline
1. Game data processing: `src/data_processing/pgn_processor.py`
2. Feature extraction: `src/feature_engineering/game_features.py`
3. Analysis and clustering: `src/model/cluster.py`

## Analysis Reports
Analysis results and visualizations can be found in:
- `reports/figures/`: Visual analysis of chess positions and player styles
- `results/`: Generated analysis files and figures
- `notebooks/reports/`: Detailed analysis notebooks

## Contributing
1. Create a new branch for your feature
2. Write tests in the `tests/` directory
3. Submit a pull request

## Project Status
Active development - Part of thesis research at Uppsala University
