.PHONY: setup data features train analyze clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

data:
	python src/data_processing/pgn_processor.py
	python src/data_processing/csv_to_sql.py

features:
	python src/feature_engineering/process_features.py

train:
	python src/model/cluster.py

analyze:
	jupyter nbconvert --execute notebooks/reports/*.ipynb --to html

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete