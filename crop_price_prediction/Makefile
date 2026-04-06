install:
	pip install -r requirements.txt

features:
	python src/features/build_features.py

train:
	jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --output notebooks/02_modeling_executed.ipynb

run:
	python app.py

test:
	pytest tests/ -v

lint:
	python -m flake8 src/ app.py --max-line-length=100 --ignore=E501,W503

all: install features train test run
