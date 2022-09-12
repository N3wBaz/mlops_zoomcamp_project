lint:
	isort .
	black .
	pylint --recursive=y .

test: lint
	pytest tests/

setup:
	pipenv install --dev
	pre-commit install
