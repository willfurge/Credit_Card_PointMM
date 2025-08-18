.PHONY: init test fmt lint precommit

init:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

fmt:
	black .
	isort .

lint:
	flake8 .

test:
	pytest

precommit:
	pre-commit run --all-files
