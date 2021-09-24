install:
	python -m pip install --upgrade pip
	pip install -r requirements_dev.txt --upgrade
	pip install -r requirements.txt --upgrade
	pip install -e .
	pip install "ray[default]"
	pre-commit install

conda:
	conda install -c mnishida -c defaults -c conda-forge --file conda_pkg/conda_requirements.txt
	conda install -c mnishida -c defaults -c conda-forge --file conda_pkg/conda_requirements_dev.txt
	conda build -c mnishida -c defaults -c conda-forge --numpy 1.20 conda_pkg
	conda install -c mnishida  -c defaults -c conda-forge --use-local --force-reinstall pymwm
	pip install "ray[default]"
	pre-commit install

test:
	pytest

cov:
	pytest --cov pymwm

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8

pylint:
	pylint pymwm

lintd2:
	flake8 --select RST

lintd:
	pydocstyle --convention google pymwm

doc8:
	doc8 docs/
