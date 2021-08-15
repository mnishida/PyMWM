install:
	pip install -r requirements.txt --upgrade
	pip install -r requirements_dev.txt --upgrade
	pip install git+https://github.com/mnishida/RII_Pandas
	pip install git+https://github.com/mnishida/PyOptMat
	pip install -e .
	pre-commit install

conda:
	conda install --file conda_requirements.txt
	conda install --file conda_requirements_dev.txt
	pip install pre-commit tox pytest-regressions doc8
	pip install sphinx-markdown-tables sphinx-autodoc-typehints
	pip install mkl-include
	pip install ray
	pip install git+https://github.com/mnishida/RII_Pandas
	pip install git+https://github.com/mnishida/PyOptMat
	pip install -e .
	pre-commit install

test:
	pytest

cov:
	pytest --cov= riip

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8

pylint:
	pylint riip

lintd2:
	flake8 --select RST

lintd:
	pydocstyle riip

doc8:
	doc8 docs/