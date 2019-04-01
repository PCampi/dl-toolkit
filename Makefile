SRC_FOLDER=./src

.PHONY: clean-pyc clean-build clean-coverage

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

coverage:
	coverage run --source $(SRC_FOLDER) -m pytest && coverage html

coverage-view: coverage
	open htmlcov/index.html

clean-coverage:
	rm -r htmlcov
	rm .coverage*

dist: coverage
	python3 setup.py sdist bdist_wheel

clean: clean-pyc clean-build clean-coverage
