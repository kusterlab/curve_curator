install:
	pip install --use-feature=in-tree-build .

unit_tests:
	python3 -m pytest tests/unit