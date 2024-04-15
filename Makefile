sources = src tests *.py

.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff check $(sources)
	isort $(sources) --check-only --df
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(sources) --exclude ^dlib/

.PHONY: test
test:
	# TODO(liamvdv): Await fix https://github.com/Lightning-AI/pytorch-lightning/issues/16756
	# lightning_utilities use deprecated pkg_resources API
	pytest -W ignore::DeprecationWarning  tests