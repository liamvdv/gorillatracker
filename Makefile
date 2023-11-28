sources = *.py src tests

.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff $(sources)
	isort $(sources) --check-only --df
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(sources) --disable-recursive-aliases

.PHONY: test
test:
	pytest tests