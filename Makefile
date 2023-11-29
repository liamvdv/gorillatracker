sources = src tests

.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff $(sources)
	isort $(sources) --check-only --df --settings-path gorillatracker/pyproject.toml
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(sources) --disable-recursive-aliases

.PHONY: test
test:
	pytest tests