sources = src tests *.py

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
	# TODO(memben): reinclude cvat_import.py
	# TODO(joschaSchroff): reinclude video_json_tracker.py
	mypy $(sources) --exclude ^dlib/ --exclude ^src/gorillatracker/utils/cvat_import.py --exclude ^src/gorillatracker/scripts/video_json_tracker.py

.PHONY: test
test:
	# TODO(liamvdv): Await fix https://github.com/Lightning-AI/pytorch-lightning/issues/16756
	# lightning_utilities use deprecated pkg_resources API
	pytest -W ignore::DeprecationWarning  tests