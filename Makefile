.PHONY: help
help: 
	@echo "Commands:"
	@echo "install		: installs requirements."
	@echo "install-dev	: installs development requirements."
	@echo "install-test	: installs test requirements."


.PHONY: install 
install: 
	python -m pip install -e .


.PHONY: install-dev
install-dev: 
	python -m pip install -e ".[dev]"
	pre-commit install


.PHONY: install-test
install-test:
	python -m pip install -e ".[test]"


.PHONY: app
app: 
	uvicorn app.api:app --host 127.0.0.1 --port 8001 --reload --reload-dir CausalityExtraction --reload-dir app
