.PHONY: version/get
version/get: ## Get version.
	@jq -r '.metadata.library_version' griptape_nodes_library.json

.PHONY: version/set
version/set: ## Set version. Usage: make version/set v=1.2.3
	@jq --arg v "$(v)" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp
	@mv griptape_nodes_library.json.tmp griptape_nodes_library.json
	@make version/commit

.PHONY: version/patch
version/patch: ## Bump patch version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$${major}.$${minor}.$$((patch + 1))"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp; \
	mv griptape_nodes_library.json.tmp griptape_nodes_library.json; \
	echo "Bumped to $$NEW_VERSION"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/minor
version/minor: ## Bump minor version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$${major}.$$((minor + 1)).0"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp; \
	mv griptape_nodes_library.json.tmp griptape_nodes_library.json; \
	echo "Bumped to $$NEW_VERSION"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/major
version/major: ## Bump major version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$$((major + 1)).0.0"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp; \
	mv griptape_nodes_library.json.tmp griptape_nodes_library.json; \
	echo "Bumped to $$NEW_VERSION"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/commit
version/commit: ## Commit version.
	@git add griptape_nodes_library.json
	@git commit -m "chore: bump v$$($(MAKE) --no-print-directory version/get)"

.PHONY: version/publish
version/publish: ## Create and push git tags.
	@git fetch --tags --force
	@VERSION=$$($(MAKE) --no-print-directory version/get); \
	git tag "v$$VERSION"; \
	git tag stable -f; \
	git push origin "v$$VERSION"; \
	git push -f origin stable

.PHONY: deps/sync
deps/sync: ## Sync pip_dependencies in griptape_nodes_library.json from pyproject.toml.
	@uv run python -c "\
import tomllib, json; \
pyproject = tomllib.load(open('pyproject.toml', 'rb')); \
deps = [d for d in pyproject['project']['dependencies'] if not d.startswith('griptape-nodes')]; \
lib = json.load(open('griptape_nodes_library.json')); \
lib['metadata'].setdefault('dependencies', {})['pip_dependencies'] = deps; \
open('griptape_nodes_library.json', 'w').write(json.dumps(lib, indent=2) + '\n'); \
print(f'Synced {len(deps)} dependencies to griptape_nodes_library.json')"

.PHONY: install
install: ## Install all dependencies.
	@make install/all

.PHONY: install/core
install/core: deps/sync ## Install core dependencies.
	@uv sync

.PHONY: install/all
install/all: deps/sync ## Install all dependencies.
	@uv sync --all-groups --all-extras

.PHONY: install/dev
install/dev: ## Install dev dependencies.
	@uv sync --group dev

.PHONY: lint
lint: ## Lint project.
	@uv run ruff check --fix

.PHONY: format
format: ## Format project.
	@uv run ruff format

.PHONY: fix
fix: ## Fix project.
	@make format
	@uv run ruff check --fix --unsafe-fixes

.PHONY: check
check: check/format check/lint check/types check/json ## Run all checks.

.PHONY: check/format
check/format:
	@uv run ruff format --check

.PHONY: check/lint
check/lint:
	@uv run ruff check .

.PHONY: check/types
check/types:
	@uv run pyright .

.PHONY: check/json
check/json: ## Validate JSON files.
	@echo "Checking JSON files..."
	@find . -name "*.json" -type f \
		! -path "./.venv/*" \
		! -path "./node_modules/*" \
		-exec sh -c 'jq empty "{}" > /dev/null 2>&1 || (echo "Invalid JSON: {}" && exit 1)' \;

.PHONY: test
test: ## Run all tests.
	@uv run pytest tests/

.PHONY: test/unit
test/unit: ## Run unit tests.
	@uv run pytest tests/unit

.PHONY: test/unit/coverage
test/unit/coverage: ## Run unit tests with coverage.
	@uv run pytest --cov=griptape_nodes_library --cov-report=xml --cov-report=term tests/unit

.PHONY: test/workflows
test/workflows: ## Run workflow tests.
	@uv run pytest -s tests/workflows

.DEFAULT_GOAL := help
.PHONY: help
help: ## Print Makefile help text.
	@# Matches targets with a comment in the format <target>: ## <comment>
	@# then formats help output using these values.
	@grep -E '^[a-zA-Z_\/-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[36m%-12s\033[0m%s\n", $$1, $$2}'

