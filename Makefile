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
	@CURRENT=$$(make version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$${major}.$${minor}.$$((patch + 1))"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp; \
	mv griptape_nodes_library.json.tmp griptape_nodes_library.json; \
	echo "Bumped to $$NEW_VERSION"
	@make version/commit

.PHONY: version/minor
version/minor: ## Bump minor version.
	@CURRENT=$$(make version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$${major}.$$((minor + 1)).0"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp; \
	mv griptape_nodes_library.json.tmp griptape_nodes_library.json; \
	echo "Bumped to $$NEW_VERSION"
	@make version/commit

.PHONY: version/major
version/major: ## Bump major version.
	@CURRENT=$$(make version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$$((major + 1)).0.0"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' griptape_nodes_library.json > griptape_nodes_library.json.tmp; \
	mv griptape_nodes_library.json.tmp griptape_nodes_library.json; \
	echo "Bumped to $$NEW_VERSION"
	@make version/commit

.PHONY: version/commit
version/commit: ## Commit version.
	@git add griptape_nodes_library.json
	@git commit -m "chore: bump v$$(make version/get)"

.PHONY: version/publish
version/publish: ## Create and push git tags.
	@git fetch --tags --force
	@git tag "v$$(make version/get)"
	@git tag stable -f
	@git push origin "v$$(make version/get)"
	@git push -f origin stable

.PHONY: install
install: ## Install all dependencies.
	@make install/all

.PHONY: install/core
install/core: ## Install core dependencies.
	@uv sync

.PHONY: install/all
install/all: ## Install all dependencies.
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
check: check/format check/lint check/types ## Run all checks.

.PHONY: check/format
check/format:
	@uv run ruff format --check

.PHONY: check/lint
check/lint:
	@uv run ruff check .

.PHONY: check/types
check/types:
	@uv run pyright .

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
