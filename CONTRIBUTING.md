# Contributing to Griptape Nodes Core Library

We welcome contributions to the Griptape Nodes Core Library! This library provides nodes organized by category including agents, audio, image processing, text manipulation, and more.

## Development Setup

1. **Clone the Repository:**

    ```shell
    git clone https://github.com/griptape-ai/griptape-nodes-library-standard.git
    cd griptape-nodes-library-standard
    ```

1. **Install `uv`:**
    If you don't have `uv` installed, follow the official instructions: [Astral's uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

1. **Install Dependencies:**
    Install the library and the dev group into `.venv-dev`:

    ```shell
    make install/dev
    ```

    Use the `make` targets rather than `uv sync` directly. Plain `uv` installs into `.venv`, the
    environment the engine builds for the library, and that makes the library load as FLAWED. See
    [Two virtual environments](#two-virtual-environments).

### Two virtual environments

Development uses two environments beside `griptape_nodes_library.json`:

| Environment | Owner and purpose |
| --- | --- |
| `.venv-dev` | Developer tools: IDE interpreter, tests, linting and type checking. |
| `.venv` | Engine-managed runtime dependencies used when loading the library. |

The dev group includes `griptape-nodes-engine` and its transitive dependencies. The engine adds
the library's `.venv` site-packages to the front of its import path, so older copies of shared
packages there shadow the engine's own. The engine reports these as
`ShadowedEnginePackagesProblem` and marks the library FLAWED.

Keeping the environments apart keeps developer dependencies out of the engine's library import
path. The engine creates `.venv` from the manifest's `pip_dependencies`. The unit tests register
the library the same way, so `.venv` appears after `make test/unit` as well.

The Makefile sets `UV_PROJECT_ENVIRONMENT`, so `make` targets use the right one. Running `uv`
directly needs it too, since a plain `uv run` syncs the dev group into `.venv`:

```shell
UV_PROJECT_ENVIRONMENT=.venv-dev uv run pytest tests/unit
```

### Migrating an existing checkout

`make` does not change an IDE's interpreter or remove an existing `.venv`.

1. Run `make install/dev` from the repository root to create `.venv-dev`.
2. Select `.venv-dev` as your IDE's project interpreter using the instructions below. Update any
   test, debug or run configuration that explicitly references `.venv`.
3. Close terminals with `.venv` activated, or run `deactivate` in those terminals.
4. If `.venv` holds developer dependencies (for example, `griptape-nodes-engine` shows up in
   `uv pip list --python .venv/bin/python`), stop Griptape Nodes and any other process using it.
   Check for anything you need to preserve, then delete the repository-root `.venv`. This removes
   every package installed there. Restart Griptape Nodes and let it rebuild the environment when
   it loads the library.
5. Run `make check` and `make test/unit` to check the developer environment. After registering
   the library in Griptape Nodes, check that no shadowing problem is reported.

If `.venv` holds only the engine-managed runtime dependencies, leave it in place.

### PyCharm

1. Run `make install/dev` in the repository root.
2. Open **Settings > Project > Python Interpreter**. The exact labels vary by PyCharm version.
   Add an existing local interpreter and select `<repo>/.venv-dev/bin/python`. On Windows,
   select `<repo>\.venv-dev\Scripts\python.exe`.
3. Set test, debug and Python run configurations to use the project interpreter. Replace any
   explicit path to `.venv` with the new interpreter.
4. Open a fresh terminal and check that any automatically activated environment is `.venv-dev`.

The Makefile's environment setting applies to commands launched through Make. PyCharm's own
`uv sync` or `uv run` actions need `UV_PROJECT_ENVIRONMENT=.venv-dev` in their environment.
If your PyCharm integration cannot configure that destination, use `make install/dev` for
installation and keep the IDE configured with the existing `.venv-dev` interpreter.

### VS Code

1. Install the Microsoft Python extension and run `make install/dev` in the repository root.
2. Open the Command Palette and choose **Python: Select Interpreter**. Select `.venv-dev`, or
   choose **Enter interpreter path** and browse to `<repo>/.venv-dev/bin/python`. On Windows,
   select `<repo>\.venv-dev\Scripts\python.exe`.
3. Check `.vscode/launch.json` and `.vscode/tasks.json`, if present, for explicit `.venv` paths.
   Python debugging should use the selected interpreter, or an explicit `.venv-dev` path.
   Shell tasks should call Make or set `UV_PROJECT_ENVIRONMENT=.venv-dev` when invoking `uv`.
4. Close existing integrated terminals and open a new one so terminal activation picks up the
   selected interpreter. For the Testing panel, choose **Python: Configure Tests**, select
   pytest and use `tests/unit` as the test directory.

The Python extension's interpreter selection and `uv`'s environment destination are separate
settings. Selecting `.venv-dev` does not redirect a plain `uv sync` or `uv run`. Use Make targets
or set `UV_PROJECT_ENVIRONMENT=.venv-dev` for those commands.

## Contributing Code

1. **Find the library code** - All nodes for this library are located in:

    ```
    griptape_nodes_library/
    ```

    Nodes are organized by category:

    - `agents/` - Agent nodes
    - `audio/` - Audio processing nodes
    - `image/` - Image manipulation nodes
    - `text/` - Text processing nodes
    - `video/` - Video processing nodes
    - And more (see the directory for the full list).

1. **Make your changes** - Follow the existing code structure and style in the library.

1. **Run tests** - Test the library to ensure your changes work:

    ```shell
    make test
    ```

    Or run a specific test suite:

    ```shell
    make test/unit
    make test/workflows
    ```

1. **Follow code quality standards** - Run checks before submitting:

    ```shell
    make check  # Check linting, formatting, and type errors
    make fix    # Auto-fix issues where possible
    ```

1. **Add a changelog entry** - If the change is user-facing, add an entry file to `changelog.d/`. See [Changelog](#changelog).

1. **Submit a pull request** - Open a PR against the `main` branch of this repository.

## Changelog

[`CHANGELOG.md`](CHANGELOG.md) follows [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/), with one section per `library_version`.

A PR with a user-facing change does not edit `CHANGELOG.md`. It adds one file to [`changelog.d/`](changelog.d/README.md), named `<type>-<slug>.md` for one of `added`, `changed`, `deprecated`, `removed`, `fixed`, or `security`. The whole file is the bullet, with no heading above it. One entry per file keeps PRs from conflicting. `make changelog` prints the `[Unreleased]` section with every entry file folded in.

Write for the person updating the library: which node changed, what they will notice, and why. Link the GitHub issue on the entry's last line when there is one. Start breaking changes, anything that makes a saved workflow stop working or give a different result, with `**Breaking:**`. Refactors, tests, CI, and docs-only changes get no entry. The full style guide is in [CLAUDE.md](CLAUDE.md#changelog).

`make check/changelog`, part of `make check` and CI, validates the structure of `CHANGELOG.md` and of every entry file. It never requires a PR to add an entry. That call stays with people.

## Making a Release (Maintainers)

The library version is stored in `griptape_nodes_library.json` under the `metadata.library_version` field. Releases involve bumping the version and publishing tags.

### Step 1: Bump the Version

You can bump the version locally with the Makefile or via a GitHub Actions workflow.

**Locally:**

```shell
make version/patch   # Bump patch (e.g. 0.52.3 -> 0.52.4)
make version/minor   # Bump minor (e.g. 0.52.4 -> 0.53.0)
make version/major   # Bump major (e.g. 0.53.0 -> 1.0.0)
```

These targets update `griptape_nodes_library.json`, fold the `changelog.d/` entries into a `CHANGELOG.md` section for the new version and delete them, and commit both. You can then `git push`. They stop first if `CHANGELOG.md` or `changelog.d/` holds uncommitted changes, since the roll would overwrite them.

**Via GitHub Actions:** Trigger the `Version Bump (Patch)` or `Version Bump (Minor)` workflow manually from the Actions tab.

### Step 2: Publish the Release

After the version bump is on `main`:

1. Go to the [Actions](https://github.com/griptape-ai/griptape-nodes-library-standard/actions) tab.
1. Run the **Publish Version** workflow manually. It will:
    - Create the version tag (e.g. `v0.52.4`)
    - Update the `stable` tag
    - Create a GitHub release with auto-generated notes

### Nightly Releases

The `nightly-release.yml` workflow runs daily at 02:00 UTC and updates the `nightly` tag and prerelease automatically.

## Questions or Issues?

For questions about contributing, please [open an issue](https://github.com/griptape-ai/griptape-nodes-library-standard/issues) in this repository.

Thank you for contributing!
