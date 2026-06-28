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
    Use `uv` to create a virtual environment and install all required dependencies:

    ```shell
    uv sync --all-groups --all-extras
    ```

    Or use the Makefile shortcut:

    ```shell
    make install
    ```

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

1. **Submit a pull request** - Open a PR against the `main` branch of this repository.

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

These targets update `griptape_nodes_library.json`, commit the change, and you can then `git push`.

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
