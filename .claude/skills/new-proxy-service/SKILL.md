---
name: new-proxy-service
description: End-to-end implementation of a new proxy service. Takes a GitHub issue and API key, produces a working proxy client and node with PRs in both repos.
argument-hint: <owner/repo#issue> --key <api-key>
allowed-tools: Bash Read Write Edit Grep Glob WebFetch Skill
disable-model-invocation: false
---

# Implement a New Proxy Service End-to-End

This skill orchestrates the full implementation of a new proxy service, from API research through to PRs in both repos.

Input: `$ARGUMENTS` should contain a GitHub issue reference and API key, e.g.:
```
griptape-ai/griptape-nodes#4176 --key sk-xxx
```

## Phase 1: Research the API

Run the `/api-research` skill:

```
/api-research $ARGUMENTS
```

This produces a spec file at `.scratch/proxy-spec-<service-name>/spec.md` and saves the API key to `.scratch/proxy-spec-<service-name>/.api_key`.

After it completes, read the spec file to confirm it looks reasonable. Verify:
- All sections are populated
- Model IDs are defined **and were verified with a successful API call** (check the Request/Response Examples section for real response data)
- The endpoint path was verified with a successful API call
- Request/response examples look correct and contain real data (not placeholders)
- Classification (sync/async) is determined

**Gate check:** If the spec's Request/Response Examples section does not contain a real successful response, do NOT proceed to Phase 2. The spec must be backed by at least one verified successful API call to avoid building on incorrect assumptions.

Note the spec file path for the next phases.

## Phase 2: Implement the Proxy Client

Run the `/impl-proxy-client` skill from the griptape-cloud repo:

```
/impl-proxy-client .scratch/proxy-spec-<service-name>/spec.md
```

This phase:
1. Creates the proxy client in `../../griptape-cloud`
2. Starts local infrastructure with `make up/debug`
3. Creates DB records and tests the proxy locally
4. **Reviews the client against the spec** (cross-checks endpoints, request format, response parsing, billing, model IDs)
5. Creates a PR in griptape-cloud

After it completes, verify:
- The local proxy test succeeded
- The review checklist passed with no discrepancies
- The PR was created
- Local infrastructure is still running (needed for Phase 3)

Note the griptape-cloud PR URL.

## Phase 3: Implement the Node

Run the `/impl-proxy-node` skill from this repo:

```
/impl-proxy-node .scratch/proxy-spec-<service-name>/spec.md
```

This phase:
1. Creates the node in this repo
2. Writes an integration test
3. Runs the integration test against the still-running local griptape-cloud
4. **Reviews the node against the spec and docs** (uses agent-browser to re-read the API docs and cross-checks parameters, payload structure, response parsing, model mapping)
5. Creates a PR in this repo

After it completes, verify:
- The integration test passed
- The review checklist passed with no discrepancies
- The PR was created

Note the griptape-nodes-library-standard PR URL.

## Phase 4: Cleanup

Tear down the local infrastructure:

```bash
cd ../../griptape-cloud && make down
```

## Phase 5: Summary

Print a summary of everything that was done:

```
=== New Proxy Service Implementation Complete ===

Service: <service name>
GitHub Issue: <issue reference>

Model IDs: <list>

PRs Created:
  - griptape-cloud: <PR URL>
  - griptape-nodes-library-standard: <PR URL>

Files Created/Modified:
  griptape-cloud:
    - control_plane/api/griptapecloud/components/proxy/clients/<provider>.py (new)
    - control_plane/api/griptapecloud/components/proxy/views.py (modified)
    - control_plane/api/griptapecloud/components/proxy/v2/tasks.py (modified, if sync)
    - control_plane/api/griptapecloud/components/credits/activities.py (modified)
    - control_plane/api/griptapecloud/components/service_model_config/models.py (modified, if new provider)

  griptape-nodes-library-standard:
    - griptape_nodes_library/<category>/<provider>_<modality>.py (new)
    - griptape_nodes_library/<category>/__init__.py (modified)
    - griptape_nodes_library.json (modified)
    - tests/integration/test_<provider>_<modality>.py (new)

Manual Steps Remaining:
  - Review and merge both PRs
  - In production: create ServiceModelConfig and ServiceModelConfigAuthDetails DB records
  - In production: create ActivityCreditCost records for billing
```

## Error Recovery

If any phase fails:
1. Check the error output carefully
2. Fix the issue in the relevant repo
3. Re-run the failed skill (they are idempotent for the implementation steps)
4. Always ensure local infra is torn down when done: `cd ../../griptape-cloud && make down`
