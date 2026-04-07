---
name: new-proxy-service
description: End-to-end implementation of a new proxy service. Takes a GitHub issue and credentials (API key OR client credentials), produces a working proxy client and node with PRs in both repos.
argument-hint: <owner/repo#issue> --key <api-key> | --client-id <id> --client-secret <secret>
allowed-tools: Bash Read Write Edit Grep Glob WebFetch Skill Agent
disable-model-invocation: false
---

# Implement a New Proxy Service End-to-End

This skill orchestrates the full implementation of a new proxy service, from API research through to PRs in both repos. Each phase runs in a subagent to keep context clean.

Input: `$ARGUMENTS` should contain a GitHub issue reference and credentials in one of two formats:

**API Key authentication:**
```
griptape-ai/griptape-nodes#4176 --key sk-xxx
```

**Client credentials authentication (OAuth/JWT):**
```
griptape-ai/griptape-nodes#4176 --client-id abc123 --client-secret xyz789
```

The sub-skills (`/api-research`, `/impl-proxy-client`, `/impl-proxy-node`) will automatically detect which authentication pattern is used based on the credentials files present in the spec directory.

## Phase 1: Research the API

Launch a subagent to run the `/api-research` skill:

```
Use the Agent tool to launch a general-purpose subagent with this prompt:

"Run the /api-research skill with arguments: $ARGUMENTS

Use the Skill tool to invoke it. When it completes, read the generated spec file
at .scratch/proxy-spec-<service-name>/spec.md and report back:
1. The spec file path
2. The service name
3. The model IDs found
4. Whether it's sync or async
5. The full contents of the Request/Response Examples section (so we can verify real data)"
```

After the subagent completes, read the spec file yourself and verify:
- All sections are populated
- Model IDs are defined **and were verified with a successful API call** (the Request/Response Examples section should contain real response data, not placeholders)
- The endpoint path was verified with a successful API call
- Classification (sync/async) is determined

**Gate check:** If the spec's Request/Response Examples section does not contain a real successful response, do NOT proceed to Phase 2. The spec must be backed by at least one verified successful API call to avoid building on incorrect assumptions.

Note the spec file path for the next phases.

## Phase 2: Implement the Proxy Client

Launch a subagent to run the `/impl-proxy-client` skill:

```
Use the Agent tool to launch a general-purpose subagent with this prompt:

"Run the /impl-proxy-client skill with arguments: .scratch/proxy-spec-<service-name>/spec.md

Use the Skill tool to invoke it. When it completes, report back:
1. Whether the local proxy test succeeded
2. The griptape-cloud PR URL
3. Whether the review checklist passed
4. Whether local infrastructure is still running
5. Any issues encountered and how they were resolved"
```

After the subagent completes, verify:
- The local proxy test succeeded
- The review checklist passed with no discrepancies
- The PR was created
- Local infrastructure is still running (needed for Phase 3)

Note the griptape-cloud PR URL.

## Phase 3: Implement the Node

Launch a subagent to run the `/impl-proxy-node` skill:

```
Use the Agent tool to launch a general-purpose subagent with this prompt:

"Run the /impl-proxy-node skill with arguments: .scratch/proxy-spec-<service-name>/spec.md

Use the Skill tool to invoke it. When it completes, report back:
1. Whether the integration test passed
2. The griptape-nodes-library-standard PR URL
3. Whether the review checklist passed
4. Any issues encountered and how they were resolved"
```

After the subagent completes, verify:
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
    - control_plane/api/griptapecloud/components/proxy/clients/<provider>.py (new or modified)
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
3. Re-run the failed phase's subagent (the skills are idempotent for the implementation steps)
4. Always ensure local infra is torn down when done: `cd ../../griptape-cloud && make down`
