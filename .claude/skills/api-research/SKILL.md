---
name: api-research
description: Research a new API service from a GitHub issue, test endpoints with live credentials, and produce a structured specification file.
argument-hint: <owner/repo#issue> --key <api-key>
allowed-tools: Bash Read Write Grep Glob WebFetch
disable-model-invocation: false
---

# Research and Live-Test a New API Service

Research the API service described in the GitHub issue and produce a structured specification that downstream skills (`/impl-proxy-client` and `/impl-proxy-node`) can consume.

## 1. Parse Arguments

Extract from `$ARGUMENTS`:
- **Issue reference**: e.g. `griptape-ai/griptape-nodes#4176` (the part before `--key`)
- **API key**: the value after `--key`

If either is missing, stop and ask the user.

## 2. Fetch the GitHub Issue

```bash
gh issue view <number> -R <owner/repo> --json title,body,comments,labels
```

Extract the service URL and any documentation links from the issue body and comments.

## 3. Set Up Output Directory

```bash
SERVICE_NAME="<lowercase-hyphenated-service-name>"
mkdir -p .scratch/proxy-spec-$SERVICE_NAME/responses
echo "<api-key>" > .scratch/proxy-spec-$SERVICE_NAME/.api_key
```

The `.scratch/` directory is for temporary working files and MUST NOT be committed. Verify it is in `.gitignore` before proceeding (add it if missing).

Store the API key in `.api_key` so downstream skills can read it without it being embedded in the spec.

## 4. Fetch and Read API Documentation

Use `WebFetch` on the service URL and any docs links found in the issue. Extract and understand:

- Base URL and endpoint paths
- Authentication method (Bearer token, API key header, JWT, OAuth, etc.)
- Request schemas for each endpoint (required/optional fields, types, constraints)
- Response schemas (JSON structure, binary content types)
- Whether the API is synchronous (immediate response) or asynchronous (submit job, poll for status, fetch result)
- If async: polling endpoint, status field names, terminal state values, result fetch endpoint
- Rate limits and quotas
- Error response format and status codes

**Fallback strategies when docs are inaccessible** (e.g. JS-rendered pages that return empty content):
1. Use the `/agent-browser` skill to render JS-heavy documentation pages and extract their text content. This is the most reliable approach for SPA-based doc sites.
2. Search for an OpenAPI/Swagger spec URL (often at `/openapi.json` or similar)
3. Check if the provider has a model listing endpoint (e.g. `/v1/models`) and call it to discover available model IDs programmatically
4. Search GitHub for the provider's official SDK source code, which often contains endpoint URLs, request schemas, and model ID constants
5. Use web search to find blog posts, tutorials, or community documentation
6. Try the provider's API with common endpoint patterns and inspect error messages for clues about the correct schema

**Important:** Do not rely solely on API trial-and-error to determine parameter schemas. Even if a request succeeds, you may miss optional parameters, constraints, or valid value ranges. Always prefer reading the actual documentation (via agent-browser if needed) over reverse-engineering.

## 5. Verify Model IDs Before Proceeding

Before writing exploration tests, verify that the model IDs from the documentation actually work. Many providers use different IDs than what their docs suggest.

1. If the provider has a model listing endpoint, call it and confirm the exact model ID strings.
2. If no listing endpoint exists, make a minimal test request with the documented model ID and check that you don't get a "model not found" error.
3. If the documented model IDs fail, try variations (e.g. with/without version suffixes, different casing, hyphenated vs underscored).

Do NOT proceed to write the spec until you have confirmed at least one model ID works with a successful API call.

## 6. Write and Run Exploration Test Scripts

Create `.scratch/proxy-spec-$SERVICE_NAME/test_api.py` with `requests` or `httpx` and run it with `python3`. The script should perform these tests and save results:

### Required Tests

1. **Minimal success** - Call the primary endpoint with only required parameters. Save the full response (status code, headers, body) to `responses/minimal_success.json`.

2. **Full parameters** - Call with all optional parameters populated with reasonable values. Save to `responses/full_params.json`.

3. **Error: missing required field** - Omit a required field and capture the error response format. Save to `responses/error_missing_field.json`.

4. **Error: invalid value** - Send an invalid value for a parameter (e.g. out-of-range number, wrong type). Save to `responses/error_invalid_value.json`.

5. **Error: bad auth** - Send a request with an invalid API key. Save to `responses/error_bad_auth.json`.

6. **Async polling** (if applicable) - Submit a job, poll until completion, fetch the result. Measure and record:
   - Time from submission to first non-queued status
   - Total time to completion
   - Number of poll attempts
   - Save to `responses/async_full_cycle.json`

### Script Requirements

- Print a summary table after each test: test name, status code, response shape (top-level keys), timing
- Sanitize the API key from ALL saved output (replace with `<REDACTED>`)
- Handle and report errors gracefully (don't crash on one failed test)
- Use the API key from the argument, not hardcoded

Run the script and review the output. If any test fails unexpectedly, investigate and adjust.

## 7. Classify the API

Based on the test results, determine:

- **Sync vs Async**: Does the API return results immediately (`ProxyClient`) or via job polling (`AsyncProxyClient`)?
- **Media category**: image, video, audio, 3d, or text
- **Result format**: JSON containing a URL, JSON containing base64 data, or raw binary bytes
- **Auth method**: How the API key is passed (header name, bearer token, etc.)

## 8. Write the Specification File

Write `.scratch/proxy-spec-$SERVICE_NAME/spec.md` using EXACTLY these section headers (downstream skills grep for them):

```markdown
# {Service Name} Proxy Specification

## Service Info

- **Name**: {service name}
- **Base URL**: {base API URL}
- **Auth Method**: {e.g. "Bearer token via Authorization header", "API key via X-Api-Key header"}
- **Auth Header**: {exact header name and format}
- **Docs URL**: {link to API docs}
- **GitHub Issue**: {issue reference}

## Model IDs

List of model ID strings that will be used in the proxy factory's `get_model_ids()`:

- `{model-id-1}` - {description}
- `{model-id-2}` - {description}

## Classification

- **Sync/Async**: {sync or async}
- **Media Type**: {image, video, audio, 3d, text}
- **Result Format**: {json_url, json_base64, raw_binary}
- **Proxy Base Class**: {ProxyClient or AsyncProxyClient}

## Endpoints

### {Endpoint Name}

- **Method**: {POST/GET}
- **Path**: {/path/to/endpoint}
- **Content-Type**: {application/json, multipart/form-data, etc.}

**Request Schema:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| {field} | {type} | {yes/no} | {default} | {description} |

**Response Schema:**

```json
{example response with field descriptions as comments}
```

## Async Polling

(Include this section only if the API is async)

- **Poll Endpoint**: {method} {path}
- **Status Field**: {field name in response, e.g. "status"}
- **Terminal States**: {e.g. "completed", "failed", "error"}
- **In-Progress States**: {e.g. "pending", "processing"}
- **Result Endpoint**: {method} {path} (if separate from poll endpoint)
- **Typical Completion Time**: {observed from tests}
- **Recommended Poll Interval**: {seconds}

## Billing

- **Metric**: {per-request, per-second, per-megapixel, per-character, etc.}
- **Volume Extraction**: {how to calculate volume from request params, e.g. "duration_seconds field", "count of 1 per request"}
- **Suggested ActivityType Name**: {e.g. "IMAGE_GENERATION_CARTWHEEL", following existing naming pattern}
- **ServiceModelConfigType**: {e.g. "IMAGE_GENERATION", "VIDEO_GENERATION", "AUDIO_GENERATION" - must match existing enum values}

## Error Format

**Standard error response structure:**

```json
{example error response}
```

**Status code mapping:**
- {status_code}: {meaning}

## Request/Response Examples

### Minimal Successful Request

```
{method} {url}
Headers: {headers with key redacted}

{request body}
```

**Response** ({status_code}):

```json
{response body}
```

### Full Parameters Request

```
{method} {url}
Headers: {headers with key redacted}

{request body}
```

**Response** ({status_code}):

```json
{response body}
```

## Quirks

- {Any unexpected behavior, edge cases, or gotchas discovered during testing}
- {e.g. "API returns 200 with error in body instead of 4xx"}
- {e.g. "Polling endpoint returns different structure before vs after completion"}
- {e.g. "File uploads require multipart form data, not JSON"}

## Suggested Node Parameters

### Inputs

| Name | Type | Default | Required | Constraints | Tooltip |
|------|------|---------|----------|-------------|---------|
| {name} | {ParameterString/Int/Float/Bool/Image/etc.} | {default} | {yes/no} | {e.g. min/max, choices} | {description} |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| generation_id | ParameterString | Generation ID from proxy API |
| provider_response | ParameterDict | Raw provider response |
| {media_output} | {ParameterImage/Audio/Video} | Generated media artifact |
```

## 9. Review: Cross-Check Spec Against Docs

Use `/agent-browser` to open the API documentation page and systematically compare the spec against the official docs. For each section, verify:

**Model IDs:**
- [ ] Every model ID in the spec exists in the docs
- [ ] No model IDs in the docs are missing from the spec (unless intentionally excluded)

**Endpoints:**
- [ ] The endpoint path matches the docs exactly
- [ ] The HTTP method matches
- [ ] Required headers are documented (e.g. async headers, content-type)

**Request parameters:**
- [ ] Every parameter in the spec exists in the docs with the same name, type, and constraints
- [ ] Default values match the docs (do not invent defaults)
- [ ] Min/max ranges match the docs
- [ ] No documented parameters are missing from the spec (flag optional ones you chose to exclude)

**Response format:**
- [ ] The response JSON structure matches the docs
- [ ] Field names and nesting are correct

**Quirks:**
- [ ] Any discrepancies between the docs and observed API behavior are noted

Print any discrepancies found and fix them in the spec before finishing.

## 10. Final Checklist

- [ ] All spec sections are filled in with real data from the API tests
- [ ] No API key appears anywhere in the spec or saved responses
- [ ] The model IDs have been verified with a successful API call (not just copied from docs)
- [ ] The endpoint path has been verified with a successful API call
- [ ] The request/response examples match what was actually observed in testing
- [ ] The quirks section captures anything that could trip up implementation
- [ ] The suggested node parameters cover all useful API options

Print the path to the spec file: `.scratch/proxy-spec-$SERVICE_NAME/spec.md`
