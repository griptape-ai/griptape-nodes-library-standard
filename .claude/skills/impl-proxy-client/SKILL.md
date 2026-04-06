---
name: impl-proxy-client
description: Implement a GTC proxy client from a spec file. Works in the griptape-cloud repo.
argument-hint: <spec-file-path>
allowed-tools: Bash Read Write Edit Grep Glob
disable-model-invocation: false
---

# Implement a GTC Proxy Client

Implement a proxy client in the griptape-cloud repo based on the specification file produced by `/api-research`.

The griptape-cloud repo is located at `../../griptape-cloud` relative to this repo root (both repos live under the same parent directory).

## 1. Read the Spec and API Key

Read the spec file from `$ARGUMENTS`. Also read the API key from `.api_key` in the same directory.

```bash
SPEC_DIR=$(dirname "$ARGUMENTS")
API_KEY=$(cat "$SPEC_DIR/.api_key")
```

Extract from the spec:
- Service name, base URL, auth method, auth header
- Model IDs
- Sync vs Async classification
- Endpoint schemas
- Billing info (metric, suggested ActivityType, ServiceModelConfigType)
- Error format

## 2. Set Up in the Cloud Repo

```bash
cd ../../griptape-cloud
git checkout main && git pull
git checkout -b feat/<service-name>-proxy-client
```

## 3. Read Reference Implementations

Based on the spec's classification, read the most similar existing client:

- **Sync returning bytes**: Read `control_plane/api/griptapecloud/components/proxy/clients/elevenlabs.py` (focus on `ElevenLabsMusicClient`)
- **Sync returning JSON**: Read `control_plane/api/griptapecloud/components/proxy/clients/elevenlabs.py` (focus on `ElevenLabsTtsClient`)
- **Async with polling**: Read `control_plane/api/griptapecloud/components/proxy/clients/openai.py` or `control_plane/api/griptapecloud/components/proxy/clients/xai.py`
- **JWT-based auth**: Read `control_plane/api/griptapecloud/components/proxy/clients/kling.py`

Also read the base classes in `control_plane/api/griptapecloud/components/proxy/clients/client.py` to confirm the abstract interface:

- `ProxyClient`: `create_generation()`, `get_billing_activity_type_and_volume_from_request()`, `get_billing_activity_type_and_volume_from_response()`, `get_moderation_objects()`, `register_user_auth_info()`, `get_model_ids()`
- `AsyncProxyClient` (extends ProxyClient): adds `get_generation()`, `fetch_completed_generation()`

## 4. Add ActivityType Entries

Edit `control_plane/api/griptapecloud/components/credits/activities.py`.

Add new `ActivityType` enum values following the naming pattern from the spec's "Suggested ActivityType Name". Look at existing entries for the naming convention.

## 5. Add ServiceModelConfigProvider Entry

If this is a new provider (not already in the enum), edit `control_plane/api/griptapecloud/components/service_model_config/models.py`.

Add a new `ServiceModelConfigProvider` choice tuple. Follow the existing pattern.

## 6. Create the Proxy Client

Create a new file at `control_plane/api/griptapecloud/components/proxy/clients/<provider_name>.py`.

### Required Implementation

```python
from griptapecloud.components.proxy.clients.client import (
    ProxyClient,  # or AsyncProxyClient
    ProxyClientError,
    ProxyProviderError,
    ProxyInputError,
    ModerationObject,
    TextModerationObject,
    ImageModerationObject,
    emit_credit_exhaustion_event,
)
from griptapecloud.components.service_model_config.models import (
    ServiceModelConfig,
    ServiceModelConfigType,
    ServiceModelConfigProvider,
)
from griptapecloud.components.credits.activities import ActivityType
```

### Methods to implement:

**`__init__(self, model_id: str)`**
- Store `model_id`
- Fetch API key from `ServiceModelConfig.objects.filter(model_type=..., model_name=model_id, active=True).first()`
- Raise `ProxyClientError("unconfigured proxy client")` if not found

**`get_model_ids(cls) -> list[str]`** (classmethod)
- Return the model ID strings from the spec's "Model IDs" section

**`create_generation(self, **kwargs) -> tuple[str | None, dict | bytes]`**
- Make the HTTP request to the upstream API using `requests.post()`
- Call `self._raise_for_provider_response(response)` before processing
- For sync APIs: return `(None, response.json())` or `(None, response.content)`
- For async APIs: return `(job_id, response.json())`

**`get_billing_activity_type_and_volume_from_request(self, **kwargs) -> tuple[ActivityType, int]`**
- Calculate billing volume from the request kwargs based on the spec's billing metric
- Return `(ActivityType.<YOUR_TYPE>, volume)`

**`get_billing_activity_type_and_volume_from_response(self, provisional_activity_type, provisional_volume, **kwargs) -> tuple[ActivityType, int]`**
- Adjust billing based on actual response if needed
- Most implementations just return `(provisional_activity_type, provisional_volume)`

**`get_moderation_objects(self, **kwargs) -> list[ModerationObject]`**
- Return `[TextModerationObject(kwargs["prompt"])]` for text prompts
- Add `ImageModerationObject(data_uri)` for image inputs

**`register_user_auth_info(self, user_auth_info: str)`**
- Override `self.api_key` with the user-provided key (BYOK support)

**`_raise_for_provider_response(self, response)`**
- Follow the EXACT standard error pattern:
```python
def _raise_for_provider_response(self, response) -> None:
    if response.status_code == HTTPStatus.PAYMENT_REQUIRED:
        emit_credit_exhaustion_event(
            provider=ServiceModelConfigProvider.<PROVIDER>[0],
            model_id=self.model_id,
            status_code=response.status_code,
        )
        logger.error(str(response.status_code) + " " + response.text)
        raise ProxyClientError("proxy client error")

    if response.status_code >= 500:
        logger.error(str(response.status_code) + " " + response.text)
        raise ProxyProviderError("model provider error")

    if response.status_code in [400, 422]:
        logger.warning(str(response.status_code) + " " + response.text)
        raise ProxyInputError(response.text)

    if response.status_code >= 400:
        logger.error(str(response.status_code) + " " + response.text)
        raise ProxyClientError("proxy client error")
```

**If AsyncProxyClient, also implement:**

**`get_generation(self, job_id: str) -> dict | bytes`**
- Poll the provider's status endpoint
- Return the status response

**`fetch_completed_generation(self, job_id: str) -> bytes`**
- Check if the generation is complete
- If still in progress: raise `IncompleteGenerationError`
- If failed: raise `ProxyProviderError`
- If complete: fetch and return the result as bytes

## 7. Register in the Factory

Edit `control_plane/api/griptapecloud/components/proxy/views.py`:

1. Add the import at the top of the file
2. Add the client class to the `proxy_classes` list in `make_proxy_client()` (around line 544)
3. If async: also add to `async_proxy_classes` in `make_async_proxy_client()` (around line 607)

## 8. Register in V2 Tasks (Sync Clients Only)

If the client is **synchronous** (returns `job_id=None` from `create_generation()`):

Edit `control_plane/api/griptapecloud/components/proxy/v2/tasks.py`:

1. Add the import at the top
2. Add the client class to the `PROXY_CLIENTS` list (around line 65)

Async clients do NOT need to be added here.

## 9. Commit Changes

```bash
git add -A
git commit -m "feat: add <service-name> proxy client"
```

## 10. Run Django Migrations

If you added new enum values to `ServiceModelConfigProvider` or any model fields:

```bash
cd control_plane/api
python manage.py makemigrations
python manage.py migrate
```

If migrations were generated, commit them:

```bash
git add -A
git commit -m "feat: add <service-name> migrations"
```

## 11. Start Local Infrastructure

First check if containers are already running from a previous session:

```bash
docker compose ps
```

If containers are already running, you can skip `make up/debug`. If ports are in conflict (e.g. postgres on 54320), stop the existing containers first with `make down` before starting fresh.

If no containers are running:

```bash
make up/debug
```

Wait for all services to be healthy. The local setup includes:
- Django API at `http://localhost:8000`
- PostgreSQL, Redis, MinIO
- Celery workers for async task processing
- Auth is disabled (auto-login)

Verify the server is up:
```bash
curl -s http://localhost:8000/api/ | head -20
```

### Ensure MinIO bucket exists

For async clients that store generation results, the `proxy-generations` bucket must exist in MinIO. Create it if missing:

```python
import boto3
s3 = boto3.client("s3", endpoint_url="http://localhost:9000", aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin")
try:
    s3.head_bucket(Bucket="proxy-generations")
except:
    s3.create_bucket(Bucket="proxy-generations")
```

## 12. Create DB Records

Use the Django shell to create the required database records. The management command path varies by container setup:

```bash
# Try this first:
docker compose exec web python manage.py shell
# If that fails (e.g. "No such file or directory"), use the full path:
docker exec griptape-debug-web-1 bash -c "cd /opt/webapp/api && uv run python manage.py shell"
```

In the shell, run:

```python
from griptapecloud.components.service_model_config.models import (
    ServiceModelConfig,
    ServiceModelConfigType,
    ServiceModelConfigAuthDetails,
    ServiceModelConfigProvider,
)
from griptapecloud.components.entitlements.models import EntitlementLimits

# Ensure model proxy is enabled for the default entitlement
for el in EntitlementLimits.objects.all():
    el.can_use_model_proxy = True
    el.save()

# Create auth details
auth_details = ServiceModelConfigAuthDetails.objects.create(
    provider=ServiceModelConfigProvider.<PROVIDER>,
    api_key="<API_KEY_FROM_.api_key_FILE>",
)

# Create model config for each model ID
for model_id in [<list of model IDs from spec>]:
    ServiceModelConfig.objects.create(
        model_name=model_id,
        model_type=ServiceModelConfigType.<TYPE_FROM_SPEC>,
        auth_details=auth_details,
        active=True,
    )
```

Replace the placeholders with actual values from the spec.

## 13. Test the Proxy Locally

Make a request through the local proxy to verify end-to-end:

```bash
# Submit a generation
curl -X POST http://localhost:8000/api/proxy/v2/models/<model-id> \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test prompt", ...minimal params from spec...}'

# Response should contain generation_id
# Poll for status
curl http://localhost:8000/api/proxy/v2/generations/<generation-id>

# Fetch result when completed
curl http://localhost:8000/api/proxy/v2/generations/<generation-id>/result
```

If the test fails, investigate the error. Check Django logs via `docker compose logs web -f` and Celery worker logs via `docker compose logs workers -f`.

Fix any issues before proceeding.

## 14. Review: Verify Client Against Spec

Re-read the spec file and compare it against the implemented client code. Check:

**Endpoints and headers:**
- [ ] The base URL and endpoint path in `create_generation()` match the spec
- [ ] All required headers (auth, content-type, async headers) are included
- [ ] The poll/status endpoint in `get_generation()` matches the spec

**Request building:**
- [ ] `create_generation()` sends the request body in the format the spec documents
- [ ] No fields are added that aren't in the spec
- [ ] No required fields from the spec are missing

**Response parsing:**
- [ ] `fetch_completed_generation()` extracts data from the correct JSON paths
- [ ] Terminal state names (SUCCEEDED, FAILED, etc.) match the spec exactly
- [ ] In-progress state names match the spec

**Billing:**
- [ ] `get_billing_activity_type_and_volume_from_request()` maps each model ID to the correct ActivityType
- [ ] The volume calculation matches the spec's billing metric

**Model IDs:**
- [ ] `get_model_ids()` returns exactly the model IDs from the spec
- [ ] The model constants at the top of the file match

Fix any discrepancies before proceeding.

## 15. Create a PR

```bash
git push -u origin feat/<service-name>-proxy-client
gh pr create \
  --title "feat: add <service-name> proxy client" \
  --body "$(cat <<'EOF'
Adds a proxy client for <service name> to support <media type> generation through the Griptape Cloud proxy.

Implements `<ClientClassName>` in `proxy/clients/<provider>.py` handling <sync/async> requests to the <service name> API. Registers the client in the proxy factory for model IDs: <list model IDs>. Adds `ActivityType` entries for billing.

## Sources

- <link to the API documentation page>
- <any other references consulted (SDK repos, blog posts, etc.)>
- <note any nuances, quirks, or workarounds discovered during implementation>

## Testing

Tested end-to-end locally:
1. Created `ServiceModelConfig` and `ServiceModelConfigAuthDetails` DB records
2. Submitted generation via `POST /api/proxy/v2/models/<model-id>`
3. Polled via `GET /api/proxy/v2/generations/<generation-id>`
4. Fetched result via `GET /api/proxy/v2/generations/<generation-id>/result`

<Include any relevant details about what was tested, what worked, what was tricky.>

## Production setup

After merging, create the following DB records in production:
- `ServiceModelConfigAuthDetails` for the <PROVIDER> provider with the API key
- `ServiceModelConfig` entries for each model ID: <list model IDs>
- `ActivityCreditCost` entries for: <list ActivityType entries>

Closes <issue-reference>
EOF
)"
```

Print the PR URL.

Note: Leave the local infrastructure running if `/impl-proxy-node` will run next.
