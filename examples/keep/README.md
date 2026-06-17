# docsync example: Keep developer docs

This directory holds a ready-to-use **docsync** manifest and config for the
[Keep](https://keephq.dev) developer documentation
(`keep-developer-docs`, a Mintlify `.mdx` site).

## What these files are

| File | Maps to | Purpose |
| --- | --- | --- |
| `manifest.yml` | `Manifest` (`src/docsync/models.py`) | The page ↔ source-of-truth mapping. Each page lists the source repos / file globs / code symbols that, when changed by a diff, should route docsync to that page. |
| `config.yml` | `DocsyncConfig` | Models, thresholds, reviewers, and the embedding stopword list. |

docsync uses these to take a code diff in one of the four Keep service repos
(`keephq/keep-api-gateway`, `keephq/keep-event-handler`, `keephq/keep-workflows`,
`keephq/keep-ui`) and decide which doc pages it can affect — first by **anchor**
matches (the curated globs/symbols here), then by embedding + judge for anything
the manifest doesn't cover.

## How to install into the docs repo

docsync reads its config from a `.docsync/` directory at the root of the docs
repo. Copy these two files there:

```bash
mkdir -p keep-developer-docs/.docsync
cp docsync/examples/keep/manifest.yml keep-developer-docs/.docsync/manifest.yml
cp docsync/examples/keep/config.yml   keep-developer-docs/.docsync/config.yml
```

Resulting layout (see `src/docsync/config.py`):

```
keep-developer-docs/
└── .docsync/
    ├── config.yml          # DocsyncConfig
    ├── manifest.yml         # Manifest (the page ↔ source mapping)
    └── state/cursors.json   # created/committed by the Action — last head_sha per repo
```

Commit them. `state/cursors.json` is created automatically (it tracks the last
processed `head_sha` per source repo for idempotency) — you don't author it.

## Pages mapped (and what part of each they cover)

The five highest-drift pages plus two verified tier-2 pages:

| Page | Anchored to (repo · file · key symbols) |
| --- | --- |
| `services/api-gateway.mdx` | `keep-api-gateway` · `routes/router_setup.py` (`setup_routers`), `config/config.py` (`AUTH_TYPE`, `CONSUMER`, `KEEP_*`, `on_starting`), `routes/alerts.py` (`receive_event`), `services/producers/**` (`EventProducer`, `KafkaEventProducer`, `EventSubscriber`) |
| `services/event-handler.mdx` | `keep-event-handler` · `core/metrics.py` (`events_in_counter`, `events_out_counter`, `events_error_counter`, `processing_time_summary`), `config/consts.py` (`KAFKA_*`, `MAX_PROCESSING_RETRIES`), `consumer_main.py` (`PROMETHEUS_METRICS_PORT`, `HEALTH_CHECK_PORT`), `controllers/event_controller.py` (`process_event_sync`), `alert_deduplicator/**` |
| `operations/authentication.mdx` | `keep-api-gateway` · `services/identity_manager/identitymanagerfactory.py` (`IdentityManagerTypes`, `IdentityManagerFactory`, `get_auth_verifier`, `_manager_cache`), `authverifierbase.py` (`AuthVerifierBase`), `authenticatedentity.py` (`AuthenticatedEntity`), `rbac.py` (`SCOPES`, `has_scopes`); plus `keep-event-handler` thin re-export of `IdentityManagerTypes` |
| `components/providers.mdx` | `keep-workflows` · `providers/providers_factory.py` (`ProvidersFactory`, `get_provider_class`, `get_consumer_providers`), `providers/providers_service.py` (`ProvidersService`, `install_provider`), `workflowmanager/workflowmanager.py` (`PREMIUM_PROVIDERS`, `_check_premium_providers`), `providers/http_provider/http_provider.py` (`BLACKLISTED_ENDPOINTS`) |
| `architecture/data-flow.mdx` | `keep-event-handler` · `models/alert.py` (`AlertDto`, `AlertStatus`, `AlertSeverity`), `event_management/process_event_task.py` (`process_event`, `KEEP_STORE_RAW_ALERTS`, `KEEP_ALERT_FIELDS_ENABLED`, `KEEP_MAINTENANCE_WINDOWS_ENABLED`, `KEEP_CALCULATE_START_FIRING_TIME_ENABLED`); plus `keep-api-gateway` ingestion route/producers |
| `components/workflow-engine.mdx` *(tier 2)* | `keep-workflows` · `workflowmanager/workflowscheduler.py` (`WorkflowScheduler`, `MAX_WORKERS`, `WorkflowStrategy`), `workflowmanager.py` (`WorkflowManager`, `insert_events`) |
| `services/workflows.mdx` *(tier 2)* | `keep-workflows` · `workflowmanager/**`, `main.py` (`WorkflowManager`, `WorkflowScheduler`, `insert_events`) |

> Note: the event-handler source lives under `src/` (e.g. `src/core/metrics.py`,
> `src/models/alert.py`), even though some prose in the `.mdx` pages shows
> repo-root paths like `core/metrics.py` / `models/event_dto.py`. The globs in
> `manifest.yml` use the real on-disk `src/...` paths so they match actual diffs.

## Validate

```bash
cd docsync && .venv/bin/python -c "
from docsync.config import Manifest, DocsyncConfig
from ruamel.yaml import YAML
y=YAML(typ='safe')
Manifest.model_validate(y.load(open('examples/keep/manifest.yml')))
DocsyncConfig.model_validate(y.load(open('examples/keep/config.yml')))
print('ok')
"
```
