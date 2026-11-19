# Configuration

authtracker is configured **only with environment variables**. There is no config file. Everything has a
default, so nothing is required to start.

Use variables to set *defaults for a machine, container or CI job*. Anything you pass in code
(`at.init(save_code=False)`, `at.configure(Settings(...))`) wins over the variable.

```bash
authtracker config show     # the resolved settings (credentials in URIs are masked)
authtracker doctor          # checks that the backend is reachable
```

A bad value (`AUTHTRACKER_CAPTURE_ENV=maybe`) stops with a clear error. An unknown `AUTHTRACKER_*` name
gives a warning, which catches typos.

Booleans accept `1/true/yes/on` and `0/false/no/off` (any case). Lists are comma separated.

## Where runs go

| Variable | Default | Description |
|---|---|---|
| `AUTHTRACKER_BACKEND` | `mlflow` | `mlflow`, `wandb`, `local`, or the name of an installed plugin. |
| `AUTHTRACKER_TRACKING_URI` | the platform's default | MLflow: a server URL (`https://mlflow.example.com`) or a store (`sqlite:///mlflow.db`). Falls back to `MLFLOW_TRACKING_URI`. W&B: base URL of a self-hosted server. `local`: the folder the JSON store uses (default `.authtracker`). |
| `AUTHTRACKER_REGISTRY_URI` | the tracking URI | The model registry, if it lives somewhere else. |

With MLflow and no URI, runs go to a `mlflow.db` file in the current folder.

## Who and what

| Variable | Default | Description |
|---|---|---|
| `AUTHTRACKER_OWNER` | `$USER` | The `owner` tag on every run. Resuming only continues runs of the same owner. |
| `AUTHTRACKER_IMAGE` | none | Container image the job runs in, ideally with a digest (`registry/app@sha256:...`). Same as `image=` in `at.init`. |

## Rules (naming and promotion)

Off unless you set them. They refuse a bad run **before** anything is logged.

| Variable | Default | Description |
|---|---|---|
| `AUTHTRACKER_REQUIRED_TAGS` | none | Tags every run must have, for example `owner,dataset_version`. |
| `AUTHTRACKER_ALLOWED_MODELS` | any | Allowed model names. Others raise `ValidationError`. |
| `AUTHTRACKER_METRIC_NAMES` | any | Allowed metric names. Others raise `ValidationError`. |
| `AUTHTRACKER_PROMOTE_METRIC` | none | The metric the promotion rules look at (higher is better). Rules are off while this is unset. |
| `AUTHTRACKER_PROMOTE_MIN` | none | The version's metric must be at least this to be promoted. |
| `AUTHTRACKER_PROMOTE_MUST_BEAT_PROD` | `false` | The version must beat the one that holds the alias now. A tie does not count. |
| `AUTHTRACKER_GATED_ALIASES` | `production` | The aliases the promotion rules apply to. `staging` and others are free. |

## What a run records

| Variable | Default | Description |
|---|---|---|
| `AUTHTRACKER_CAPTURE_ENV` | `true` | Git commit, Python and package versions, and a `requirements.txt`. |
| `AUTHTRACKER_SAVE_CODE` | `true` | Save the uncommitted changes (`code_changes.patch`) and the running script. A file that looks like it holds a secret is skipped. |
| `AUTHTRACKER_CAPTURE_LOGS` | `true` | Save everything the run prints as `output.log` (MLflow and `local`; W&B records this itself). |
| `AUTHTRACKER_LOG_UPLOAD_INTERVAL` | `60` | Seconds between uploads of `output.log` while the run is open. `0` uploads only at the end. |
| `AUTHTRACKER_SYSTEM_METRICS` | `true` | Chart CPU, memory, disk, network and NVIDIA GPU use (MLflow only). GPU charts appear when an NVIDIA GPU and driver are present. `at.init(system_metrics=False)` wins. |

## Alerts (optional)

A JSON message POSTed to a webhook URL when something you chose to hear about happens. Any URL that accepts JSON works,
including the incoming webhooks of chat tools. **With no URL, nothing is sent.**

| Variable | Default | Description |
|---|---|---|
| `AUTHTRACKER_ALERT_WEBHOOK` | none | The webhook URL (`http://` or `https://`). Optional: leave it unset and alerts are off. |
| `AUTHTRACKER_ALERT_ON` | `failed,stopped,custom` | Which events are sent, from the list below. Only the events you list are sent. |

| Event | Sent when | Level |
|---|---|---|
| `failed` | a run ends with an error (the error type and its first line are included) | `error` |
| `stopped` | a run is interrupted with Ctrl+C | `warn` |
| `finished` | a run ends well (off by default) | `info` |
| `custom` | your code calls `at.alert(...)` | the level you give |

```bash
export AUTHTRACKER_ALERT_WEBHOOK='https://...'            # a secret, see below
export AUTHTRACKER_ALERT_ON=failed,finished               # only failures and successes, no Ctrl+C or custom messages
```

One alert per experiment, not one per nested run. The message has the model, experiment, owner and run id, and looks
like this:

```json
{"text": "[ERROR] Run failed\nmodel: churn\nexperiment: baseline\nowner: me\nrun_id: ...\nerror: ValueError: loss is NaN",
 "level": "error", "title": "Run failed", "message": "", "model": "churn", "experiment": "baseline", "owner": "me",
 "run_id": "...", "error": "ValueError: loss is NaN", "timestamp": "2026-11-19T12:00:00+00:00"}
```

`text` is the whole message ready to read, which is the field chat tools show. The other keys hold the same
information separately, for anything that parses the JSON.

**The URL is a secret.** Anyone who has it can post to your channel. Keep it in your shell, CI secrets or a secrets
manager, never in code, YAML or git. authtracker only reads it from this variable, masks it in `authtracker config
show` and `doctor`, and never puts it in a log line or an error message. Sending never raises and waits at most
5 seconds, so a broken webhook cannot stop a training job.

Check it with `authtracker test-alert` (it sends even if `custom` is not chosen). For your own messages use
`at.alert("Loss is NaN", "step 1200", "error")`.

## When the server is down

| Variable | Default | Description |
|---|---|---|
| `AUTHTRACKER_RETRIES` | `3` | Retries of a call after a temporary error. |
| `AUTHTRACKER_RETRY_WAIT` | `1.0` | Seconds before the first retry. It doubles each time. |
| `AUTHTRACKER_OFFLINE_QUEUE` | `true` | After the retries, queue writes on disk and upload them later. |
| `AUTHTRACKER_QUEUE_DIR` | `~/.authtracker/queue` | Where the queue lives. `authtracker sync` uploads what is left. |

Saving a model, `register`, `promote`, `load` and `best_run` need the server's answer, so they raise
`BackendError` instead of queuing.

## Variables of other tools

authtracker passes these through to the tools it wraps. They are not `AUTHTRACKER_*` variables.

| Variable | Used by | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow | Used when `AUTHTRACKER_TRACKING_URI` is not set. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_TOKEN` | MLflow | Credentials for a protected server. Keep them out of files and git. |
| `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` | MLflow | Seconds between system-metric samples (default 10). |
| `WANDB_API_KEY` | W&B | Your key. Keep it out of files and git. |
| `WANDB_ENTITY` | W&B | Team or user the project belongs to. |
| `WANDB_MODE` | W&B | `offline` writes runs to `./wandb` with no account; sync later with `wandb sync`. |
| `WANDB_DIR` | W&B | Where offline runs are written. |

## Examples

A training container that talks to a team server and enforces naming:

```bash
export AUTHTRACKER_TRACKING_URI=https://mlflow.example.com
export AUTHTRACKER_REQUIRED_TAGS=owner,dataset_version
export AUTHTRACKER_ALLOWED_MODELS=fraud-detector,churn
export AUTHTRACKER_IMAGE=registry.example.com/team/train@sha256:9f2c...
```

A release job that only promotes a better model:

```bash
export AUTHTRACKER_PROMOTE_METRIC=f1
export AUTHTRACKER_PROMOTE_MIN=0.8
export AUTHTRACKER_PROMOTE_MUST_BEAT_PROD=true
```

Weights & Biases, offline:

```bash
export AUTHTRACKER_BACKEND=wandb
export WANDB_MODE=offline
```

## Setting them from code

For tests, notebooks and scripts. Code wins over the environment.

```python
import authtracker as at

at.set_tracking_uri("https://mlflow.example.com")      # like mlflow.set_tracking_uri: where runs go from now on
```

For any other setting, replace it on the current settings and `configure` them. `configure(None)` goes back to the
environment.

```python
import dataclasses

at.configure(dataclasses.replace(at.get_settings(), promote_metric="f1", promote_min=0.8))
at.configure(None)
```
