"""MLflow backend. Requires: pip install 'authtracker[mlflow]'."""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
from typing import Any, Dict

import mlflow
from mlflow.tracking import MlflowClient

from ..errors import BackendError
from ..settings import Settings
from .base import Backend, TAG_MODEL_URI, TAG_RUN_ID

_FLAVOR_BY_MODULE = {
    "sklearn": "sklearn", "torch": "pytorch", "xgboost": "xgboost",
    "lightgbm": "lightgbm", "transformers": "transformers",
    "tensorflow": "tensorflow", "keras": "keras",
}


def _detect_flavor(model: Any) -> str:
    """Pick the MLflow saver from the model's class or any of its parents.

    Looking at the parents matters: your own `class MyNet(nn.Module)` is defined in your code, but
    its parent comes from torch, so it is a PyTorch model. The class closest to your own wins
    (an `XGBClassifier` is xgboost, even though it also is a scikit-learn estimator).
    """
    for cls in type(model).__mro__:
        top = (cls.__module__ or "").split(".")[0]
        if top in _FLAVOR_BY_MODULE:
            return _FLAVOR_BY_MODULE[top]
    raise BackendError(
        f"Cannot detect an MLflow flavor for {type(model).__name__}; "
        "pass flavor='sklearn'|'pytorch'|... to save()"
    )


@contextlib.contextmanager
def _on_cpu_while_saving(model: Any, flavor: str):
    """A PyTorch model that was trained on a GPU is saved from its CPU weights.

    MLflow checks the model with a CPU example input while saving, which fails if the model sits on
    a GPU (`cuda` or `mps`). So the model is moved to the CPU for the save and put back afterwards,
    in place, so the training code (and its optimizer) carries on as before.
    """
    try:
        import torch
    except ImportError:
        yield
        return
    if flavor != "pytorch" or not isinstance(model, torch.nn.Module):
        yield
        return
    devices = {t.device for t in list(model.parameters()) + list(model.buffers())}
    if len(devices) != 1 or next(iter(devices)).type == "cpu":
        yield                                   # already on the CPU, or spread over several devices
        return
    original = next(iter(devices))
    model.to("cpu")
    try:
        yield
    finally:
        model.to(original)


def _pytorch_save_hint(error: Exception, flavor: str, kwargs: Dict[str, Any]):
    """A plain-English message for the two ways saving a PyTorch model usually goes wrong.

    MLflow saves PyTorch models as a traced graph (`pt2`) by default. That needs an example input,
    and the model must be traceable. Both failures come with long, technical messages.
    """
    if flavor != "pytorch" or "serialization_format" in kwargs:
        return None
    text = str(error)
    if "input_example is required" in text:
        return ("Saving a PyTorch model needs an example input, so it can be traced:\n"
                "    at.log_model(net, input_example=batch.cpu().numpy())\n"
                "If the model cannot be traced, save it with serialization_format='pickle' instead "
                "and ship the file that defines your model class: code_paths=['net.py'].")
    if type(error).__module__.split(".")[0] == "torch" or "pt2" in text:
        first = text.strip().splitlines()[0][:200] if text.strip() else type(error).__name__
        return (f"This model could not be traced with torch.export ({type(error).__name__}: {first}).\n"
                "Typical causes: an `if` that depends on tensor values, or an operation that is not "
                "supported. Save it as a pickle instead, and ship the file that defines your model "
                "class so it can be loaded elsewhere:\n"
                "    at.log_model(net, input_example=batch.cpu().numpy(), serialization_format='pickle', "
                "code_paths=['net.py'])\n"
                "Pickled models can run code when loaded, so only load models you trust.")
    return None


def _quote(v: str) -> str:
    return "'" + str(v).replace("'", "\\'") + "'"


_DATABASE_SCHEMES = ("sqlite", "postgresql", "mysql", "mssql")


def _check_store_support(uri) -> None:
    """A database URI needs the full MLflow; the light `mlflow-skinny` can only talk to a server."""
    if uri and uri.split(":", 1)[0].split("+", 1)[0] in _DATABASE_SCHEMES \
            and importlib.util.find_spec("sqlalchemy") is None:
        raise BackendError(
            f"The tracking URI {uri!r} is a database, which needs the full MLflow, but only "
            "mlflow-skinny is installed. Point AUTHTRACKER_TRACKING_URI at an MLflow server "
            "(http://...), or install the full package: pip install 'authtracker[mlflow]'")


class MlflowBackend(Backend):
    _warned_psutil = False

    def __init__(self, settings: Settings):
        _check_store_support(settings.tracking_uri or settings.registry_uri)
        if settings.tracking_uri:
            mlflow.set_tracking_uri(settings.tracking_uri)
        registry = settings.registry_uri or settings.tracking_uri
        if registry:
            mlflow.set_registry_uri(registry)
        self._settings = settings
        self.client = MlflowClient()

    def ping(self) -> None:
        self.client.search_experiments(max_results=1)

    _MARKERS = ("Max retries exceeded", "Connection refused", "Connection aborted",
                "Failed to establish a new connection", "timed out", "Bad Gateway",
                "Service Unavailable", "Gateway Time-out")

    def is_transient(self, exc: Exception) -> bool:
        if super().is_transient(exc):
            return True
        if isinstance(exc, mlflow.exceptions.MlflowException):
            return (exc.error_code == "TEMPORARILY_UNAVAILABLE"
                    or any(m in str(exc) for m in self._MARKERS))
        return False

    # ---- runs (fluent API for start/end so nested runs and log_model work)
    def start_run(self, experiment, run_name, tags, parent_id):
        if parent_id is None:
            mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name, nested=parent_id is not None, tags=tags,
                               **self._system_metrics_args())
        return run.info.run_id

    def _system_metrics_args(self) -> Dict[str, Any]:
        """CPU / memory / disk / network (and GPU) charts, if asked for and possible.

        MLflow creates the run first and only then fails when `psutil` is missing, which would
        crash the training script and leave a run stuck as "running". So check before starting.
        """
        if not self._settings.system_metrics:
            return {"log_system_metrics": False}      # explicit, so MLflow's own environment variable cannot turn it on
        if importlib.util.find_spec("psutil") is None:
            if not MlflowBackend._warned_psutil:
                MlflowBackend._warned_psutil = True
                logging.getLogger("authtracker").warning(
                    "System metrics are enabled but `psutil` is not installed, so they are "
                    "skipped. Install it with: pip install psutil (or set AUTHTRACKER_SYSTEM_METRICS=false)")
            return {"log_system_metrics": False}
        return {"log_system_metrics": True}

    def download_artifact(self, run_id, name, folder):
        if name not in [a.path for a in self.client.list_artifacts(run_id)]:
            return None
        return self.client.download_artifacts(run_id, name, folder)

    def find_run(self, experiment, name, tags, parent_id):
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            return None
        parts = [f"tags.`mlflow.runName` = {_quote(name)}"] + [f"tags.`{k}` = {_quote(v)}" for k, v in tags.items()]
        if parent_id:
            parts.append(f"tags.`mlflow.parentRunId` = {_quote(parent_id)}")
        runs = self.client.search_runs([exp.experiment_id], " and ".join(parts),
                                       order_by=["attributes.start_time DESC"], max_results=1)
        return runs[0].info.run_id if runs else None

    def resume_run(self, run_id, parent_id):
        run = mlflow.start_run(run_id=run_id, nested=parent_id is not None, **self._system_metrics_args())
        return run.info.run_id

    def last_step(self, run_id):
        steps = [m.step for m in getattr(self.client.get_run(run_id).data, "_metric_objs", [])]
        return max(steps) if steps else None

    def end_run(self, run_id, status):
        active = mlflow.active_run()
        if active is None or active.info.run_id != run_id:
            self.client.set_terminated(run_id, status)  # not the innermost run
        else:
            mlflow.end_run(status=status)

    def set_tags(self, run_id, tags):
        for k, v in tags.items():
            self.client.set_tag(run_id, k, str(v))

    def log_params(self, run_id, params):
        for k, v in params.items():
            self.client.log_param(run_id, k, v)

    def log_metrics(self, run_id, metrics, step):
        for k, v in metrics.items():
            self.client.log_metric(run_id, k, float(v), step=step or 0)

    def log_artifact(self, run_id, path):
        self.client.log_artifact(run_id, path)

    def log_model(self, run_id, model, flavor, **kwargs):
        active = mlflow.active_run()
        if active is None or active.info.run_id != run_id:
            raise BackendError(
                "log_model must be called on the innermost open run "
                "(call it inside the `with` block of the run that owns the model)"
            )
        flavor = flavor or _detect_flavor(model)
        module = importlib.import_module(f"mlflow.{flavor}")
        # pyfunc takes the model as `python_model=`; other flavors take it positionally.
        args, kw = ((), {"python_model": model}) if flavor == "pyfunc" else ((model,), {})
        try:
            with _on_cpu_while_saving(model, flavor):
                try:
                    info = module.log_model(*args, name="model", **kw, **kwargs)
                except TypeError:  # older MLflow
                    info = module.log_model(*args, artifact_path="model", **kw, **kwargs)
        except Exception as e:
            hint = _pytorch_save_hint(e, flavor, kwargs)
            if hint:
                raise BackendError(hint) from e
            raise
        uri = info.model_uri
        self.client.set_tag(run_id, TAG_MODEL_URI, uri)
        return uri

    def _summary(self, run) -> Dict[str, Any]:
        return {
            "run_id": run.info.run_id,
            "name": run.data.tags.get("mlflow.runName"),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }

    def get_run(self, run_id):
        return self._summary(self.client.get_run(run_id))

    def search_runs(self, experiment, tags, metric, mode, limit):
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            return []
        flt = " and ".join(f"tags.`{k}` = {_quote(v)}" for k, v in tags.items())
        flt = f"{flt} and metrics.`{metric}` >= -1e308" if flt else f"metrics.`{metric}` >= -1e308"
        order = f"metrics.`{metric}` {'DESC' if mode == 'max' else 'ASC'}"
        runs = self.client.search_runs([exp.experiment_id], filter_string=flt,
                                       order_by=[order], max_results=limit)
        return [self._summary(r) for r in runs]

    def _log_logged_model(self, run_id, local_dir, model_type):
        """A folder kept as it is, inside an MLflow 'logged model' (so it can be registered), and linked to the run
        so the run's page lists it, like a model saved by MLflow itself."""
        from mlflow.entities import LoggedModelOutput
        exp_id = self.client.get_run(run_id).info.experiment_id
        lm = self.client.create_logged_model(experiment_id=exp_id, name="model", source_run_id=run_id,
                                             model_type=model_type)
        try:
            self.client.log_model_artifacts(lm.model_id, local_dir)
        except Exception:
            self.client.finalize_logged_model(lm.model_id, "FAILED")
            raise
        self.client.finalize_logged_model(lm.model_id, "READY")
        self.client.log_outputs(run_id, [LoggedModelOutput(model_id=lm.model_id, step=0)])
        uri = f"models:/{lm.model_id}"
        self.client.set_tag(run_id, TAG_MODEL_URI, uri)
        return uri

    def log_model_dir(self, run_id, local_dir):
        from .. import bundle
        return self._log_logged_model(run_id, local_dir, bundle.MODEL_TYPE)

    def log_model_files(self, run_id, local_dir):
        from .. import bundle
        return self._log_logged_model(run_id, local_dir, bundle.FILES_TYPE)

    def _download_model_of_type(self, name, alias, model_type, dst):
        try:
            source = self.client.get_model_version_by_alias(name, alias).source
        except mlflow.exceptions.MlflowException:
            return None
        if not source.startswith("models:/m-"):
            return None
        if self.client.get_logged_model(source.split("/", 1)[1]).model_type != model_type:
            return None
        import tempfile
        return mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{name}@{alias}", dst_path=dst or tempfile.mkdtemp(prefix="authtracker-load-"))

    def download_bundle(self, name, alias, dst=None):
        from .. import bundle
        return self._download_model_of_type(name, alias, bundle.MODEL_TYPE, dst)

    def download_model_files(self, name, alias, dst=None):
        from .. import bundle
        return self._download_model_of_type(name, alias, bundle.FILES_TYPE, dst)

    # ---- moving a model between stores
    def export_model(self, name, dest):
        from . import mlflow_transfer
        return mlflow_transfer.export_model(self.client, name, dest)

    def import_model(self, src, name=None):
        from . import mlflow_transfer
        return mlflow_transfer.import_model(self.client, src, name)

    def export_all(self, dest):
        from . import mlflow_transfer
        return mlflow_transfer.export_all(self.client, dest)

    def import_all(self, src):
        from . import mlflow_transfer
        return mlflow_transfer.import_all(self.client, src)

    # ---- registry
    def register(self, run_id, name, model_uri):
        mv = mlflow.register_model(model_uri, name)
        self.client.set_model_version_tag(name, mv.version, TAG_RUN_ID, run_id)
        return str(mv.version)

    def set_alias(self, name, version, alias):
        self.client.set_registered_model_alias(name, alias, version)

    def get_alias(self, name, alias):
        try:
            return str(self.client.get_model_version_by_alias(name, alias).version)
        except mlflow.exceptions.MlflowException:
            return None

    def version_run_id(self, name, version):
        mv = self.client.get_model_version(name, str(version))
        return mv.tags.get(TAG_RUN_ID) or mv.run_id

    def load(self, name, alias):
        return mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
