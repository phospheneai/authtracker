"""MLflow wrapper: gives developers a single predict() call."""
import mlflow.pyfunc
import torch


def _paths(model_input):
    """Accept a DataFrame, dict, list or single string; return a list of image paths."""
    if isinstance(model_input, str):
        return [model_input]
    if hasattr(model_input, "to_dict"):                      # pandas DataFrame
        return list(model_input["image_path"])
    if isinstance(model_input, dict):
        value = model_input["image_path"]
        return [value] if isinstance(value, str) else list(value)
    return list(model_input)


class FakeImageModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from arch.network import FakeImageNet
        self.net = FakeImageNet()
        state = torch.load(context.artifacts["weights"], map_location="cpu")
        self.net.load_state_dict(state)
        self.net.eval()

    def predict(self, context, model_input, params=None):
        from infer_single import infer_single
        return [infer_single(self.net, p) for p in _paths(model_input)]
    