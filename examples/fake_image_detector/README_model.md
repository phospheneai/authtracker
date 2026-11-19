# fake-image classifier

Decides whether an image is **real** or **fake**.

## Input and output

- Input: path to an image file (any size; resized to 32 x 32, RGB).
- Output: `{"label": "real" | "fake", "confidence": 0-1, "probabilities": {...}}`

## Use with MLflow (recommended)

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/fake-image@production")
model.predict({"image_path": ["photo.jpg"]})
```

## Use without MLflow

Download the model folder, then:

```bash
pip install -r requirements.txt
python <model>/code/infer_single.py <model>/artifacts/weights.pt photo.jpg
```

## Notes

- Trained on toy data for demonstration. Do not use for real decisions.
- Architecture: `arch/network.py`. Preprocessing: `arch/preprocess.py`.
