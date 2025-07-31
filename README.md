# MLOps Pipeline - Linear Regression on California Housing

This repository implements a full MLOps pipeline in a single branch:
- **Model**: Scikit-learn Linear Regression
- **Dataset**: California Housing
- **Pipeline**: Training → Testing → Quantization → Dockerization → CI/CD (GitHub Actions)

## Run Locally
```bash
pip install -r requirements.txt
python src/train.py
pytest src/test_model.py
python src/quantize.py
python src/inference.py
