# Heart Disease Prediction API

A FastAPI service that predicts the presence of heart disease from 13 clinical features. The model is a scikit-learn classifier saved with `joblib`, packaged with Docker, and deployed on Render.

**Live API:** https://heart-disease-api-z24r.onrender.com
**Interactive docs (Swagger UI):** https://heart-disease-api-z24r.onrender.com/docs

> The service runs on a free Render instance, so it sleeps after a period of inactivity. The first request after a pause can take about a minute.

> **Disclaimer:** This is a learning project. It is not a medical device and must not be used for real diagnosis.

## Tech stack

- Python 3.12, scikit-learn 1.6.1, pandas, joblib
- FastAPI + Pydantic (input validation), Uvicorn
- Docker and Docker Compose
- Render (deployment)

## Dataset

[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (Kaggle, `johnsmith88`).

- 1,025 rows and 13 features, with a binary target (`1` = heart disease present, `0` = absent).
- The raw file contains **723 duplicate rows**. Training on it directly leaks identical rows into the test set and gives an unrealistic 100% accuracy with Random Forest. After removing duplicates, **302 unique rows** remain, and all reported results use this cleaned data.

## Model

| Setting | Value |
|---|---|
| Split | 80/20, stratified, `random_state=42` |
| Candidates | Logistic Regression (with `StandardScaler`), Random Forest |
| Selected model | Logistic Regression (higher test ROC-AUC) |

Test-set results (61 rows):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.803 | 0.800 | 0.848 | 0.824 | 0.871 |
| Random Forest | 0.787 | 0.813 | 0.788 | 0.800 | 0.866 |

The test set is small, so the difference between the two models is not statistically meaningful.

The saved file `model/heart_model.joblib` is a dictionary:

```python
{"model": <sklearn Pipeline>, "features": [...13 feature names...], "name": "LogisticRegression"}
```

## Input features

| Feature | Description | Allowed values |
|---|---|---|
| `age` | Age in years | 1-120 |
| `sex` | 1 = male, 0 = female | 0-1 |
| `cp` | Chest pain type | 0-3 |
| `trestbps` | Resting blood pressure (mm Hg) | 50-250 |
| `chol` | Serum cholesterol (mg/dl) | 50-700 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0-1 |
| `restecg` | Resting ECG result | 0-2 |
| `thalach` | Maximum heart rate achieved | 50-250 |
| `exang` | Exercise-induced angina | 0-1 |
| `oldpeak` | ST depression induced by exercise | 0-10 |
| `slope` | Slope of peak exercise ST segment | 0-2 |
| `ca` | Number of major vessels colored by fluoroscopy | 0-4 |
| `thal` | Thalassemia | 0-3 |

Requests with missing or out-of-range values return `422`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` when the model is loaded |
| GET | `/info` | Model type and the list of expected features |
| POST | `/predict` | Returns `heart_disease` (true/false) and `probability` |

### Example

```bash
curl -X POST https://heart-disease-api-z24r.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

Response:

```json
{"heart_disease": false, "probability": 0.2489}
```

The prediction is `true` when the estimated probability is 0.5 or higher.

## Run locally

### With Docker (recommended)

```bash
docker compose up --build
```

Then open http://localhost:8000/docs.

### Without Docker

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Project structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app and endpoints
│   └── schemas.py        # Pydantic request/response models
├── model/
│   └── heart_model.joblib
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── .gitignore
```

## Deployment

The app is deployed on Render as a Docker web service built from this repository. The Docker image copies the `app/` package and the `model/` folder and starts the server with `uvicorn app.main:app`. The container reads the `PORT` environment variable (default `8000` locally), and `/health` is used as the health check path.

## Notes

- `scikit-learn` is pinned to `1.6.1` because the model file was saved with that version; loading it with a different version can trigger warnings or errors.
- `.joblib` files are pickles. Only load model files from sources you trust.
