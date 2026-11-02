from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    """Input features for the heart disease classifier (UCI/Kaggle heart.csv)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
                "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
            }
        }
    )

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")


class PredictionResponse(BaseModel):
    heart_disease: bool
    probability: float = Field(..., ge=0, le=1, description="Estimated probability of heart disease")


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    model_type: str
    model_name: str
    n_features: int
    features: list[str]
    target: str