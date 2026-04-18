from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# تحميل الموديل
model = joblib.load("xgb_pipeline.pkl")

app = FastAPI()

# mapping للنتيجة
label_mapping = {
    0: "Slight Injury",
    1: "Serious Injury",
    2: "Fatal injury"
}

# شكل البيانات
class AccidentInput(BaseModel):
    Day_of_week: str
    hour: int
    Road_surface_type: str
    Road_surface_conditions: str
    Road_allignment: str
    Lanes_or_Medians: str
    Types_of_Junction: str
    Weather_conditions: str
    Light_conditions: str
    Area_accident_occured: str


@app.get("/")
def home():
    return {"status": "ok", "message": "API is running 🚀"}


@app.post("/predict")
def predict(data: AccidentInput):
    try:
        # تحويل البيانات لـ DataFrame
        df = pd.DataFrame([data.dict()])

        # prediction
        prediction = int(model.predict(df)[0])

        # probabilities
        proba = None
        proba_dict = None
        confidence = None

        try:
            proba = model.predict_proba(df)[0]

            # تحويلها لـ dict واضح
            proba_dict = {
                label_mapping[i]: float(p)
                for i, p in enumerate(proba)
            }

            confidence = float(max(proba))

        except Exception:
            pass

        # تحديد risk level (مفيد جدًا لـ n8n)
        if prediction == 2:
            risk_level = "high"
        elif prediction == 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "success": True,
            "prediction": prediction,
            "label": label_mapping[prediction],
            "confidence": confidence,
            "risk_level": risk_level,
            "probabilities": proba_dict
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e)
            }
        )