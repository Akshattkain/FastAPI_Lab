from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the likelihood of heart disease based on patient metrics"
)

class HeartData(BaseModel):
    age: float
    sex: float  # 1 = male, 0 = female
    cp: float  # chest pain type (0-3)
    trestbps: float  # resting blood pressure
    chol: float  # cholesterol
    fbs: float  # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    restecg: float  # resting ECG results (0-2)
    thalach: float  # max heart rate achieved
    exang: float  # exercise induced angina (1 = yes, 0 = no)
    oldpeak: float  # ST depression induced by exercise
    slope: float  # slope of peak exercise ST segment (0-2)
    ca: float  # number of major vessels colored by fluoroscopy (0-3)
    thal: float  # thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)

class HeartResponse(BaseModel):
    prediction: int
    diagnosis: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=HeartResponse)
async def predict_heart_disease(patient_data: HeartData):
    try:
        features = [[
            patient_data.age,
            patient_data.sex,
            patient_data.cp,
            patient_data.trestbps,
            patient_data.chol,
            patient_data.fbs,
            patient_data.restecg,
            patient_data.thalach,
            patient_data.exang,
            patient_data.oldpeak,
            patient_data.slope,
            patient_data.ca,
            patient_data.thal
        ]]
        prediction = predict_data(features)
        
        diagnosis = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        
        return HeartResponse(
            prediction=int(prediction[0]),
            diagnosis=diagnosis
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))