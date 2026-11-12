
import pandas as pd
import joblib
from pathlib import Path

folder = Path(__file__).resolve().parent
model_path = folder / "educare_model.pkl"

if not model_path.exists():
    raise FileNotFoundError(
        "educare_model.pkl not found. "
        "Run train_model.py first."
    )

model_data = joblib.load(model_path)

model = model_data["model"]
feature_names = model_data["feature_names"]
risk_categories = model_data["risk_categories"]

print("EduCare model loaded successfully!")

print("\nFeatures:", feature_names)
print("Risk categories:", risk_categories)

new_teacher = pd.DataFrame([{
    "job_insecurity": 4.0,
    "work_expectations": 4.5,
    "relationships": 3.5,
    "role_ambiguity": 4.0,
    "role_overload": 4.5,
    "organization_culture": 4.0,
    "psychological_wellbeing": 4.0
}])

new_teacher = new_teacher[feature_names]

prediction = model.predict(new_teacher)[0]
probabilities = model.predict_proba(new_teacher)[0]

print("\nPrediction successful!")
print("Teacher risk category:", prediction)

print("\nEstimated class probabilities:")

for category, probability in zip(
    model.classes_, probabilities
):
    print(f"{category}: {probability * 100:.2f}%")

print("\nEduCare model test completed!")
