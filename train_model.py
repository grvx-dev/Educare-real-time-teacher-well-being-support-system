#Project:Educare Real Time Teacher Well being support system
#Dataset Information
#Dataset Name:Teacher Burnout Dataset
#Source:Mendeley Data
#Dataset URL:https://data.mendeley.com/datasets/stgdrjckyt/1/files/445990bc-c697-4f9e-9e71-efb4b716dcdb
#Dataset Date:2 May 2025
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Finding the Excel File
folder = Path(__file__).resolve().parent
file_path = folder / "Response sheet (educare) 3.xlsx"

if not file_path.exists():
    raise FileNotFoundError(
        f"Excel file not found: {file_path}\n"
        "Keep the Excel file in the same folder as train_model.py."
    )

# Load dataset
data = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Total responses:", len(data))
print("Total columns:", len(data.columns))


# Questions for the website
groups = {
    "job_insecurity": [f"JI{i}" for i in range(1, 9)],
    "work_expectations": [f"WE{i}" for i in range(1, 7)],
    "relationships": [f"IR{i}" for i in range(1, 7)],
    "role_ambiguity": [f"RA{i}" for i in range(1, 7)],
    "role_overload": [f"RO{i}" for i in range(1, 7)],
    "organization_culture": [f"OC{i}" for i in range(1, 7)],
    "psychological_wellbeing": [f"PWB{i}" for i in range(1, 9)]
}

burnout_columns = [f"JB{i}" for i in range(1, 9)]


# Checking whether all required columns exist
required_columns = burnout_columns.copy()

for columns in groups.values():
    required_columns.extend(columns)

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in Excel dataset: {missing_columns}"
    )


# Convert questionnaire responses to numeric values
for column in required_columns:
    data[column] = pd.to_numeric(
        data[column], errors="coerce"
    )


# Calculate the average score for each group
X = pd.DataFrame(index=data.index)

for name, columns in groups.items():
    X[name] = data[columns].mean(axis=1)


# Calculate burnout score
burnout_score = data[burnout_columns].mean(axis=1)


# Create Low, Moderate and High categories
y = pd.cut(
    burnout_score,
    bins=[0, 2.5, 3.5, 5.01],
    labels=["Low", "Moderate", "High"],
    include_lowest=True
)


# Remove rows without a usable burnout label
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Handle missing feature values using the ML pipeline
# The target label itself must be available for training.

print("\nUsable responses:", len(X))

print("\nRisk category distribution:")
print(y.value_counts())

if y.nunique() < 2:
    raise ValueError(
        "At least two risk categories are required for training."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "There are too few samples in one category for a "
        "stratified train/test split."
    )


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create the machine learning pipeline
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
)


# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["Low", "Moderate", "High"],
        zero_division=0
    )
)


# Save the trained model and feature names
model_data = {
    "model": model,
    "feature_names": list(X.columns),
    "risk_categories": ["Low", "Moderate", "High"]
}

model_path = folder / "educare_model.pkl"

joblib.dump(model_data, model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)


# Test the trained model with example questionnaire scores
new_teacher = pd.DataFrame([{
    "job_insecurity": 4.0,
    "work_expectations": 4.5,
    "relationships": 3.5,
    "role_ambiguity": 4.0,
    "role_overload": 4.5,
    "organization_culture": 4.0,
    "psychological_wellbeing": 4.0
}])

# Ensure features are in the correct order
new_teacher = new_teacher[X.columns]

risk = model.predict(new_teacher)[0]
probabilities = model.predict_proba(new_teacher)[0]

print("\nExample prediction:", risk)

print("\nEstimated class probabilities:")

for category, probability in zip(
    model.classes_, probabilities
):
    print(
        f"{category}: {probability * 100:.2f}%"
    )

print("\nEduCare model training completed!")


