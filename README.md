# EduCare: Real-Time Teacher Well-Being Support System

EduCare is a machine-learning-powered teacher well-being support platform designed to assess and monitor occupational stress and potential burnout risk among teachers.

The system analyzes seven workplace and psychological dimensions and predicts a risk category: **Low, Moderate, or High**. It also provides wellness suggestions and helps teachers monitor their well-being over time.

---

## 📌 Project Overview

Teacher stress and burnout can affect teacher retention, teaching quality, and overall well-being.

EduCare aims to address this problem through:

1. **Risk Prediction:** Uses a trained Machine Learning model to estimate teacher well-being risk.
2. **Personalized Support:** Provides wellness suggestions and practical coping strategies.
3. **Progress Monitoring:** Allows teachers to review their previous check-ins.
4. **Institutional Insights:** Provides aggregated information to help administrators understand workplace stress patterns.

---

## 🔬 Psychological Framework and Core Features

The Machine Learning model uses seven input dimensions based on the Teacher Burnout Dataset.

The inputs are measured on a 1–5 Likert scale.

| Dimension | Feature Name | Description |
|---|---|---|
| Job Insecurity | `job_insecurity` | Concerns about job stability and security. |
| Work Expectations | `work_expectations` | Work pressure, deadlines, and expectations. |
| Workplace Relationships | `relationships` | Interpersonal relationships and workplace support. |
| Role Ambiguity | `role_ambiguity` | Unclear responsibilities and job expectations. |
| Role Overload | `role_overload` | Excessive workload and responsibilities. |
| Organization Culture | `organization_culture` | Workplace environment and organizational support. |
| Psychological Well-Being | `psychological_wellbeing` | Psychological well-being-related responses. |

### Prediction Categories

- Low Risk
- Moderate Risk
- High Risk

---

## 🛠️ Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Frontend | HTML5, Tailwind CSS, JavaScript |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Database | SQLite |
| Data Visualization | Chart.js |
| Model Serialization | Joblib |
| Version Control | Git and GitHub |

---

## 🤖 Machine Learning Implementation

EduCare uses **Logistic Regression** for multi-class risk classification.

The machine learning pipeline includes:

- Simple Imputer for handling missing values.
- Standard Scaler for feature scaling.
- Logistic Regression for classification.
- Joblib for saving and loading the trained model.

The trained model is stored in:

`educare_model.pkl`

---

## 🚀 Key Modules and Features

### 1. Dashboard

- Displays teacher well-being information.
- Shows the current predicted risk category.
- Provides visual summaries and progress indicators.

### 2. Daily Check-In

- Allows teachers to submit daily responses.
- Collects information across seven input dimensions.
- Sends responses to the trained Machine Learning model.
- Displays the predicted risk category.

### 3. My History

- Stores previous check-in records.
- Allows teachers to review their prediction history.
- Helps monitor changes over time.

### 4. Wellness Hub

- Provides breathing exercises.
- Includes mindfulness activities.
- Offers practical stress-management suggestions.
- Encourages teachers to seek appropriate support.

### 5. Notifications

- Provides wellness reminders.
- Displays relevant updates and notifications.

### 6. Administrator Analytics

- Displays aggregated teacher well-being information.
- Helps identify general workplace stress patterns.
- Provides visual reports and analytical insights.

---

## 📂 Project Structure

```text
EduCare/
│
├── app.py
├── database.py
├── train_model.py
├── test_model.py
├── test_educare.py
│
├── educare_model.pkl
├── educare.db
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── frontend_educare.html
│
└── Response sheet (educare) 3.xlsx
```

---

## ⚙️ Installation and Setup

Follow these steps to run EduCare on your local system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/grvx-dev/Educare-real-time-teacher-well-being-support-system.git
```

### Step 2: Open the Project Folder

```bash
cd Educare-real-time-teacher-well-being-support-system
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

### Step 5: Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🧪 Testing

To run the automated test suite:

```bash
python -m unittest test_educare.py
```

To verify the Machine Learning model:

```bash
python test_model.py
```

---

## 🔮 Future Enhancements

- Improve prediction performance through model experimentation.
- Add more personalized wellness recommendations.
- Enhance teacher progress visualization.
- Improve administrator analytics.
- Strengthen data privacy and security.
- Deploy the application on a cloud platform.

---

## 🎯 Project Objective

The main objective of EduCare is to develop a supportive digital platform that helps educational institutions understand teacher well-being and encourages a healthier working environment.

**Disclaimer:** EduCare provides supportive risk estimates and wellness information. It is not a medical diagnostic tool and does not replace professional mental health support.

---

## 👩‍💻 Project Information

| Details | Information |
|---|---|
| Project Name | EduCare |
| Project Domain | Machine Learning and Web Development |
| Backend | Python Flask |
| Machine Learning Algorithm | Logistic Regression |
| Database | SQLite |
| Repository | GitHub |
 #Click here for live website
 https://educare-real-time-teacher-well-being.onrender.com
---
**EduCare – Supporting Teacher Well-Being Through Machine Learning.**
