# EduCare: Real-Time Teacher Well-Being Support System

EduCare is an intelligent, machine-learning-powered educator well-being support platform designed to assess, monitor, and alleviate teacher burnout and occupational stress in educational institutions.

---

## 📌 Project Overview

Teacher burnout is a major global concern impacting teacher retention, pedagogical quality, and educator mental health. EduCare leverages empirical data from the **Teacher Burnout Dataset (Mendeley Data)** to provide:
1. **Scientifically Grounded Risk Prediction**: Evaluates 7 core psychological constructs to classify burnout risk as **Low**, **Moderate**, or **High**.
2. **Real-Time Intervention**: Recommends personalized, evidence-based coping mechanisms and micro-interventions.
3. **Institutional Insight**: Aggregates school-wide metrics for academic administrators to identify high-stress departments and enact supportive workload policies.

---

## 🔬 Psychological Framework & Core Constructs

The Machine Learning model is trained on 7 validated psychological dimensions (scored on a 1.0 to 5.0 Likert scale from *Strongly Disagree* to *Strongly Agree*):

| Dimension | Feature Name | Description |
| :--- | :--- | :--- |
| **Job Insecurity** | `job_insecurity` | Concerns regarding contract renewal, position stability, or institutional restructuring. |
| **Work Expectations** | `work_expectations` | Unrealistic workload pace, administrative pressure, and tight deadlines. |
| **Workplace Relationships** | `relationships` | Interpersonal tension, collegial friction, or feeling isolated in the school role. |
| **Role Ambiguity** | `role_ambiguity` | Unclear duties, conflicting directives, and uncertainty about responsibilities. |
| **Role Overload** | `role_overload` | Excessive grading, lesson planning, extracurricular supervision, and meeting volume. |
| **Organization Culture** | `organization_culture` | Institutional climate, top-down friction, and lack of teacher autonomy/voice. |
| **Psychological Well-Being** | `psychological_wellbeing` | Feelings of mental exhaustion, chronic fatigue, stress headaches, and anxiety. |

---

## 🛠️ System Architecture & Tech Stack

```text
EduCare/
├── app.py                      # Flask backend & REST APIs (/api/predict, /api/history, /api/login, /api/register)
├── database.py                 # SQLite database schema, CRUD queries, migrations, and analytics aggregations
├── educare.db                  # SQLite database storing predictions, teachers, and check-in logs
├── educare_model.pkl           # Trained Scikit-Learn machine learning pipeline (StandardScaler + Balanced LogisticRegression)
├── train_model.py              # ML model training script using Mendeley Teacher Burnout Dataset
├── test_model.py               # ML model inference verification script
├── test_educare.py             # Comprehensive automated test suite (Unit & Integration tests)
├── requirements.txt            # Python dependencies
├── Response sheet (educare) 3.xlsx # Mendeley Data Teacher Burnout dataset
└── templates/
    └── frontend_educare.html   # Single Page Application (Dashboard, Check-In, History, Wellness Hub, Notifications, Admin Analytics)
```

- **Backend**: Python 3, Flask, SQLite3, Pandas
- **Machine Learning**: Scikit-Learn (SimpleImputer, StandardScaler, LogisticRegression with balanced class weights), Joblib
- **Frontend**: HTML5, Tailwind CSS, Chart.js

---

## 🚀 Key Modules & Features

### 1. 🏠 Dashboard
- Real-time metric cards: Overall Well-Being Score (0-100), Scaled Stress Level (1-10), Workload Index (1-10), and Current Risk Category badge.
- Weekly Well-Being Trend chart powered by actual historical records.
- Adaptive Status card providing instant encouraging feedback based on current risk.

### 2. 📝 Daily Check-In
- Interactive Likert sliders for the 7 research constructs (1.0 = Strongly Disagree to 5.0 = Strongly Agree) with live verbal indicators.
- Asynchronous submission (POST /api/predict) directly to the trained ML model.
- **Results Modal**: Displays color-coded risk category, exact class probability breakdown (Low %, Moderate %, High %), primary stress contributor, and personalized action steps.
- Instantly updates the Dashboard and History table without a full browser reload.

### 3. 📊 My History
- Trajectory line chart tracking stress and workload across check-ins.
- Historical check-in table with timestamp, department, scores, and risk badges.
- **Record Details Modal**: Allows inspecting individual 1-5 construct scores and notes for any past check-in.

### 4. 🌿 Wellness Hub
- **🌬️ Box Breathing Coach**: Interactive 4-4-4-4 cycle (Inhale 4s, Hold 4s, Exhale 4s, Hold 4s) with animated pulse and cycle counter.
- **🧘 Mindful Reset Timer**: 2 min, 5 min, and 10 min focus presets with start, pause, and reset controls.
- **📚 Practical Coping Strategies**: Categorized educator triage guides for grading batching, boundary setting, and collegial debriefing.
- **🆘 Institutional Support**: Emergency helplines (Dial 988, Crisis Text Line 741741) and an interactive confidential counselor callback request form.

### 5. 🔔 Notifications
- Context-sensitive risk alerts (e.g. prompt to take a restorative break when stress is elevated).
- Daily check-in reminders and institutional notices.
- Mark all as read functionality.

### 6. 👨‍💼 Administrator Analytics
- Aggregated school-wide indicators: Total teachers, active check-in rate, average stress, and average workload.
- **Risk Distribution Chart**: Doughnut chart showing proportion of faculty in Low, Moderate, and High burnout risk.
- **Department Overview**: Comparative bar chart evaluating stress and workload across Science, Math, Humanities, Arts, Languages, and Physical Ed.
- **Workload vs. Stress Correlation Matrix**: Scatter plot visualizing check-in clusters.
- **📥 One-Click CSV Export**: Downloads a comprehensive educare_wellbeing_report.csv report.

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
`ash
pip install -r requirements.txt
`

### 2. Run the Application
`ash
python app.py
`
Open your web browser and navigate to:
`
http://127.0.0.1:5000
`

### 3. Retrain the Machine Learning Model (Optional)
`ash
python train_model.py
`

### 4. Run Automated Tests
`ash
python -m unittest test_educare.py
`
