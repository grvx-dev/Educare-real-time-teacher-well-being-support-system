
from flask import Flask, render_template, request, jsonify, Response, session
import os
import joblib
from pathlib import Path
import pandas as pd
import json
import io
import csv

from database import (
    initialize_database,
    get_connection,
    insert_prediction,
    get_recent_history,
    get_dashboard_summary,
    get_admin_analytics,
    register_teacher,
    authenticate_teacher,
    get_teacher_by_id,
    get_all_teachers
)


# Initialize Flask application
app = Flask(__name__)

# Secure secret key using environment variables
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "local-development-key-change-before-deployment"
)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# ML model path
MODEL_PATH = BASE_DIR / "educare_model.pkl"

# Initialize database
initialize_database()


# Load trained ML model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Run train_model.py first."
    )

model_data = joblib.load(MODEL_PATH)


# Check model format
if not isinstance(model_data, dict):
    raise ValueError(
        "The saved model must contain a dictionary "
        "with model metadata."
    )

required_keys = {"model", "feature_names", "risk_categories"}
missing_keys = required_keys - set(model_data.keys())

if missing_keys:
    raise ValueError(
        f"Missing information in educare_model.pkl: {missing_keys}. "
        "Check the model-saving code in train_model.py."
    )

model = model_data["model"]
feature_names = list(model_data["feature_names"])
risk_categories = list(model_data["risk_categories"])


# Questionnaire descriptions
FEATURE_LABELS = {
    "job_insecurity": "Job Insecurity",
    "work_expectations": "Work Expectations",
    "relationships": "Workplace Relationships",
    "role_ambiguity": "Role Ambiguity",
    "role_overload": "Role Overload",
    "organization_culture": "Organization Culture",
    "psychological_wellbeing": "Psychological Well-Being"
}


FEATURE_DESCRIPTIONS = {
    "job_insecurity": "Concerns regarding position stability, contract renewal, or organizational restructuring.",

    "work_expectations": "Unrealistic workload pace, administrative pressure, and tight deadlines.",

    "relationships": "Interpersonal tension, lack of collegial support, or isolation at school.",

    "role_ambiguity": "Unclear objectives, conflicting directives, and uncertainty about responsibilities.",

    "role_overload": "Excessive grading, teaching volume, lesson planning, and extracurricular tasks.",

    "organization_culture": "Institutional climate, top-down friction, and lack of teacher voice/recognition.",

    "psychological_wellbeing": "Feelings of mental exhaustion, chronic fatigue, anxiety, and low motivation."
}


def get_feature_label(feature):
    """Convert a feature name into a readable label."""

    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


def generate_recommendations(prediction, values):
    """Generate personalized recommendations for the teacher."""

    recommendations = []

    # Overall recommendation based on predicted risk category
    if prediction == "Low":

        recommendations.append(
            "Your well-being indicators are currently balanced. Keep maintaining healthy boundaries between school prep and evening personal time."
        )

        recommendations.append(
            "Continue taking brief 2-minute restorative pauses between instructional periods."
        )

    elif prediction == "Moderate":

        recommendations.append(
            "Early stress indicators detected. Review upcoming deadlines and triage non-urgent administrative work."
        )

        recommendations.append(
            "Take 5 minutes today in the Wellness Hub for guided Box Breathing to reset your nervous system."
        )

    else:

        recommendations.append(
            "High burnout risk detected. Prioritize immediate rest and consider discussing current demands with your department head or a trusted peer."
        )

        recommendations.append(
            "Reach out to your school's employee assistance program or the confidential wellness resources available in the Wellness Hub."
        )

    # Factor-specific recommendations
    highest_factor = max(
        values.items(),
        key=lambda item: item[1]
    ) if values else None

    if highest_factor:

        factor_name, factor_val = highest_factor

        if factor_val >= 3.5:

            if factor_name == "role_overload":

                recommendations.append(
                    f"Noticeable burden in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Consider delegating tasks, utilizing peer reviews for assignments, and declining optional committee requests this week."
                )

            elif factor_name == "work_expectations":

                recommendations.append(
                    f"Noticeable pressure in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Set realistic pacing targets with colleagues and clarify priority deliverables with administration."
                )

            elif factor_name == "relationships":

                recommendations.append(
                    f"Noticeable strain in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Schedule a casual coffee or debrief with an empathetic colleague to share classroom experiences."
                )

            elif factor_name == "role_ambiguity":

                recommendations.append(
                    f"Noticeable uncertainty in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Request a brief 10-minute alignment check-in with your supervisor to clarify core expectations."
                )

            elif factor_name == "job_insecurity":

                recommendations.append(
                    f"Heightened concern in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Focus on your daily classroom accomplishments and consult official institutional career progression guidelines."
                )

            elif factor_name == "organization_culture":

                recommendations.append(
                    f"Institutional climate concern in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Connect with your faculty association or teacher forum for collective support."
                )

            elif factor_name == "psychological_wellbeing":

                recommendations.append(
                    f"Elevated mental fatigue in {FEATURE_LABELS.get(factor_name)} ({factor_val}/5): Prioritize restorative sleep (7-8 hours) and unplug completely from school communications after hours."
                )

    return recommendations, highest_factor


def get_current_user():
    """Retrieve currently authenticated teacher."""

    tid = session.get("teacher_id")

    if tid:

        teacher = get_teacher_by_id(tid)

        if teacher:
            return teacher

    # Fallback demo teacher
    fallback = get_teacher_by_id("T-101")

    if fallback:
        return fallback

    return {
        "teacher_id": "T-101",
        "name": "Gauravi Gupta",
        "email": "gauravi@educare.edu",
        "department": "Science"
    }


def render_home(**context):
    """Render EduCare frontend with dynamic dashboard data."""

    current_user = get_current_user()

    teacher_id = current_user.get(
        "teacher_id",
        "T-101"
    )

    dashboard_summary = get_dashboard_summary(teacher_id)

    recent_history = get_recent_history(
        teacher_id=teacher_id,
        limit=10
    )

    admin_analytics = get_admin_analytics()

    all_teachers = get_all_teachers()

    return render_template(
        "frontend_educare.html",

        feature_names=feature_names,
        feature_labels=FEATURE_LABELS,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        get_feature_label=get_feature_label,
        risk_categories=risk_categories,

        dashboard_summary=dashboard_summary,
        recent_history=recent_history,
        admin_analytics=admin_analytics,

        current_user=current_user,
        all_teachers=all_teachers,

        **context
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_home()


# --------------------------------------------------
# TEACHER REGISTRATION API
# --------------------------------------------------

@app.route("/api/register", methods=["POST"])
def api_register():

    try:

        data = request.get_json(silent=True) or request.form.to_dict()

        teacher_id = data.get("teacher_id", "").strip()
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "")
        department = data.get("department", "General").strip()

        teacher = register_teacher(
            teacher_id=teacher_id,
            name=name,
            email=email,
            password=password,
            department=department
        )

        session["teacher_id"] = teacher["teacher_id"]

        return jsonify({
            "success": True,
            "teacher": teacher,
            "message": f"Welcome to EduCare, {teacher['name']}!"
        })

    except ValueError as val_err:

        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 400

    except Exception:

        app.logger.exception(
            "Error during teacher registration"
        )

        return jsonify({
            "success": False,
            "error": "Registration failed. Please check your details."
        }), 500


# --------------------------------------------------
# TEACHER LOGIN API
# --------------------------------------------------

@app.route("/api/login", methods=["POST"])
def api_login():

    try:

        data = request.get_json(silent=True) or request.form.to_dict()

        identifier = (
            data.get("identifier")
            or data.get("email")
            or data.get("teacher_id")
            or ""
        )

        password = data.get("password", "")

        teacher = authenticate_teacher(
            identifier=identifier,
            password=password
        )

        if not teacher:

            return jsonify({
                "success": False,
                "error": "Invalid email/teacher ID or password."
            }), 401

        session["teacher_id"] = teacher["teacher_id"]

        return jsonify({
            "success": True,
            "teacher": teacher,
            "message": f"Welcome back, {teacher['name']}!"
        })

    except Exception:

        app.logger.exception(
            "Error during teacher login"
        )

        return jsonify({
            "success": False,
            "error": "Login request failed. Please try again."
        }), 500


# --------------------------------------------------
# TEACHER LOGOUT API
# --------------------------------------------------

@app.route("/api/logout", methods=["POST", "GET"])
def api_logout():

    session.pop("teacher_id", None)

    return jsonify({
        "success": True,
        "message": "Signed out successfully."
    })


# --------------------------------------------------
# CURRENT USER API
# --------------------------------------------------

@app.route("/api/current-user", methods=["GET"])
def api_current_user():

    return jsonify({
        "success": True,
        "teacher": get_current_user()
    })


# --------------------------------------------------
# GET ALL TEACHERS API
# --------------------------------------------------

@app.route("/api/teachers", methods=["GET"])
def api_teachers():

    return jsonify({
        "success": True,
        "teachers": get_all_teachers()
    })


# --------------------------------------------------
# ML PREDICTION API
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json(silent=True) or request.form.to_dict()

        values = {}

        # Validate all ML input features
        for feature in feature_names:

            raw_value = data.get(feature)

            if raw_value is None or str(raw_value).strip() == "":

                return jsonify({
                    "success": False,
                    "error": f"Please provide a rating for {get_feature_label(feature)}."
                }), 400

            try:

                val = float(raw_value)

            except (ValueError, TypeError):

                return jsonify({
                    "success": False,
                    "error": f"{get_feature_label(feature)} must be a numeric rating."
                }), 400

            if not 1.0 <= val <= 5.0:

                return jsonify({
                    "success": False,
                    "error": f"{get_feature_label(feature)} must be between 1 and 5."
                }), 400

            values[feature] = val

        curr = get_current_user()

        teacher_id = data.get(
            "teacher_id"
        ) or curr.get("teacher_id", "T-101")

        teacher_name = data.get(
            "teacher_name"
        ) or curr.get("name", "Gauravi Gupta")

        department = data.get(
            "department"
        ) or curr.get("department", "Science")

        notes = data.get("notes", "").strip()

        # Prepare ML input in exact feature order
        input_data = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names
        )

        # Predict risk category
        prediction = model.predict(input_data)[0]

        # Calculate prediction probabilities
        probability_results = {}

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability_results = {
                str(category): round(float(prob) * 100, 1)

                for category, prob in zip(
                    model.classes_,
                    probabilities
                )
            }

        # Save prediction in database
        record_id = insert_prediction(
            responses_dict=values,
            predicted_category=str(prediction),
            probabilities_dict=probability_results,
            teacher_id=teacher_id,
            teacher_name=teacher_name,
            department=department,
            notes=notes
        )

        recommendations, highest_factor = generate_recommendations(
            str(prediction),
            values
        )

        dashboard_summary = get_dashboard_summary(
            teacher_id
        )

        return jsonify({

            "success": True,

            "prediction_id": record_id,

            "prediction": str(prediction),

            "risk_label": f"{prediction} Risk",

            "probabilities": probability_results,

            "highest_factor": {

                "name": highest_factor[0],

                "label": get_feature_label(
                    highest_factor[0]
                ),

                "score": highest_factor[1]

            } if highest_factor else None,

            "recommendations": recommendations,

            "dashboard_summary": dashboard_summary

        })

    except Exception:

        app.logger.exception(
            "Error in /api/predict"
        )

        return jsonify({
            "success": False,
            "error": "Failed to process check-in. Please verify all inputs."
        }), 500


# --------------------------------------------------
# STANDARD FORM PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        values = {}

        for feature in feature_names:

            raw_value = request.form.get(
                feature,
                ""
            ).strip()

            if not raw_value:

                raise ValueError(
                    f"Please enter a value for {get_feature_label(feature)}."
                )

            val = float(raw_value)

            if not 1.0 <= val <= 5.0:

                raise ValueError(
                    f"{get_feature_label(feature)} must be between 1 and 5."
                )

            values[feature] = val

        input_data = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        probability_results = {}

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability_results = {
                str(category): round(float(prob) * 100, 1)

                for category, prob in zip(
                    model.classes_,
                    probabilities
                )
            }

        curr = get_current_user()

        teacher_id = request.form.get(
            "teacher_id"
        ) or curr.get("teacher_id", "T-101")

        teacher_name = request.form.get(
            "teacher_name"
        ) or curr.get("name", "Gauravi Gupta")

        department = request.form.get(
            "department"
        ) or curr.get("department", "Science")

        notes = request.form.get(
            "notes",
            ""
        ).strip()

        prediction_id = insert_prediction(
            responses_dict=values,
            predicted_category=str(prediction),
            probabilities_dict=probability_results,
            teacher_id=teacher_id,
            teacher_name=teacher_name,
            department=department,
            notes=notes
        )

        recommendations, highest_factor = generate_recommendations(
            str(prediction),
            values
        )

        return render_home(

            prediction=str(prediction),

            probabilities=probability_results,

            submitted_values=values,

            prediction_id=prediction_id,

            recommendations=recommendations,

            highest_factor=highest_factor

        )

    except ValueError as error:

        return render_home(
            error=str(error),
            submitted_values=request.form
        ), 400

    except Exception:

        app.logger.exception(
            "EduCare prediction failed"
        )

        return render_home(
            error="Unable to calculate or save the prediction. Please try again.",
            submitted_values=request.form
        ), 500


# --------------------------------------------------
# TEACHER HISTORY API
# --------------------------------------------------

@app.route("/api/history", methods=["GET"])
def api_history():

    curr = get_current_user()

    teacher_id = request.args.get(
        "teacher_id"
    ) or curr.get("teacher_id")

    history = get_recent_history(
        teacher_id=teacher_id,
        limit=30
    )

    return jsonify({
        "success": True,
        "history": history
    })


# --------------------------------------------------
# DASHBOARD STATISTICS API
# --------------------------------------------------

@app.route("/api/dashboard-stats", methods=["GET"])
def api_dashboard_stats():

    curr = get_current_user()

    teacher_id = request.args.get(
        "teacher_id"
    ) or curr.get("teacher_id")

    summary = get_dashboard_summary(
        teacher_id=teacher_id
    )

    return jsonify({
        "success": True,
        "summary": summary
    })


# --------------------------------------------------
# ADMIN ANALYTICS API
# --------------------------------------------------

@app.route("/api/admin-stats", methods=["GET"])
def api_admin_stats():

    analytics = get_admin_analytics()

    return jsonify({
        "success": True,
        "analytics": analytics
    })


# --------------------------------------------------
# EXPORT ANALYTICS REPORT
# --------------------------------------------------

@app.route("/api/export-report", methods=["GET"])
def api_export_report():

    history = get_recent_history(
        limit=500
    )

    analytics = get_admin_analytics()

    output = io.StringIO()

    writer = csv.writer(output)

    # Executive Summary
    writer.writerow([
        "EDUCARE - TEACHER WELL-BEING & BURNOUT ANALYTICS REPORT"
    ])

    writer.writerow([
        "Generated At",
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    ])

    writer.writerow([])

    writer.writerow(["EXECUTIVE SUMMARY"])

    writer.writerow([
        "Total Teachers",
        analytics["total_teachers"]
    ])

    writer.writerow([
        "Total Check-Ins",
        analytics["total_checkins"]
    ])

    writer.writerow([
        "Active Participation Rate",
        analytics["active_rate"]
    ])

    writer.writerow([
        "School-wide Average Stress (1-10)",
        analytics["avg_stress"]
    ])

    writer.writerow([
        "School-wide Average Workload (1-10)",
        analytics["avg_workload"]
    ])

    writer.writerow([
        "School-wide Average Well-Being Score (0-100)",
        analytics["avg_wellbeing"]
    ])

    writer.writerow([])

    # Risk Distribution
    writer.writerow(["BURNOUT RISK DISTRIBUTION"])

    for cat, count in analytics["risk_counts"].items():

        pct = round(
            count / max(analytics["total_checkins"], 1) * 100,
            1
        )

        writer.writerow([
            f"{cat} Risk",
            count,
            f"{pct}%"
        ])

    writer.writerow([])

    # Department Breakdown
    writer.writerow(["DEPARTMENT BREAKDOWN"])

    writer.writerow([
        "Department",
        "Average Stress (1-10)",
        "Average Workload (1-10)",
        "Total Check-Ins"
    ])

    for dept in analytics["department_overview"]:

        writer.writerow([
            dept["department"],
            dept["avg_stress"],
            dept["avg_workload"],
            dept["count"]
        ])

    writer.writerow([])

    # Recent Check-In Logs
    writer.writerow(["RECENT CHECK-IN LOGS"])

    writer.writerow([
        "Record ID",
        "Timestamp",
        "Teacher ID",
        "Teacher Name",
        "Department",
        "Predicted Risk",
        "Well-Being Score",
        "Stress Score",
        "Workload Score",
        "Job Insecurity",
        "Work Expectations",
        "Relationships",
        "Role Ambiguity",
        "Role Overload",
        "Organization Culture",
        "Psychological Well-Being",
        "Notes"
    ])

    for item in history:

        resp = item.get(
            "responses",
            {}
        )

        writer.writerow([

            item.get("id"),

            item.get("created_at"),

            item.get("teacher_id"),

            item.get("teacher_name"),

            item.get("department"),

            item.get("predicted_category"),

            item.get("wellbeing_score"),

            item.get("stress_score"),

            item.get("workload_score"),

            resp.get("job_insecurity", ""),

            resp.get("work_expectations", ""),

            resp.get("relationships", ""),

            resp.get("role_ambiguity", ""),

            resp.get("role_overload", ""),

            resp.get("organization_culture", ""),

            resp.get("psychological_wellbeing", ""),

            item.get("notes", "")

        ])

    csv_data = output.getvalue()

    return Response(

        csv_data,

        mimetype="text/csv",

        headers={
            "Content-Disposition":
            "attachment; filename=educare_wellbeing_report.csv"
        }

    )


# --------------------------------------------------
# RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=os.environ.get("FLASK_DEBUG") == "1"

    )
