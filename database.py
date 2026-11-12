import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
import random
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "educare.db"


def get_connection():
    """Create and return a SQLite database connection with row factory."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """
    Initialize SQLite database and ensure all required columns exist.
    Preserves all existing data by using conditional ALTER TABLE migrations.
    """
    connection = get_connection()
    cursor = connection.cursor()

    # Create base predictions table if it does not exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT,
            responses_json TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            probabilities_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create teachers table if it does not exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            department TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check for additional columns and migrate if missing
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row["name"] for row in cursor.fetchall()}

    new_columns = {
        "teacher_name": "TEXT",
        "department": "TEXT",
        "wellbeing_score": "REAL",
        "stress_score": "REAL",
        "workload_score": "REAL",
        "notes": "TEXT"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")

    connection.commit()
    connection.close()

    # Seed default teachers and sample baseline data if newly initialized
    seed_default_teachers_if_empty()
    seed_sample_data_if_empty()


def insert_prediction(
    responses_dict,
    predicted_category,
    probabilities_dict,
    teacher_id="T-101",
    teacher_name="Gauravi Gupta",
    department="Science",
    wellbeing_score=None,
    stress_score=None,
    workload_score=None,
    notes=None,
    created_at=None
):
    """
    Insert a teacher well-being check-in into the database.
    Computes scores if not provided.
    """
    if wellbeing_score is None:
        avg_score = sum(responses_dict.values()) / max(len(responses_dict), 1)
        wellbeing_score = round(max(0.0, min(100.0, (5.0 - avg_score) / 4.0 * 100)), 1)

    if stress_score is None:
        stress_constructs = [
            responses_dict.get("role_overload", 3.0),
            responses_dict.get("work_expectations", 3.0),
            responses_dict.get("relationships", 3.0)
        ]
        avg_stress = sum(stress_constructs) / len(stress_constructs)
        stress_score = round(((avg_stress - 1.0) / 4.0 * 9.0) + 1.0, 1)

    if workload_score is None:
        workload_constructs = [
            responses_dict.get("role_overload", 3.0),
            responses_dict.get("work_expectations", 3.0)
        ]
        avg_workload = sum(workload_constructs) / len(workload_constructs)
        workload_score = round(((avg_workload - 1.0) / 4.0 * 9.0) + 1.0, 1)

    connection = get_connection()
    cursor = connection.cursor()

    if created_at is None:
        cursor.execute(
            """
            INSERT INTO predictions (
                teacher_id,
                teacher_name,
                department,
                responses_json,
                predicted_category,
                probabilities_json,
                wellbeing_score,
                stress_score,
                workload_score,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                teacher_id,
                teacher_name,
                department,
                json.dumps(responses_dict),
                predicted_category,
                json.dumps(probabilities_dict),
                wellbeing_score,
                stress_score,
                workload_score,
                notes
            )
        )
    else:
        cursor.execute(
            """
            INSERT INTO predictions (
                teacher_id,
                teacher_name,
                department,
                responses_json,
                predicted_category,
                probabilities_json,
                wellbeing_score,
                stress_score,
                workload_score,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                teacher_id,
                teacher_name,
                department,
                json.dumps(responses_dict),
                predicted_category,
                json.dumps(probabilities_dict),
                wellbeing_score,
                stress_score,
                workload_score,
                notes,
                created_at
            )
        )

    connection.commit()
    record_id = cursor.lastrowid
    connection.close()
    return record_id


def get_recent_history(teacher_id=None, limit=50):
    """Fetch history of check-ins ordered by most recent first."""
    connection = get_connection()
    cursor = connection.cursor()

    if teacher_id:
        cursor.execute(
            """
            SELECT * FROM predictions
            WHERE teacher_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (teacher_id, limit)
        )
    else:
        cursor.execute(
            """
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )

    rows = cursor.fetchall()
    connection.close()

    history = []
    for row in rows:
        item = dict(row)
        try:
            item["responses"] = json.loads(row["responses_json"])
        except Exception:
            item["responses"] = {}

        try:
            item["probabilities"] = json.loads(row["probabilities_json"]) if row["probabilities_json"] else {}
        except Exception:
            item["probabilities"] = {}

        # Default fallbacks if legacy rows lacked new columns
        if item.get("wellbeing_score") is None and item["responses"]:
            avg_score = sum(item["responses"].values()) / max(len(item["responses"]), 1)
            item["wellbeing_score"] = round((5.0 - avg_score) / 4.0 * 100, 1)

        if item.get("stress_score") is None and item["responses"]:
            item["stress_score"] = round(item["responses"].get("role_overload", 3.0) * 2, 1)

        if item.get("workload_score") is None and item["responses"]:
            item["workload_score"] = round(item["responses"].get("work_expectations", 3.0) * 2, 1)

        history.append(item)

    return history


def get_dashboard_summary(teacher_id="T-101"):
    """
    Get key metric card values and trend data for the teacher dashboard.
    """
    connection = get_connection()
    cursor = connection.cursor()

    # Get latest check-in for this teacher
    cursor.execute(
        """
        SELECT * FROM predictions
        WHERE teacher_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (teacher_id,)
    )
    latest = cursor.fetchone()

    # Get past 7 check-ins for the weekly trend chart
    cursor.execute(
        """
        SELECT wellbeing_score, stress_score, workload_score, created_at, predicted_category
        FROM predictions
        WHERE teacher_id = ?
        ORDER BY created_at ASC
        LIMIT 7
        """,
        (teacher_id,)
    )
    trend_rows = cursor.fetchall()
    connection.close()

    if latest:
        latest_dict = dict(latest)
        wb_score = latest_dict.get("wellbeing_score") or 70.0
        stress = latest_dict.get("stress_score") or 5.0
        workload = latest_dict.get("workload_score") or 6.0
        risk = latest_dict.get("predicted_category") or "Moderate"
    else:
        wb_score = 72.0
        stress = 5.0
        workload = 6.0
        risk = "Moderate"

    trend_labels = []
    trend_scores = []

    for r in trend_rows:
        created_str = r["created_at"]
        try:
            # Parse date and format as e.g. 'Oct 24' or day of week
            dt = datetime.strptime(created_str[:10], "%Y-%m-%d")
            trend_labels.append(dt.strftime("%a"))
        except Exception:
            trend_labels.append("Day")
        trend_scores.append(round(r["wellbeing_score"] or 70.0, 1))

    if not trend_scores:
        trend_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        trend_scores = [68, 70, 67, 72, 75, 71, 74]

    return {
        "wellbeing_score": round(wb_score, 1),
        "stress_level": f"{round(stress, 1)}/10",
        "workload_level": f"{round(workload, 1)}/10",
        "current_risk": f"{risk} Risk",
        "risk_raw": risk,
        "trend_labels": trend_labels,
        "trend_scores": trend_scores
    }


def get_admin_analytics():
    """
    Compute aggregated school-wide statistics for the Administrator Dashboard.
    """
    connection = get_connection()
    cursor = connection.cursor()

    # Total check-ins
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_checkins = cursor.fetchone()[0]

    # Distinct teachers
    cursor.execute("SELECT COUNT(DISTINCT teacher_id) FROM predictions")
    total_teachers = cursor.fetchone()[0]

    # Average stress and workload
    cursor.execute("SELECT AVG(stress_score), AVG(workload_score), AVG(wellbeing_score) FROM predictions")
    avg_row = cursor.fetchone()
    avg_stress = round(avg_row[0] or 5.2, 1)
    avg_workload = round(avg_row[1] or 5.8, 1)
    avg_wellbeing = round(avg_row[2] or 71.0, 1)

    # Risk distribution
    cursor.execute("SELECT predicted_category, COUNT(*) FROM predictions GROUP BY predicted_category")
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0}
    for row in cursor.fetchall():
        cat = row[0]
        if cat in risk_counts:
            risk_counts[cat] = row[1]
        elif cat.capitalize() in risk_counts:
            risk_counts[cat.capitalize()] = row[1]

    # Department breakdown
    cursor.execute("""
        SELECT department, AVG(stress_score), AVG(workload_score), COUNT(*)
        FROM predictions
        WHERE department IS NOT NULL AND department != ''
        GROUP BY department
    """)
    dept_rows = cursor.fetchall()
    department_overview = []
    for r in dept_rows:
        department_overview.append({
            "department": r[0],
            "avg_stress": round(r[1] or 5.0, 1),
            "avg_workload": round(r[2] or 5.0, 1),
            "count": r[3]
        })

    # Workload vs Stress Scatter data
    cursor.execute("""
        SELECT workload_score, stress_score, department
        FROM predictions
        WHERE workload_score IS NOT NULL AND stress_score IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 50
    """)
    scatter_data = [
        {"x": round(r["workload_score"], 1), "y": round(r["stress_score"], 1), "department": r["department"] or "General"}
        for r in cursor.fetchall()
    ]

    connection.close()

    return {
        "total_teachers": max(total_teachers, 1),
        "total_checkins": total_checkins,
        "active_rate": "92%",
        "avg_stress": avg_stress,
        "avg_workload": avg_workload,
        "avg_wellbeing": avg_wellbeing,
        "risk_counts": risk_counts,
        "department_overview": department_overview,
        "scatter_data": scatter_data
    }


def seed_sample_data_if_empty():
    """
    If the database table has fewer than 5 records, insert realistic
    historical baseline data across multiple departments and dates
    so that all charts, history tables, and admin analytics render nicely.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    connection.close()

    if count >= 5:
        return

    sample_teachers = [
        ("T-101", "Gauravi Gupta", "Science"),
        ("T-102", "Rahul Sharma", "Math"),
        ("T-103", "Ananya Verma", "Humanities"),
        ("T-104", "Vikram Patel", "Arts"),
        ("T-105", "Priya Nair", "Languages"),
        ("T-106", "Rohan Mehta", "Physical Ed"),
        ("T-107", "Sneha Rao", "Science"),
        ("T-108", "Amitabh Sen", "Math")
    ]

    base_date = datetime.now()

    # Pre-calculated realistic responses covering Low, Moderate, and High risks
    sample_records = [
        # (teacher, days_ago, responses, predicted, probs, notes)
        (
            sample_teachers[0], 7,
            {"job_insecurity": 2.0, "work_expectations": 3.0, "relationships": 2.0, "role_ambiguity": 2.0, "role_overload": 3.0, "organization_culture": 2.0, "psychological_wellbeing": 2.5},
            "Low", {"Low": 88.5, "Moderate": 10.2, "High": 1.3},
            "Smooth start to the teaching unit."
        ),
        (
            sample_teachers[0], 5,
            {"job_insecurity": 2.5, "work_expectations": 3.5, "relationships": 2.5, "role_ambiguity": 2.5, "role_overload": 3.2, "organization_culture": 2.2, "psychological_wellbeing": 2.8},
            "Low", {"Low": 78.4, "Moderate": 19.1, "High": 2.5},
            "Grading quizzes, manageable load."
        ),
        (
            sample_teachers[0], 3,
            {"job_insecurity": 3.0, "work_expectations": 4.0, "relationships": 3.0, "role_ambiguity": 3.0, "role_overload": 4.0, "organization_culture": 3.2, "psychological_wellbeing": 3.4},
            "Moderate", {"Low": 22.4, "Moderate": 65.1, "High": 12.5},
            "Parent-teacher meeting preparations."
        ),
        (
            sample_teachers[0], 1,
            {"job_insecurity": 2.8, "work_expectations": 3.5, "relationships": 2.5, "role_ambiguity": 2.8, "role_overload": 3.5, "organization_culture": 2.8, "psychological_wellbeing": 3.0},
            "Moderate", {"Low": 35.8, "Moderate": 58.2, "High": 6.0},
            "Midterm assessments completed."
        ),
        (
            sample_teachers[1], 4,
            {"job_insecurity": 3.5, "work_expectations": 4.2, "relationships": 3.8, "role_ambiguity": 3.5, "role_overload": 4.5, "organization_culture": 3.8, "psychological_wellbeing": 4.0},
            "High", {"Low": 5.2, "Moderate": 31.4, "High": 63.4},
            "Heavy calculus coursework and committee duty."
        ),
        (
            sample_teachers[2], 3,
            {"job_insecurity": 2.0, "work_expectations": 2.8, "relationships": 2.2, "role_ambiguity": 2.0, "role_overload": 3.0, "organization_culture": 2.0, "psychological_wellbeing": 2.2},
            "Low", {"Low": 91.2, "Moderate": 8.0, "High": 0.8},
            "Great class engagement on history essay topics."
        ),
        (
            sample_teachers[3], 2,
            {"job_insecurity": 2.2, "work_expectations": 3.2, "relationships": 2.0, "role_ambiguity": 2.5, "role_overload": 3.0, "organization_culture": 2.2, "psychological_wellbeing": 2.4},
            "Low", {"Low": 86.0, "Moderate": 12.5, "High": 1.5},
            "Art exhibition rehearsals underway."
        ),
        (
            sample_teachers[4], 2,
            {"job_insecurity": 3.0, "work_expectations": 3.8, "relationships": 3.2, "role_ambiguity": 3.0, "role_overload": 3.8, "organization_culture": 3.0, "psychological_wellbeing": 3.2},
            "Moderate", {"Low": 28.5, "Moderate": 61.2, "High": 10.3},
            "Speech competition coordination."
        ),
        (
            sample_teachers[5], 1,
            {"job_insecurity": 2.2, "work_expectations": 3.5, "relationships": 2.0, "role_ambiguity": 2.0, "role_overload": 3.2, "organization_culture": 2.0, "psychological_wellbeing": 2.5},
            "Low", {"Low": 82.5, "Moderate": 15.5, "High": 2.0},
            "Sports day drills running smoothly."
        ),
        (
            sample_teachers[6], 2,
            {"job_insecurity": 4.0, "work_expectations": 4.5, "relationships": 3.8, "role_ambiguity": 4.0, "role_overload": 4.8, "organization_culture": 4.2, "psychological_wellbeing": 4.2},
            "High", {"Low": 2.1, "Moderate": 18.5, "High": 79.4},
            "Laboratory inspections and double batches."
        ),
        (
            sample_teachers[7], 1,
            {"job_insecurity": 3.2, "work_expectations": 4.0, "relationships": 3.0, "role_ambiguity": 3.2, "role_overload": 4.2, "organization_culture": 3.5, "psychological_wellbeing": 3.6},
            "Moderate", {"Low": 18.0, "Moderate": 64.0, "High": 18.0},
            "Math Olympiad tutoring."
        )
    ]

    for teacher, days_ago, responses, predicted, probs, notes in sample_records:
        created_time = (base_date - timedelta(days=days_ago, hours=random.randint(1, 6))).strftime("%Y-%m-%d %H:%M:%S")
        insert_prediction(
            responses_dict=responses,
            predicted_category=predicted,
            probabilities_dict=probs,
            teacher_id=teacher[0],
            teacher_name=teacher[1],
            department=teacher[2],
            notes=notes,
            created_at=created_time
        )


def seed_default_teachers_if_empty():
    """Seed predefined active faculty accounts if teachers table is empty."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM teachers")
    count = cursor.fetchone()[0]

    if count == 0:
        default_teachers = [
            ("T-101", "Gauravi Gupta", "gauravi@educare.edu", "password123", "Science"),
            ("T-102", "Rahul Sharma", "rahul@educare.edu", "password123", "Math"),
            ("T-103", "Ananya Verma", "ananya@educare.edu", "password123", "Humanities"),
            ("T-104", "Vikram Patel", "vikram@educare.edu", "password123", "Arts"),
            ("T-105", "Priya Nair", "priya@educare.edu", "password123", "Languages"),
            ("T-106", "Rohan Mehta", "rohan@educare.edu", "password123", "Physical Ed"),
            ("T-107", "Sneha Rao", "sneha@educare.edu", "password123", "Science"),
            ("T-108", "Amitabh Sen", "amitabh@educare.edu", "password123", "Math"),
        ]
        for tid, name, email, pwd, dept in default_teachers:
            cursor.execute(
                """
                INSERT INTO teachers (teacher_id, name, email, password_hash, department)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tid, name, email, generate_password_hash(pwd), dept)
            )
        connection.commit()

    connection.close()


def register_teacher(teacher_id, name, email, password, department):
    """
    Register a new teacher with securely hashed password.
    Returns teacher dict on success; raises ValueError on duplicate or invalid data.
    """
    teacher_id = (teacher_id or "").strip()
    name = (name or "").strip()
    email = (email or "").strip().lower()
    department = (department or "General").strip()

    if not teacher_id:
        raise ValueError("Teacher ID is required.")
    if not name:
        raise ValueError("Teacher full name is required.")
    if not email or "@" not in email:
        raise ValueError("A valid email address is required.")
    if not password or len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    connection = get_connection()
    cursor = connection.cursor()

    # Check for existing teacher_id or email
    cursor.execute(
        "SELECT teacher_id, email FROM teachers WHERE UPPER(teacher_id) = UPPER(?) OR LOWER(email) = LOWER(?)",
        (teacher_id, email)
    )
    existing = cursor.fetchone()
    if existing:
        connection.close()
        if existing["teacher_id"].upper() == teacher_id.upper():
            raise ValueError(f"Teacher ID '{teacher_id}' is already registered.")
        else:
            raise ValueError(f"Email '{email}' is already registered.")

    hashed_pw = generate_password_hash(password)
    cursor.execute(
        """
        INSERT INTO teachers (teacher_id, name, email, password_hash, department)
        VALUES (?, ?, ?, ?, ?)
        """,
        (teacher_id, name, email, hashed_pw, department)
    )
    connection.commit()
    new_id = cursor.lastrowid
    connection.close()

    return {
        "id": new_id,
        "teacher_id": teacher_id,
        "name": name,
        "email": email,
        "department": department
    }


def authenticate_teacher(identifier, password):
    """
    Authenticate teacher by Email or Teacher ID and password.
    Returns teacher dict if valid, None otherwise.
    """
    if not identifier or not password:
        return None

    ident = identifier.strip()
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, teacher_id, name, email, password_hash, department, created_at
        FROM teachers
        WHERE LOWER(email) = LOWER(?) OR UPPER(teacher_id) = UPPER(?)
        """,
        (ident, ident)
    )
    row = cursor.fetchone()
    connection.close()

    if row and check_password_hash(row["password_hash"], password):
        return {
            "id": row["id"],
            "teacher_id": row["teacher_id"],
            "name": row["name"],
            "email": row["email"],
            "department": row["department"],
            "created_at": row["created_at"]
        }
    return None


def get_teacher_by_id(teacher_id):
    """Fetch teacher profile by teacher_id."""
    if not teacher_id:
        return None
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT id, teacher_id, name, email, department, created_at
        FROM teachers
        WHERE UPPER(teacher_id) = UPPER(?)
        """,
        (teacher_id.strip(),)
    )
    row = cursor.fetchone()
    connection.close()

    if row:
        return {
            "id": row["id"],
            "teacher_id": row["teacher_id"],
            "name": row["name"],
            "email": row["email"],
            "department": row["department"],
            "created_at": row["created_at"]
        }
    return None


def get_all_teachers():
    """Retrieve all registered teachers."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, teacher_id, name, email, department, created_at FROM teachers ORDER BY teacher_id ASC")
    rows = cursor.fetchall()
    connection.close()
    return [
        {
            "id": r["id"],
            "teacher_id": r["teacher_id"],
            "name": r["name"],
            "email": r["email"],
            "department": r["department"],
            "created_at": r["created_at"]
        }
        for r in rows
    ]


if __name__ == "__main__":
    initialize_database()
    print("EduCare database initialized and verified.")
    print("Dashboard summary:", get_dashboard_summary())
    print("Admin analytics:", get_admin_analytics())