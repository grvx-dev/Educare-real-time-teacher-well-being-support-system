import unittest
import json
import sqlite3
import pandas as pd
from app import app, model, feature_names
from database import (
    initialize_database,
    get_connection,
    insert_prediction,
    get_recent_history,
    get_dashboard_summary,
    get_admin_analytics
)


class EduCareTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

    def test_database_initialization(self):
        """Test database connection and table existence."""
        initialize_database()
        conn = get_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        table = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(table)

    def test_insert_and_get_history(self):
        """Test inserting a prediction and retrieving it from history."""
        sample_responses = {feat: 3.0 for feat in feature_names}
        sample_probs = {"Low": 20.0, "Moderate": 70.0, "High": 10.0}

        record_id = insert_prediction(
            responses_dict=sample_responses,
            predicted_category="Moderate",
            probabilities_dict=sample_probs,
            teacher_id="TEST-999",
            teacher_name="Test Teacher",
            department="Science",
            notes="Automated test note"
        )
        self.assertIsInstance(record_id, int)

        history = get_recent_history(teacher_id="TEST-999", limit=5)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["teacher_name"], "Test Teacher")
        self.assertEqual(history[0]["predicted_category"], "Moderate")
        self.assertEqual(history[0]["notes"], "Automated test note")

    def test_dashboard_and_admin_stats(self):
        """Test summary and aggregated analytics methods."""
        summary = get_dashboard_summary()
        self.assertIn("wellbeing_score", summary)
        self.assertIn("stress_level", summary)
        self.assertIn("current_risk", summary)
        self.assertIn("trend_scores", summary)

        analytics = get_admin_analytics()
        self.assertIn("total_teachers", analytics)
        self.assertIn("total_checkins", analytics)
        self.assertIn("risk_counts", analytics)
        self.assertIn("department_overview", analytics)
        self.assertIn("scatter_data", analytics)

    def test_model_prediction(self):
        """Test ML model inference on valid features."""
        input_data = pd.DataFrame([[2.0] * len(feature_names)], columns=feature_names)
        pred = model.predict(input_data)[0]
        self.assertIn(pred, ["Low", "Moderate", "High"])

        probs = model.predict_proba(input_data)[0]
        self.assertEqual(len(probs), len(model.classes_))
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_route_home(self):
        """Test home route GET /."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"EduCare", response.data)

    def test_api_predict_success(self):
        """Test POST /api/predict with valid payload."""
        payload = {feat: 3.5 for feat in feature_names}
        payload.update({
            "teacher_id": "T-101",
            "teacher_name": "Gauravi Gupta",
            "department": "Science",
            "notes": "Testing API endpoint"
        })

        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertIn(data.get("prediction"), ["Low", "Moderate", "High"])
        self.assertIn("probabilities", data)
        self.assertIn("recommendations", data)
        self.assertIn("dashboard_summary", data)

    def test_api_predict_validation_failure(self):
        """Test POST /api/predict with invalid or out-of-range value."""
        # Missing field
        payload = {feat: 3.0 for feat in feature_names}
        del payload["role_overload"]

        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data.get("success"))

        # Out-of-bounds value (> 5)
        payload["role_overload"] = 8.5
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_api_history(self):
        """Test GET /api/history."""
        response = self.client.get("/api/history?teacher_id=T-101")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertIsInstance(data.get("history"), list)

    def test_api_dashboard_stats(self):
        """Test GET /api/dashboard-stats."""
        response = self.client.get("/api/dashboard-stats?teacher_id=T-101")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertIn("summary", data)

    def test_api_admin_stats(self):
        """Test GET /api/admin-stats."""
        response = self.client.get("/api/admin-stats")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertIn("analytics", data)

    def test_api_export_report(self):
        """Test GET /api/export-report CSV generation."""
        response = self.client.get("/api/export-report")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn(b"EDUCARE - TEACHER WELL-BEING & BURNOUT ANALYTICS REPORT", response.data)
        self.assertIn(b"EXECUTIVE SUMMARY", response.data)

    def test_route_predict_form_post(self):
        """Test standard HTML form submission POST /predict."""
        formData = {feat: "3.0" for feat in feature_names}
        formData.update({
            "teacher_id": "T-101",
            "teacher_name": "Gauravi Gupta",
            "department": "Science",
            "notes": "Testing form post"
        })
        response = self.client.post("/predict", data=formData)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"EduCare", response.data)

    def test_template_renders_all_seven_features(self):
        """Test that the rendered home page contains all 7 model features."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        for feat in feature_names:
            self.assertIn(f'data-feature="{feat}"', html)

    def test_api_predict_boundary_values(self):
        """Test API prediction with boundary values 1.0 (minimum) and 5.0 (maximum)."""
        # Test minimum boundary 1.0
        min_payload = {feat: 1.0 for feat in feature_names}
        res_min = self.client.post("/api/predict", data=json.dumps(min_payload), content_type="application/json")
        self.assertEqual(res_min.status_code, 200)
        data_min = json.loads(res_min.data)
        self.assertEqual(data_min["prediction"], "Low")

        # Test maximum boundary 5.0
        max_payload = {feat: 5.0 for feat in feature_names}
        res_max = self.client.post("/api/predict", data=json.dumps(max_payload), content_type="application/json")
        self.assertEqual(res_max.status_code, 200)
        data_max = json.loads(res_max.data)
        self.assertEqual(data_max["prediction"], "High")

    def test_api_predict_non_numeric(self):
        """Test API rejection of non-numeric input."""
        payload = {feat: 3.0 for feature in feature_names for feat in [feature]}
        payload["role_overload"] = "not-a-number"
        response = self.client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data.get("success"))

    def test_teacher_registration_and_login(self):
        """Test teacher registration, duplicate detection, and authentication flow."""
        import uuid
        unique_suffix = uuid.uuid4().hex[:6]
        tid = f"T-REG-{unique_suffix}"
        email = f"teacher_{unique_suffix}@educare.edu"

        # 1. Register new teacher
        reg_payload = {
            "teacher_id": tid,
            "name": f"Registered Teacher {unique_suffix}",
            "email": email,
            "password": "securePassword123",
            "department": "Science"
        }
        reg_res = self.client.post("/api/register", data=json.dumps(reg_payload), content_type="application/json")
        self.assertEqual(reg_res.status_code, 200)
        reg_data = json.loads(reg_res.data)
        self.assertTrue(reg_data["success"])
        self.assertEqual(reg_data["teacher"]["teacher_id"], tid)

        # 2. Re-registering duplicate should fail
        dup_res = self.client.post("/api/register", data=json.dumps(reg_payload), content_type="application/json")
        self.assertEqual(dup_res.status_code, 400)
        dup_data = json.loads(dup_res.data)
        self.assertFalse(dup_data["success"])

        # 3. Login with valid credentials
        login_payload = {
            "identifier": email,
            "password": "securePassword123"
        }
        login_res = self.client.post("/api/login", data=json.dumps(login_payload), content_type="application/json")
        self.assertEqual(login_res.status_code, 200)
        login_data = json.loads(login_res.data)
        self.assertTrue(login_data["success"])
        self.assertEqual(login_data["teacher"]["email"], email)

        # 4. Login with invalid password
        bad_login = {
            "identifier": email,
            "password": "wrongPassword"
        }
        bad_res = self.client.post("/api/login", data=json.dumps(bad_login), content_type="application/json")
        self.assertEqual(bad_res.status_code, 401)
        bad_data = json.loads(bad_res.data)
        self.assertFalse(bad_data["success"])

        # 5. Check current session user
        user_res = self.client.get("/api/current-user")
        self.assertEqual(user_res.status_code, 200)
        user_data = json.loads(user_res.data)
        self.assertTrue(user_data["success"])

        # 6. Logout
        logout_res = self.client.post("/api/logout")
        self.assertEqual(logout_res.status_code, 200)
        logout_data = json.loads(logout_res.data)
        self.assertTrue(logout_data["success"])

    def test_get_teachers_list(self):
        """Test GET /api/teachers endpoint."""
        res = self.client.get("/api/teachers")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertIsInstance(data["teachers"], list)
        self.assertGreaterEqual(len(data["teachers"]), 1)


if __name__ == "__main__":
    unittest.main()


