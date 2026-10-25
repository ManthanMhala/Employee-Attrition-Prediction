
from pathlib import Path
import io
import json
import sqlite3

import joblib
import pandas as pd
import plotly.express as px

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.secret_key = "attritioniq-local-development-key"

DATA_PATH = BASE_DIR / "dataset" / "employee.csv"
MODEL_PATH = BASE_DIR / "model" / "attrition_model.pkl"
INFO_PATH = BASE_DIR / "model" / "model_info.json"
DB_PATH = BASE_DIR / "prediction_history.db"

TARGET = "Attrition"

IGNORED_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

# =========================================================
# 2. DATABASE
# =========================================================


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_label TEXT,
                department TEXT,
                model_name TEXT,
                probability REAL,
                risk_level TEXT,
                prediction TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


init_db()

# =========================================================
# 3. LOAD DATA AND MODEL
# =========================================================


def load_resources():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "'python model/train_model.py' first."
        )

    if not INFO_PATH.exists():
        raise FileNotFoundError(
            f"Model information not found: {INFO_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    if TARGET not in data.columns:
        raise ValueError(
            "Dataset must contain an 'Attrition' column."
        )

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)

    model = joblib.load(MODEL_PATH)

    return data, model, info


DATA = None
MODEL = None
MODEL_INFO = {}
STARTUP_ERROR = None

try:
    DATA, MODEL, MODEL_INFO = load_resources()
except (FileNotFoundError, ValueError, OSError, EOFError) as error:
    STARTUP_ERROR = str(error)
    print(f"Startup notice: {STARTUP_ERROR}", flush=True)


@app.context_processor
def common_template_values():
    return {
        "model_name": MODEL_INFO.get(
            "best_model",
            "Model not trained",
        )
    }


# =========================================================
# 4. SHARED HELPER FUNCTIONS
# =========================================================


def require_model():
    if DATA is None or MODEL is None:
        flash(
            STARTUP_ERROR or "The model is not available yet.",
            "error",
        )
        return False

    return True


def get_history(limit=None):
    with get_connection() as connection:
        if limit is not None:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                ORDER BY id DESC
                """
            ).fetchall()

    return [dict(row) for row in rows]


def get_risk_level(probability):
    if probability >= 0.65:
        return "High"

    if probability >= 0.35:
        return "Medium"

    return "Low"


def make_chart(fig):
    """Apply consistent styling and return responsive Plotly HTML."""

    fig.update_layout(
        template="plotly_white",
        height=360,
        autosize=True,
        title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Arial, sans-serif",
            "size": 12,
            "color": "#526078",
        },
        margin={
            "l": 55,
            "r": 20,
            "t": 20,
            "b": 65,
        },
        showlegend=False,
        hoverlabel={
            "bgcolor": "#17243a",
            "font_color": "white",
        },
    )

    fig.update_xaxes(
        showgrid=False,
        automargin=True,
        tickfont={"size": 11},
        linecolor="#e8edf5",
    )

    fig.update_yaxes(
        gridcolor="#edf0f6",
        zeroline=False,
        automargin=True,
        tickfont={"size": 11},
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displayModeBar": False,
        },
    )


def attrition_rate_by_column(data, column):
    """Calculate the historical attrition percentage per group."""

    working = data.copy()

    working["_left"] = (
        working[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    grouped = (
        working.groupby(column, dropna=False)["_left"]
        .mean()
        .mul(100)
        .reset_index()
    )

    grouped["_left"] = grouped["_left"].round(2)

    return grouped


# =========================================================
# 5. DASHBOARD
# =========================================================


@app.route("/")
@app.route("/dashboard")
def index():
    if DATA is None:
        return render_template(
            "index.html",
            setup_error=True,
        )

    total_employees = len(DATA)

    attrition_values = (
        DATA[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    employees_left = int(attrition_values.eq("yes").sum())

    attrition_rate = (
        employees_left / total_employees * 100
        if total_employees
        else 0
    )

    saved_predictions = get_history()

    high_risk_count = sum(
        row["risk_level"] == "High"
        for row in saved_predictions
    )

    department_chart = None

    if "Department" in DATA.columns:
        department_data = attrition_rate_by_column(
            DATA,
            "Department",
        )

        fig = px.bar(
            department_data,
            x="Department",
            y="_left",
            color="Department",
            color_discrete_sequence=[
                "#5378ff",
                "#8870ff",
                "#36b89a",
                "#f0b34d",
            ],
            labels={
                "Department": "Department",
                "_left": "Attrition rate (%)",
            },
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Attrition rate (%)",
        )

        department_chart = make_chart(fig)

    return render_template(
        "index.html",
        setup_error=False,
        employee_count=total_employees,
        attrition_count=employees_left,
        attrition_rate=round(attrition_rate, 1),
        high_risk=high_risk_count,
        dept_chart=department_chart,
        recent=get_history(5),
    )


# =========================================================
# 6. EMPLOYEE RISK PREDICTION
# =========================================================


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not require_model():
        return redirect(url_for("index"))

    features = MODEL_INFO.get("features", [])

    if not features:
        flash("The model feature list is empty.", "error")
        return redirect(url_for("index"))

    form_fields = []

    for column in features:
        if column not in DATA.columns:
            continue

        series = DATA[column]

        if pd.api.types.is_numeric_dtype(series):
            median = series.median()

            default_value = (
                float(median)
                if pd.notna(median)
                else 0.0
            )

            form_fields.append({
                "name": column,
                "label": column.replace("_", " "),
                "kind": "number",
                "options": [],
                "default": default_value,
            })

        else:
            options = sorted(
                series.dropna().astype(str).unique().tolist()
            )

            form_fields.append({
                "name": column,
                "label": column.replace("_", " "),
                "kind": "select",
                "options": options,
                "default": options[0] if options else "",
            })

    if request.method == "GET":
        return render_template(
            "prediction.html",
            fields=form_fields,
        )

    values = {}

    try:
        for field in form_fields:
            name = field["name"]
            value = request.form.get(name, "").strip()

            if field["kind"] == "number":
                if not value:
                    raise ValueError(
                        f"Please enter a value for {field['label']}."
                    )

                number = float(value)

                if not pd.notna(number):
                    raise ValueError(
                        f"Please enter a valid number for {field['label']}."
                    )

                values[name] = number

            else:
                if value not in field["options"]:
                    raise ValueError(
                        f"Please select a valid {field['label']}."
                    )

                values[name] = value

        row = pd.DataFrame(
            [values],
            columns=features,
        )

        # The saved pipeline performs the same preprocessing
        # that was used during model training.
        probability = float(
            MODEL.predict_proba(row)[0][1]
        )

        level = get_risk_level(probability)

        prediction_label = (
            "Likely to Leave"
            if probability >= 0.5
            else "Likely to Stay"
        )

        employee_label = (
            request.form.get("employee_label", "").strip()
            or "Employee"
        )[:100]

        department = str(
            values.get("Department", "Unknown")
        )

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    employee_label,
                    department,
                    model_name,
                    probability,
                    risk_level,
                    prediction
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    employee_label,
                    department,
                    MODEL_INFO.get("best_model", "Unknown"),
                    probability,
                    level,
                    prediction_label,
                ),
            )

        return render_template(
            "result.html",
            employee_label=employee_label,
            department=department,
            probability=round(probability * 100, 1),
            risk_level=level,
            prediction=prediction_label,
        )

    except (ValueError, TypeError) as error:
        flash(str(error), "error")

        return render_template(
            "prediction.html",
            fields=form_fields,
        ), 400


# =========================================================
# 7. PREDICTION HISTORY
# =========================================================


@app.route("/history")
def history():
    return render_template(
        "history.html",
        records=get_history(),
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with get_connection() as connection:
        connection.execute("DELETE FROM predictions")

    flash(
        "Prediction history cleared successfully.",
        "success",
    )

    return redirect(url_for("history"))


@app.route("/export")
def export_history():
    records = get_history()

    if not records:
        flash(
            "There are no predictions to export yet.",
            "error",
        )
        return redirect(url_for("history"))

    output = io.StringIO()

    pd.DataFrame(records).to_csv(
        output,
        index=False,
    )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=attrition_history.csv"
        },
    )


# =========================================================
# 8. ANALYTICS
# =========================================================


@app.route("/analytics")
def analytics():
    if DATA is None:
        return render_template(
            "analytics.html",
            setup_error=True,
            department_chart=None,
            overtime_chart=None,
            satisfaction_chart=None,
        )

    charts = {
        "department_chart": None,
        "overtime_chart": None,
        "satisfaction_chart": None,
    }

    # Department attrition
    if "Department" in DATA.columns:
        department_data = attrition_rate_by_column(
            DATA,
            "Department",
        )

        fig = px.bar(
            department_data,
            x="Department",
            y="_left",
            color="Department",
            color_discrete_sequence=[
                "#5378ff",
                "#8870ff",
                "#36b89a",
                "#f0b34d",
            ],
            labels={
                "Department": "Department",
                "_left": "Attrition rate (%)",
            },
        )

        fig.update_layout(
            yaxis_title="Attrition rate (%)",
        )

        charts["department_chart"] = make_chart(fig)

    # Overtime attrition
    if "OverTime" in DATA.columns:
        overtime_data = attrition_rate_by_column(
            DATA,
            "OverTime",
        )

        fig = px.bar(
            overtime_data,
            x="OverTime",
            y="_left",
            color="OverTime",
            color_discrete_sequence=[
                "#5378ff",
                "#36b89a",
            ],
            labels={
                "OverTime": "Overtime",
                "_left": "Attrition rate (%)",
            },
        )

        fig.update_layout(
            yaxis_title="Attrition rate (%)",
        )

        charts["overtime_chart"] = make_chart(fig)

    # Job satisfaction attrition
    if "JobSatisfaction" in DATA.columns:
        satisfaction_data = attrition_rate_by_column(
            DATA,
            "JobSatisfaction",
        )

        fig = px.bar(
            satisfaction_data,
            x="JobSatisfaction",
            y="_left",
            color="_left",
            color_continuous_scale=[
                "#c9d4ff",
                "#5378ff",
                "#7250d8",
            ],
            labels={
                "JobSatisfaction": "Job satisfaction score",
                "_left": "Attrition rate (%)",
            },
        )

        fig.update_layout(
            yaxis_title="Attrition rate (%)",
            coloraxis_showscale=False,
        )

        charts["satisfaction_chart"] = make_chart(fig)

    return render_template(
        "analytics.html",
        setup_error=False,
        **charts,
    )


# =========================================================
# 9. START FLASK
# =========================================================


if __name__ == "__main__":
    print("Starting AttritionIQ...", flush=True)
    print(f"Project directory: {BASE_DIR}", flush=True)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
