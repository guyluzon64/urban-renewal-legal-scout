from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="Urban Renewal Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_PATH = PROCESSED_DIR / "urban_renewal_scored.csv"
CITY_SUMMARY_PATH = PROCESSED_DIR / "urban_renewal_city_scored_summary.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ml_similarity_pipeline.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "ml_similarity_metadata.json"

SHORT_DISCLAIMER = (
    "המידע במערכת מבוסס על מקורות מידע ציבוריים ועשוי להיות חלקי, לא מעודכן או לא מדויק. "
    "הציונים המוצגים הם אינדיקטיביים בלבד ואינם מהווים ייעוץ משפטי, תכנוני, שמאי, עסקי או נדל״ני."
)
FULL_DISCLAIMER = (
    "המידע במערכת מבוסס על מקורות מידע ציבוריים ועשוי להיות חלקי, לא מעודכן או לא מדויק. "
    "הציונים המוצגים הם אינדיקטיביים בלבד ואינם מהווים ייעוץ משפטי, תכנוני, שמאי, עסקי או נדל״ני. "
    "כל שימוש במידע מחייב בדיקה עצמאית מול המקורות הרשמיים והמסמכים התכנוניים הרלוונטיים."
)
ML_DISCLAIMER = (
    "Indicative Advancement Score הוא מדד דמיון אינדיקטיבי בלבד למתחמים שמופיעים במידע הציבורי כמתקדמים יותר. "
    "הוא אינו תחזית, אינו הסתברות לאישור ואינו ייעוץ משפטי, תכנוני או עסקי."
)


USAGE_RIGHTS_TOP_HTML = (
    '<strong>אודות הכלי:</strong> '
    'מערכת חקירה מבוססת מידע ציבורי לבחינה ראשונית של מתחמי התחדשות עירונית בישראל. '
    'המערכת מרכזת שדות תכנוניים, קישורים למקורות ציבוריים וציונים אינדיקטיביים לצורכי מיון, תיעדוף ובדיקת המשך.'
)
USAGE_RIGHTS_FULL_MARKDOWN = """
**אודות הכלי**

Urban Renewal Intelligence Dashboard הוא כלי חקירה מבוסס מידע ציבורי לבחינה ראשונית של מתחמי התחדשות עירונית בישראל. הכלי מאפשר חיפוש, סינון, השוואה והפניה למקורות מידע ציבוריים כגון מבא״ת ומקורות ממשלתיים נוספים.

**הבהרת שימוש**

המערכת מיועדת לתמיכה בבדיקת היתכנות ראשונית ובתיעדוף בדיקות המשך. אין להסתמך על המידע ללא אימות מול המקורות הרשמיים, מסמכי התוכנית, החלטות ועדה, פרוטוקולים והיתרים רלוונטיים.
"""
USAGE_RIGHTS_FOOTER_HTML = (
    'פותח על ידי גיא לוזון | אבטיפוס מודיעין נדל״ני מבוסס מידע ציבורי'
)
HEBREW_LABELS = {
    "record_id": "מזהה רשומה",
    "source_record_id": "מספר מתחם מקור",
    "city": "עיר",
    "city_code": "סמל יישוב",
    "neighborhood": "שכונה",
    "street_or_area": "רחוב / אזור",
    "complex_name": "שם מתחם",
    "plan_number": "מספר תוכנית",
    "renewal_type": "מסלול",
    "planning_status_raw": "סטטוס תכנוני ציבורי",
    "planning_status_normalized": "סטטוס מנורמל",
    "declared_complex": "מתחם מוכרז",
    "existing_units": "יח״ד קיימות",
    "additional_units": "יח״ד תוספתיות",
    "proposed_units": "יח״ד מוצעות",
    "permits_total": "סה״כ היתרים",
    "declaration_date": "תאריך הכרזה",
    "validity_year": "שנת תוקף",
    "in_execution": "בביצוע",
    "mavat_url": "קישור למבא״ת",
    "map_url": "קישור מפה",
    "source_url": "קישור מקור",
    "source_name": "שם מקור",
    "source_type": "סוג מקור",
    "original_resource_id": "מזהה משאב מקור",
    "last_updated": "עדכון אחרון",
    "confidence_level": "רמת אמינות נתונים",
    "data_confidence_score": "Data Confidence Score",
    "data_quality_flag": "דגל איכות נתונים",
    "lawyer_note": "הערת בדיקה",
    "planning_maturity_score": "Planning Maturity Score",
    "source_strength_score": "Source Strength Score",
    "scale_score": "ציון היקף",
    "project_momentum_score": "Project Momentum Score",
    "project_momentum_label": "רמת בשלות אינדיקטיבית",
    "project_momentum_explanation": "הסבר ניקוד",
    "ml_advancement_score": "Indicative Advancement Score",
    "ml_advancement_label": "רמת התקדמות אינדיקטיבית",
    "dashboard_note": "הערת מערכת",
}

DEFAULT_COLUMNS: dict[str, Any] = {
    "record_id": pd.NA,
    "source_record_id": pd.NA,
    "city": pd.NA,
    "city_code": pd.NA,
    "neighborhood": pd.NA,
    "street_or_area": pd.NA,
    "complex_name": pd.NA,
    "plan_number": pd.NA,
    "renewal_type": pd.NA,
    "planning_status_raw": pd.NA,
    "planning_status_normalized": pd.NA,
    "declared_complex": pd.NA,
    "existing_units": pd.NA,
    "additional_units": pd.NA,
    "proposed_units": pd.NA,
    "permits_total": pd.NA,
    "declaration_date": pd.NA,
    "validity_year": pd.NA,
    "in_execution": pd.NA,
    "mavat_url": pd.NA,
    "map_url": pd.NA,
    "source_name": pd.NA,
    "source_url": pd.NA,
    "source_type": pd.NA,
    "confidence_level": pd.NA,
    "data_confidence_score": pd.NA,
    "data_quality_flag": "OK",
    "lawyer_note": pd.NA,
    "has_plan_number": False,
    "has_mavat_url": False,
    "has_map_url": False,
    "has_existing_units": False,
    "has_proposed_units": False,
    "has_permits": False,
    "has_quality_issue": False,
    "declaration_year": pd.NA,
    "years_since_declaration": pd.NA,
    "planning_maturity_score": pd.NA,
    "source_strength_score": pd.NA,
    "scale_score": pd.NA,
    "data_quality_penalty": 0,
    "project_momentum_score": pd.NA,
    "project_momentum_label": "UNKNOWN",
    "project_momentum_explanation": pd.NA,
    "ml_advancement_score": pd.NA,
    "ml_advancement_label": "ML_NOT_AVAILABLE",
    "ml_model_used": pd.NA,
    "ml_feature_set_used": pd.NA,
    "ml_score_warning": ML_DISCLAIMER,
    "dashboard_note": pd.NA,
    "advanced_project_label": pd.NA,
    "proposed_to_existing_ratio": pd.NA,
    "additional_to_existing_ratio": pd.NA,
}

NUMERIC_COLUMNS = [
    "city_code", "existing_units", "additional_units", "proposed_units", "permits_total", "validity_year",
    "declaration_year", "years_since_declaration",
    "data_confidence_score", "planning_maturity_score", "source_strength_score",
    "scale_score", "data_quality_penalty", "project_momentum_score", "ml_advancement_score",
    "proposed_to_existing_ratio", "additional_to_existing_ratio",
]
BOOLEAN_COLUMNS = [
    "has_plan_number", "has_mavat_url", "has_map_url", "has_existing_units",
    "has_proposed_units", "has_permits", "has_quality_issue", "in_execution",
]
TABLE_COLUMNS = [
    "city", "complex_name", "plan_number", "renewal_type", "planning_status_raw",
    "planning_status_normalized", "project_momentum_score", "project_momentum_label",
    "ml_advancement_score", "ml_advancement_label", "confidence_level", "data_quality_flag",
]
SUMMARY_COLUMNS = [
    "city", "num_records", "avg_project_momentum_score", "median_project_momentum_score",
    "num_very_high_momentum", "num_high_momentum", "num_moderate_momentum", "num_low_momentum",
    "avg_ml_advancement_score", "num_advanced_public_status", "num_plan_approved",
    "num_permit_approved", "num_construction", "num_with_plan_number", "num_with_mavat_url",
    "num_with_quality_issues", "total_existing_units", "total_proposed_units",
]
SUMMARY_HEBREW_LABELS = {
    "city": "עיר",
    "num_records": "מספר מתחמים",
    "avg_project_momentum_score": "ממוצע Project Momentum Score",
    "median_project_momentum_score": "חציון Project Momentum Score",
    "num_very_high_momentum": "רמת בשלות גבוהה מאוד",
    "num_high_momentum": "רמת בשלות גבוהה",
    "num_moderate_momentum": "רמת בשלות בינונית",
    "num_low_momentum": "רמת בשלות נמוכה",
    "avg_ml_advancement_score": "ממוצע Indicative Advancement Score",
    "num_advanced_public_status": "סטטוס ציבורי מתקדם",
    "num_plan_approved": "תוכנית מאושרת",
    "num_permit_approved": "אחרי רישוי",
    "num_construction": "במימוש / ביצוע",
    "num_with_plan_number": "עם מספר תוכנית",
    "num_with_mavat_url": "עם קישור למבא״ת",
    "num_with_quality_issues": "עם דגל איכות נתונים",
    "total_existing_units": "סה״כ יח״ד קיימות",
    "total_proposed_units": "סה״כ יח״ד מוצעות",
}


PLANNING_MATURITY_MAP = {
    "UNKNOWN": 0,
    "POLICY_AREA_ONLY": 10,
    "DECLARED_COMPLEX": 20,
    "PLAN_IN_PROGRESS": 40,
    "PLAN_DEPOSITED": 55,
    "PLAN_APPROVED": 70,
    "PERMIT_REQUESTED": 80,
    "PERMIT_APPROVED": 90,
    "CONSTRUCTION": 95,
    "COMPLETED": 100,
}
QUALITY_PENALTIES = {
    "MISSING_STATUS": 10,
    "MISSING_LINKS": 10,
    "MISSING_CITY": 15,
    "MISSING_COMPLEX_NAME": 15,
    "INVALID_NUMERIC_UNITS": 20,
    "NEGATIVE_UNIT_VALUE": 20,
    "UNIT_LOGIC_ANOMALY": 20,
    "COLUMN_SHIFT_OR_SOURCE_ANOMALY": 30,
    "DUPLICATE_RECORD_ID": 25,
}
UPLOAD_ML_WARNING = (
    "Indicative Advancement Score is calculated for uploaded records when the saved local model is available. "
    "If unavailable, Project Momentum Score remains available."
)

st.markdown("""
<style>
/* Theme-safe Streamlit refinements: use Streamlit CSS variables so light and dark mode stay readable. */
html, body, [class*="css"], .stApp {
    direction: rtl;
    text-align: right;
    font-family: "Segoe UI", "Arial", "Noto Sans Hebrew", sans-serif;
}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { direction: rtl; text-align: right; }
.main-title {
    font-size: 2.35rem;
    font-weight: 850;
    color: var(--text-color) !important;
    margin-bottom: 0.15rem;
}
.welcome-line {
    font-size: 1.15rem;
    color: var(--text-color) !important;
    font-weight: 750;
    margin: 0.15rem 0 0.25rem 0;
}
.subtitle {
    font-size: 1.1rem;
    color: var(--text-color) !important;
    margin-bottom: 0.35rem;
    font-weight: 600;
    opacity: 0.88;
}
.context-caption {
    color: var(--text-color) !important;
    font-weight: 600;
    margin-bottom: 1rem;
    opacity: 0.86;
}
.creator-line {
    color: var(--text-color) !important;
    font-size: 0.92rem;
    font-weight: 600;
    margin: 0.2rem 0 0.85rem 0;
    opacity: 0.72;
}
.soft-card, .warning-card, .success-card, .professional-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.32);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin: 0.65rem 0;
    color: var(--text-color) !important;
    line-height: 1.75;
}
.professional-card {
    padding: 1.05rem 1.2rem;
}
.professional-card h3 {
    color: var(--text-color) !important;
    font-size: 1.22rem;
    margin: 0 0 0.55rem 0;
}
.professional-card p,
.professional-card li {
    color: var(--text-color) !important;
    opacity: 0.94;
}
.professional-card ul {
    margin: 0.45rem 1.25rem 0 0;
    padding: 0;
}
.warning-card {
    border-inline-start: 4px solid #d97706;
}
.success-card {
    border-inline-start: 4px solid #059669;
}
.soft-card *, .warning-card *, .success-card *, .professional-card * { color: inherit !important; }
.stMetric, div[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.28);
    border-radius: 8px;
    padding: 0.85rem;
    box-shadow: none;
    color: var(--text-color) !important;
}
.stMetric *, div[data-testid="metric-container"] *,
div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
    color: var(--text-color) !important;
    opacity: 1 !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] p,
div[data-testid="metric-container"] span,
div[data-testid="metric-container"] div,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] p,
div[data-testid="stMetric"] span,
div[data-testid="stMetric"] div {
    color: var(--text-color) !important;
    opacity: 1 !important;
}
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"] * {
    color: var(--text-color) !important;
    opacity: 1 !important;
}
.rights-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.32);
    color: var(--text-color) !important;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0 0.8rem 0;
    line-height: 1.65;
}
.rights-card *, .rights-footer * { color: inherit !important; opacity: 1 !important; }
.rights-card a, .rights-footer a { color: var(--primary-color) !important; font-weight: 700; }
.rights-footer {
    margin-top: 1.5rem;
    padding: 0.8rem 1rem;
    border-top: 1px solid rgba(127, 127, 127, 0.32);
    color: var(--text-color) !important;
    font-size: 0.9rem;
    text-align: center;
    opacity: 0.86;
}
.upload-callout {
    background: var(--secondary-background-color);
    border: 1px solid rgba(127, 127, 127, 0.32);
    color: var(--text-color) !important;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0 1rem 0;
    font-weight: 700;
}
.upload-callout * { color: var(--text-color) !important; opacity: 1 !important; }
div[data-testid="stAlert"] * {
    opacity: 1 !important;
}
/* Keep complex Streamlit widgets theme-owned, only enforce RTL and contrast inheritance. */
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] * {
    direction: rtl;
}
div[data-testid="stTabs"] button,
div[data-testid="stExpander"] summary,
div[data-testid="stDataFrame"] {
    color: var(--text-color) !important;
}
a { color: var(--primary-color); font-weight: 650; }
</style>
""", unsafe_allow_html=True)


def has_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null", "-", "--", "<na>"}



def clean_text(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "-", "--", "<na>"}:
        return pd.NA
    return " ".join(text.split())


def clean_id(value: Any) -> Any:
    value = clean_text(value)
    if pd.isna(value):
        return pd.NA
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def safe_text(value: Any, fallback: str = "לא זמין") -> str:
    return str(value).strip() if has_value(value) else fallback


def safe_url(value: Any) -> str | None:
    if not has_value(value):
        return None
    url = str(value).strip()
    return url if url.startswith(("http://", "https://")) else None


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "כן"])


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in DEFAULT_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = normalize_bool(df[col])
    df["data_quality_flag"] = df["data_quality_flag"].fillna("OK").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_scored_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("לא נמצא קובץ הנתונים data/processed/urban_renewal_scored.csv. יש לוודא שקובץ הנתונים המעובד קיים לפני הרצת המערכת.")
    return ensure_columns(pd.read_csv(INPUT_PATH, encoding="utf-8-sig"))


@st.cache_data(show_spinner=False)
def load_city_summary() -> pd.DataFrame | None:
    if CITY_SUMMARY_PATH.exists():
        return pd.read_csv(CITY_SUMMARY_PATH, encoding="utf-8-sig")
    return None


@st.cache_resource(show_spinner=False)
def load_ml_similarity_model():
    if joblib is None:
        return None, None, "joblib is not installed."
    if not MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        return None, None, "Saved advancement model files not found."
    try:
        model = joblib.load(MODEL_PATH)
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata, None
    except Exception as exc:
        return None, None, str(exc)


def normalize_yes_no_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"כן", "true", "1", "yes", "y"}:
        return True
    if text in {"לא", "false", "0", "no", "n"}:
        return False
    return pd.NA


def parse_ckan_datastore_json(uploaded_file: Any) -> tuple[pd.DataFrame | None, str | None, Any, int, str | None]:
    try:
        text = uploaded_file.getvalue().decode("utf-8-sig")
        payload = json.loads(text)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            return None, None, None, 0, "הקובץ לא נראה כמו JSON תקין של data.gov.il CKAN datastore_search."
        result = payload.get("result")
        if not isinstance(result, dict):
            return None, None, None, 0, "הקובץ לא כולל result תקין."
        records = result.get("records")
        if not isinstance(records, list) or not records:
            return None, result.get("resource_id"), result.get("total"), 0, "לא נמצאו רשומות בקובץ."
        return pd.DataFrame(records), result.get("resource_id"), result.get("total"), len(records), None
    except Exception as exc:
        return None, None, None, 0, f"שגיאה בקריאת הקובץ: {type(exc).__name__}: {exc}"


def normalize_planning_status_value(status_raw: Any, in_execution: Any) -> str:
    text = "" if pd.isna(status_raw) else str(status_raw)
    normalized = "UNKNOWN"
    if "תכנון ראשוני" in text:
        normalized = "PLAN_IN_PROGRESS"
    elif "תכנון סטטוטורי" in text:
        normalized = "PLAN_IN_PROGRESS"
    elif "מאושרת לפני מימוש" in text:
        normalized = "PLAN_APPROVED"
    elif "אחרי רישוי" in text:
        normalized = "PERMIT_APPROVED"
    elif "במימוש" in text:
        normalized = "CONSTRUCTION"
    if in_execution is True and normalized != "COMPLETED":
        normalized = "CONSTRUCTION"
    return normalized


def make_lawyer_note(status: Any, plan_number: Any, mavat_url: Any) -> str:
    status = safe_text(status, "UNKNOWN")
    if status == "PLAN_APPROVED":
        note = "נמצא מקור ציבורי רשמי עם אינדיקציה לתוכנית מאושרת; מומלץ לבדוק את מסמכי התוכנית במבא״ת ובמקורות הרשמיים."
    elif status == "PERMIT_APPROVED":
        note = "סטטוס מתקדם יחסית עם אינדיקציה לשלב רישוי; מומלצת בדיקה תכנונית ומשפטית מעמיקה."
    elif status == "CONSTRUCTION":
        note = "הרשומה הציבורית מצביעה על שלב מימוש או ביצוע; נדרש אימות עדכני במבא״ת, במערכות העירוניות ובמסמכי ההיתר."
    elif status == "PLAN_IN_PROGRESS":
        note = "נמצא מקור ציבורי רשמי עם אינדיקציה להליך תכנוני פעיל; מומלץ לבדוק סטטוס ציבורי עדכני ומסמכים תכנוניים."
    else:
        note = "מידע ציבורי חלקי או סטטוס תכנוני לא ברור; נדרש אימות מול מקור רשמי."
    if has_value(plan_number):
        note += " קיים מספר תוכנית; מומלץ לבדוק במערכת מינהל התכנון / תב״ע / מבא״ת."
    if safe_url(mavat_url):
        note += " קישור למבא״ת זמין."
    return note


def build_quality_flags(row: pd.Series, invalid_numeric: bool = False) -> str:
    flags: list[str] = []
    if not has_value(row.get("city")):
        flags.append("MISSING_CITY")
    if not has_value(row.get("complex_name")):
        flags.append("MISSING_COMPLEX_NAME")
    if not has_value(row.get("planning_status_raw")):
        flags.append("MISSING_STATUS")
    if not safe_url(row.get("mavat_url")) and not safe_url(row.get("map_url")):
        flags.append("MISSING_LINKS")
    if invalid_numeric:
        flags.append("INVALID_NUMERIC_UNITS")
    numeric_fields = ["existing_units", "additional_units", "proposed_units", "permits_total"]
    if any(pd.notna(row.get(col)) and row.get(col) < 0 for col in numeric_fields):
        flags.append("NEGATIVE_UNIT_VALUE")
    if pd.notna(row.get("existing_units")) and pd.notna(row.get("proposed_units")) and row.get("proposed_units") < row.get("existing_units"):
        flags.append("UNIT_LOGIC_ANOMALY")
    return "OK" if not flags else "|".join(dict.fromkeys(flags))


def quality_penalty_from_flag(flag: Any) -> int:
    if not has_value(flag) or str(flag) == "OK":
        return 0
    parts = [part.strip() for part in str(flag).split("|") if part.strip()]
    return int(min(50, sum(QUALITY_PENALTIES.get(part, 0) for part in parts)))


def label_score(score: Any) -> str:
    if pd.isna(score):
        return "UNKNOWN"
    score = float(score)
    if score >= 81:
        return "VERY_HIGH"
    if score >= 61:
        return "HIGH"
    if score >= 31:
        return "MODERATE"
    return "LOW"


def standardize_uploaded_urban_renewal_records(raw_uploaded_df: pd.DataFrame, resource_id: Any) -> pd.DataFrame:
    now = datetime.now().isoformat(timespec="seconds")
    df = pd.DataFrame(index=raw_uploaded_df.index)
    mapping = {
        "source_record_id": "MisparMitham",
        "city": "Yeshuv",
        "city_code": "SemelYeshuv",
        "complex_name": "ShemMitcham",
        "existing_units": "YachadKayam",
        "additional_units": "YachadTosafti",
        "proposed_units": "YachadMutza",
        "declaration_date": "TaarichHachraza",
        "plan_number": "MisparTochnit",
        "mavat_url": "KishurLatar",
        "permits_total": "SachHeterim",
        "map_url": "KishurLaMapa",
        "renewal_type": "Maslul",
        "validity_year": "ShnatMatanTokef",
        "in_execution": "Bebitzua",
        "planning_status_raw": "Status",
    }
    invalid_numeric = pd.Series(False, index=raw_uploaded_df.index)
    for target_col, source_col in mapping.items():
        df[target_col] = raw_uploaded_df[source_col] if source_col in raw_uploaded_df.columns else pd.NA
    for col in ["source_record_id", "city", "complex_name", "plan_number", "renewal_type", "planning_status_raw", "mavat_url", "map_url"]:
        df[col] = df[col].map(clean_text)
    df["source_record_id"] = df["source_record_id"].map(clean_id)
    for col in ["existing_units", "additional_units", "proposed_units", "permits_total", "validity_year", "city_code"]:
        source_values = df[col].copy()
        non_empty = source_values.map(has_value)
        converted = pd.to_numeric(source_values.astype("string").str.replace(",", "", regex=False), errors="coerce")
        invalid_numeric = invalid_numeric | (non_empty & converted.isna())
        df[col] = converted
    parsed_dates = pd.to_datetime(df["declaration_date"], dayfirst=True, errors="coerce")
    df["declaration_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    df.loc[parsed_dates.isna(), "declaration_date"] = pd.NA
    df["in_execution"] = df["in_execution"].map(normalize_yes_no_value)
    df["planning_status_normalized"] = [normalize_planning_status_value(status, execution) for status, execution in zip(df["planning_status_raw"], df["in_execution"])]
    df["record_id"] = df["source_record_id"].map(lambda value: f"UR_{value}" if has_value(value) else pd.NA)
    missing_record_id = df["record_id"].isna()
    df.loc[missing_record_id, "record_id"] = [f"UPLOAD_{idx}" for idx in df.index[missing_record_id]]
    df["source_name"] = "data.gov.il uploaded JSON"
    df["source_url"] = "uploaded_session_file"
    df["source_type"] = "uploaded_official_public_json"
    df["original_resource_id"] = resource_id
    df["last_updated"] = now
    df["declared_complex"] = True
    df["data_quality_flag"] = [build_quality_flags(row, bool(invalid_numeric.loc[idx])) for idx, row in df.iterrows()]
    df["has_quality_issue"] = df["data_quality_flag"] != "OK"
    df["data_confidence_score"] = 25
    df.loc[df["city"].map(has_value), "data_confidence_score"] += 15
    df.loc[df["complex_name"].map(has_value), "data_confidence_score"] += 15
    df.loc[df["planning_status_raw"].map(has_value), "data_confidence_score"] += 15
    df.loc[df["plan_number"].map(has_value), "data_confidence_score"] += 10
    df.loc[df["mavat_url"].map(lambda value: bool(safe_url(value))), "data_confidence_score"] += 10
    df.loc[df["map_url"].map(lambda value: bool(safe_url(value))), "data_confidence_score"] += 5
    has_numeric_units = df[["existing_units", "proposed_units", "additional_units"]].notna().any(axis=1)
    df.loc[has_numeric_units, "data_confidence_score"] += 5
    df.loc[df["has_quality_issue"], "data_confidence_score"] -= 20
    df.loc[df["planning_status_normalized"] == "UNKNOWN", "data_confidence_score"] -= 10
    df["data_confidence_score"] = df["data_confidence_score"].clip(0, 100)
    df["confidence_level"] = pd.cut(df["data_confidence_score"], bins=[-1, 49, 79, 100], labels=["LOW", "MEDIUM", "HIGH"]).astype("string")
    df["lawyer_note"] = [make_lawyer_note(row.get("planning_status_normalized"), row.get("plan_number"), row.get("mavat_url")) for _, row in df.iterrows()]
    return ensure_columns(df)


def score_uploaded_records(uploaded_standardized_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(uploaded_standardized_df)
    df["has_plan_number"] = df["plan_number"].map(has_value)
    df["has_mavat_url"] = df["mavat_url"].map(lambda value: bool(safe_url(value)))
    df["has_map_url"] = df["map_url"].map(lambda value: bool(safe_url(value)))
    df["has_existing_units"] = df["existing_units"].notna() & (df["existing_units"] > 0)
    df["has_proposed_units"] = df["proposed_units"].notna() & (df["proposed_units"] > 0)
    df["has_permits"] = df["permits_total"].notna() & (df["permits_total"] > 0)
    df["has_quality_issue"] = df["data_quality_flag"].fillna("OK").astype(str) != "OK"
    df["planning_maturity_score"] = df["planning_status_normalized"].map(PLANNING_MATURITY_MAP).fillna(0)
    df["source_strength_score"] = 25
    df.loc[df["has_plan_number"], "source_strength_score"] += 25
    df.loc[df["has_mavat_url"], "source_strength_score"] += 25
    df.loc[df["has_map_url"], "source_strength_score"] += 15
    df.loc[df["original_resource_id"].map(has_value), "source_strength_score"] += 10
    df["source_strength_score"] = df["source_strength_score"].clip(0, 100)
    df["scale_score"] = 30
    df.loc[df["proposed_units"].notna() & (df["proposed_units"] < 100), "scale_score"] = 35
    df.loc[df["proposed_units"].between(100, 299, inclusive="both"), "scale_score"] = 50
    df.loc[df["proposed_units"].between(300, 699, inclusive="both"), "scale_score"] = 65
    df.loc[df["proposed_units"].between(700, 1499, inclusive="both"), "scale_score"] = 80
    df.loc[df["proposed_units"] >= 1500, "scale_score"] = 90
    valid_ratio = (df["existing_units"].notna()) & (df["existing_units"] > 0) & (df["proposed_units"].notna()) & (df["proposed_units"] >= 0)
    df["proposed_to_existing_ratio"] = pd.NA
    df.loc[valid_ratio, "proposed_to_existing_ratio"] = df.loc[valid_ratio, "proposed_units"] / df.loc[valid_ratio, "existing_units"]
    valid_additional_ratio = (df["existing_units"].notna()) & (df["existing_units"] > 0) & (df["additional_units"].notna()) & (df["additional_units"] >= 0)
    df["additional_to_existing_ratio"] = pd.NA
    df.loc[valid_additional_ratio, "additional_to_existing_ratio"] = df.loc[valid_additional_ratio, "additional_units"] / df.loc[valid_additional_ratio, "existing_units"]
    declaration_dates = pd.to_datetime(df["declaration_date"], errors="coerce")
    df["declaration_year"] = declaration_dates.dt.year
    df["years_since_declaration"] = datetime.now().year - df["declaration_year"]
    ratio_numeric = pd.to_numeric(df["proposed_to_existing_ratio"], errors="coerce")
    df.loc[ratio_numeric >= 2, "scale_score"] += 10
    df.loc[(ratio_numeric >= 1.5) & (ratio_numeric < 2), "scale_score"] += 5
    anomaly = df["data_quality_flag"].astype(str).str.contains("NEGATIVE_UNIT_VALUE|COLUMN_SHIFT_OR_SOURCE_ANOMALY", regex=True, na=False)
    df.loc[anomaly, "scale_score"] = 30
    df.loc[anomaly, ["proposed_to_existing_ratio", "additional_to_existing_ratio"]] = pd.NA
    df["scale_score"] = df["scale_score"].clip(0, 100)
    df["data_quality_penalty"] = df["data_quality_flag"].map(quality_penalty_from_flag)
    df["project_momentum_score"] = (0.55 * df["planning_maturity_score"].fillna(0) + 0.20 * df["source_strength_score"].fillna(50) + 0.15 * df["data_confidence_score"].fillna(50) + 0.10 * df["scale_score"].fillna(50) - df["data_quality_penalty"].fillna(0)).clip(0, 100).round(1)
    df["project_momentum_label"] = df["project_momentum_score"].map(label_score)
    df["project_momentum_explanation"] = "ציון זמני חושב במערכת לפי סטטוס ציבורי, חוזק מקור, רמת אמינות נתונים והיקף פרויקט. נדרשת בדיקה ידנית."
    df["dashboard_note"] = "רשומה שהועלתה ידנית לסשן הנוכחי. Project Momentum Score זמין; יש לבדוק מקורות רשמיים לפני מסקנה."
    df["ml_advancement_score"] = pd.NA
    df["ml_advancement_label"] = "UPLOAD_NOT_SCORED_BY_ML"
    df["ml_model_used"] = "Not applied to uploaded session data"
    df["ml_feature_set_used"] = "Not applied to uploaded session data"
    df["ml_score_warning"] = UPLOAD_ML_WARNING
    return ensure_columns(df)


def ml_similarity_label(score: Any) -> str:
    if pd.isna(score):
        return "ML_NOT_AVAILABLE_FOR_UPLOAD"
    if score >= 70:
        return "HIGH_SIMILARITY"
    if score >= 40:
        return "MEDIUM_SIMILARITY"
    return "LOW_SIMILARITY"


def set_upload_ml_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ml_advancement_score"] = pd.NA
    df["ml_advancement_label"] = "ML_NOT_AVAILABLE_FOR_UPLOAD"
    df["ml_model_used"] = "Not available for uploaded data"
    df["ml_feature_set_used"] = "Not available for uploaded data"
    df["ml_score_warning"] = "Indicative Advancement Score could not be calculated for uploaded records."
    return ensure_columns(df)


def apply_saved_ml_similarity_model(uploaded_scored_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(uploaded_scored_df)
    model, metadata, load_error = load_ml_similarity_model()
    if load_error or model is None or metadata is None:
        st.warning(f"ההעלאה הצליחה, אבל Indicative Advancement Score לא חושב: {load_error or 'model unavailable'}")
        return set_upload_ml_fallback(df)

    try:
        categorical_features = list(metadata.get("categorical_features", []))
        boolean_features = list(metadata.get("boolean_features", []))
        numeric_plus_boolean_features = list(
            metadata.get("numeric_plus_boolean_features")
            or metadata.get("numeric_features", []) + boolean_features
        )
        all_features = list(
            metadata.get("all_features")
            or categorical_features + numeric_plus_boolean_features
        )

        if not all_features:
            raise ValueError("ML metadata does not include feature names.")

        feature_df = df.copy()
        for col in all_features:
            if col not in feature_df.columns:
                feature_df[col] = pd.NA
        for col in boolean_features:
            if col in feature_df.columns:
                feature_df[col] = normalize_bool(feature_df[col]).astype(int)
        for col in numeric_plus_boolean_features:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        for col in categorical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype("string")

        probabilities = model.predict_proba(feature_df[all_features])[:, 1]
        scores = np.round(probabilities * 100, 1)
        df["ml_advancement_score"] = scores
        df["ml_advancement_label"] = [ml_similarity_label(score) for score in scores]
        df["ml_model_used"] = metadata.get("model_name", "Saved local ML model")
        df["ml_feature_set_used"] = metadata.get("feature_set_used", "leakage_safe_feature_set")
        df["ml_score_warning"] = metadata.get(
            "warning",
            "Indicative Advancement Score only. Not a prediction and not legal/planning advice.",
        )
        return ensure_columns(df)
    except Exception as exc:
        st.warning(f"ההעלאה הצליחה, אבל Indicative Advancement Score לא חושב: {type(exc).__name__}: {exc}")
        return set_upload_ml_fallback(df)


def process_uploaded_json_file(uploaded_file: Any) -> None:
    if uploaded_file is None:
        return
    file_bytes = uploaded_file.getvalue()
    file_signature = f"{uploaded_file.name}:{len(file_bytes)}:{hash(file_bytes)}"
    if file_signature == st.session_state.get("last_uploaded_file_signature"):
        st.info("הקובץ כבר נטען בסשן הנוכחי.")
        return

    raw_uploaded_df, resource_id, total, records_count, error_message = parse_ckan_datastore_json(uploaded_file)
    if error_message or raw_uploaded_df is None:
        st.error(error_message or "הקובץ לא נראה כמו JSON תקין של data.gov.il CKAN datastore_search.")
        return

    uploaded_standardized_df = standardize_uploaded_urban_renewal_records(raw_uploaded_df, resource_id)
    uploaded_scored_df = score_uploaded_records(uploaded_standardized_df)
    uploaded_scored_df = apply_saved_ml_similarity_model(uploaded_scored_df)

    current_df = ensure_columns(st.session_state["dashboard_df"])
    existing_ids = set(current_df["record_id"].dropna().astype(str))
    uploaded_ids = set(uploaded_scored_df["record_id"].dropna().astype(str))
    duplicates_replaced = len(existing_ids & uploaded_ids)

    merged_df = pd.concat([current_df, uploaded_scored_df], ignore_index=True)
    merged_df["record_id"] = merged_df["record_id"].astype("string")
    merged_df = merged_df.drop_duplicates(subset=["record_id"], keep="last")
    st.session_state["dashboard_df"] = ensure_columns(merged_df)
    st.session_state["uploaded_session_active"] = True
    st.session_state["last_uploaded_file_signature"] = file_signature

    st.success(f"נוספו {len(uploaded_scored_df):,} רשומות למערכת בסשן הנוכחי.")
    if duplicates_replaced:
        st.info(f"הוחלפו {duplicates_replaced:,} רשומות קיימות לפי record_id.")
    st.warning("העדכון זמני בלבד. בכניסה הבאה יהיה צורך להעלות שוב את הקובץ.")
    st.caption(f"resource_id: {resource_id or 'לא זמין'} · total לפי API: {total if total is not None else 'לא זמין'} · records בקובץ: {records_count:,}")


def options_from_column(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    values = df[col].dropna().astype(str).map(str.strip)
    values = values[~values.str.lower().isin(["", "nan", "none", "null", "<na>"])]
    return sorted(values.unique().tolist())


def apply_multiselect_filter(df: pd.DataFrame, col: str, selected: list[str]) -> pd.DataFrame:
    if not selected or col not in df.columns:
        return df
    return df[df[col].astype(str).isin(selected)]


def display_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy().rename(columns={col: HEBREW_LABELS.get(col, col) for col in existing})


def filtered_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def metric_value(value: Any, decimals: int = 1) -> str:
    if pd.isna(value):
        return "לא זמין"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{decimals}f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


PLAN_REVIEW_CHECKLIST = [
    "נפתח קישור מבא״ת",
    "אומת מספר תכנית",
    "נבדק סטטוס אחרון במבא״ת",
    "נבדק האם התכנית הופקדה",
    "נבדק האם התכנית אושרה",
    "נבדק האם האישור כולל תנאים",
    "נבדקו התנגדויות / החלטות ועדה",
    "נבדקו תקנון / תשריט / נספחים",
    "נבדקו היתרים / שלב רישוי",
]


def row_text(row: pd.Series, col: str, fallback: str = "לא זמין") -> str:
    return safe_text(row.get(col), fallback)


def record_identifier(row: pd.Series) -> str:
    return row_text(row, "record_id", "")


def build_case_summary(row: pd.Series) -> str:
    return "\n".join([
        "תקציר בדיקה ראשונית:",
        "",
        f"מתחם: {row_text(row, 'complex_name')}",
        f"עיר: {row_text(row, 'city')}",
        f"מספר מתחם: {row_text(row, 'source_record_id')}",
        f"מספר תכנית: {row_text(row, 'plan_number')}",
        f"סוג התחדשות: {row_text(row, 'renewal_type')}",
        f"סטטוס תכנוני ציבורי: {row_text(row, 'planning_status_raw')}",
        f"סטטוס מנורמל: {row_text(row, 'planning_status_normalized')}",
        f"Project Momentum Score: {metric_value(row.get('project_momentum_score'), 1)}",
        f"רמת בשלות אינדיקטיבית: {row_text(row, 'project_momentum_label')}",
        f"קישור מבא״ת: {safe_url(row.get('mavat_url')) or 'לא זמין'}",
        f"קישור מפה: {safe_url(row.get('map_url')) or 'לא זמין'}",
        f"דגל איכות נתונים: {row_text(row, 'data_quality_flag')}",
        "",
        "הערת בדיקה:",
        row_text(row, "lawyer_note", "אין הערת בדיקה."),
        "",
        "הערת מערכת:",
        row_text(row, "dashboard_note", "אין הערת מערכת."),
        "",
        "הבהרה:",
        "המידע מבוסס על מקורות ציבוריים ואינו מחליף בדיקה משפטית, תכנונית, שמאית, עסקית או נדל״נית.",
    ])


def render_source_action(label: str, url: Any) -> None:
    clean_url = safe_url(url)
    st.markdown(f"[{label}]({clean_url})" if clean_url else f"**{label}:** לא זמין")


def render_plan_review_checklist(row: pd.Series) -> None:
    record_id = record_identifier(row) or "unknown_record"
    with st.expander("רשימת בדיקה לתוכנית", expanded=False):
        for idx, item in enumerate(PLAN_REVIEW_CHECKLIST):
            st.checkbox(item, key=f"plan_review_{record_id}_{idx}")
        st.caption(
            "צריך לדייק מאוד מה אפשר להוציא אוטומטית ומה דורש בדיקה במסמכים. "
            "המערכת נותנת אינדיקציה ציבורית ראשונית; אישור בתנאים, התנגדויות והחלטות ועדה "
            "דורשים בדיקה במסמכי מבא״ת, פרוטוקולים או החלטות ועדה."
        )


def render_case_summary(row: pd.Series) -> None:
    record_id = record_identifier(row)
    file_record_id = record_id if record_id else "unknown_record"
    summary = build_case_summary(row)
    with st.expander("תקציר בדיקה להעתקה", expanded=False):
        st.text_area("תקציר", value=summary, height=360, key=f"case_summary_{file_record_id}")
        st.download_button(
            "הורד תקציר TXT",
            data=summary.encode("utf-8-sig"),
            file_name=f"case_summary_{file_record_id}.txt",
            mime="text/plain",
            key=f"download_case_summary_{file_record_id}",
        )


def render_review_basket(current_df: pd.DataFrame) -> None:
    if "review_basket_ids" not in st.session_state:
        st.session_state["review_basket_ids"] = []

    with st.expander("תיק בדיקה זמני", expanded=False):
        st.caption("תיק הבדיקה זמני לסשן הנוכחי בלבד.")
        basket_ids = [str(record_id) for record_id in st.session_state["review_basket_ids"] if has_value(record_id)]
        if basket_ids and "record_id" in current_df.columns:
            basket_df = current_df[current_df["record_id"].astype(str).isin(basket_ids)].copy()
        else:
            basket_df = pd.DataFrame(columns=["city", "complex_name", "plan_number", "planning_status_raw", "project_momentum_score", "mavat_url"])

        basket_columns = ["city", "complex_name", "plan_number", "planning_status_raw", "project_momentum_score", "mavat_url"]
        existing_columns = [col for col in basket_columns if col in basket_df.columns]
        if basket_df.empty:
            st.caption("אין עדיין מתחמים בתיק הבדיקה.")
        else:
            st.dataframe(
                basket_df[existing_columns].rename(columns={col: HEBREW_LABELS.get(col, col) for col in existing_columns}),
                use_container_width=True,
                hide_index=True,
            )

        col_clear, col_download = st.columns(2)
        with col_clear:
            if st.button("נקה תיק בדיקה", key="clear_review_basket"):
                st.session_state["review_basket_ids"] = []
                st.success("תיק הבדיקה הזמני נוקה.")
        with col_download:
            st.download_button(
                "הורד תיק בדיקה CSV",
                data=filtered_csv_bytes(basket_df[existing_columns] if existing_columns else basket_df),
                file_name="temporary_review_basket.csv",
                mime="text/csv",
                disabled=basket_df.empty,
                key="download_review_basket_csv",
            )


def compute_city_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    temp = df.copy()
    for col in ["has_plan_number", "has_mavat_url", "has_quality_issue"]:
        temp[col] = normalize_bool(temp[col]).astype(int)
    group = temp.groupby("city", dropna=False)
    summary = group.agg(
        num_records=("record_id", "size"),
        avg_project_momentum_score=("project_momentum_score", "mean"),
        median_project_momentum_score=("project_momentum_score", "median"),
        avg_ml_advancement_score=("ml_advancement_score", "mean"),
        num_with_plan_number=("has_plan_number", "sum"),
        num_with_mavat_url=("has_mavat_url", "sum"),
        num_with_quality_issues=("has_quality_issue", "sum"),
        total_existing_units=("existing_units", "sum"),
        total_proposed_units=("proposed_units", "sum"),
    ).reset_index()
    summary["num_very_high_momentum"] = group["project_momentum_label"].apply(lambda s: int((s == "VERY_HIGH").sum())).values
    summary["num_high_momentum"] = group["project_momentum_label"].apply(lambda s: int((s == "HIGH").sum())).values
    summary["num_moderate_momentum"] = group["project_momentum_label"].apply(lambda s: int((s == "MODERATE").sum())).values
    summary["num_low_momentum"] = group["project_momentum_label"].apply(lambda s: int((s == "LOW").sum())).values
    summary["num_advanced_public_status"] = group["advanced_project_label"].apply(lambda s: int((s == 1).sum())).values
    summary["num_plan_approved"] = group["planning_status_normalized"].apply(lambda s: int((s == "PLAN_APPROVED").sum())).values
    summary["num_permit_approved"] = group["planning_status_normalized"].apply(lambda s: int((s == "PERMIT_APPROVED").sum())).values
    summary["num_construction"] = group["planning_status_normalized"].apply(lambda s: int((s == "CONSTRUCTION").sum())).values
    for col in SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = pd.NA
    return summary[SUMMARY_COLUMNS]


def render_bar_chart(data: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    st.subheader(title)
    if data.empty:
        st.caption("אין מספיק נתונים להצגה לפי הסינון הנוכחי.")
        return
    if PLOTLY_AVAILABLE:
        fig = px.bar(data, x=x, y=y, color=color, text=y if y in data.columns else None)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10), font=dict(family="Arial"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data.set_index(x)[y])


def render_field(label: str, value: Any) -> None:
    st.markdown(f"**{label}:** {safe_text(value)}")


def render_link(label: str, url: Any) -> None:
    clean_url = safe_url(url)
    st.markdown(f"[{label}]({clean_url})" if clean_url else "לא זמין")


def selected_option_label(row: pd.Series) -> str:
    city = safe_text(row.get("city"), "ללא עיר")
    complex_name = safe_text(row.get("complex_name"), "ללא שם מתחם")
    plan = safe_text(row.get("plan_number"), "ללא מספר תוכנית")
    score = metric_value(row.get("project_momentum_score"), decimals=1)
    return f"{city} | {complex_name} | {plan} | {score}"


def render_record_detail(row: pd.Series, current_df: pd.DataFrame) -> None:
    st.markdown("### פרופיל פרויקט")
    if row.get("project_momentum_label") == "VERY_HIGH":
        st.markdown('<div class="success-card">זוהתה אינדיקציה ציבורית לרמת בשלות גבוהה. נדרש אימות תכנוני/משפטי נוסף.</div>', unsafe_allow_html=True)
    if safe_text(row.get("data_quality_flag"), "OK") != "OK":
        st.markdown('<div class="warning-card">קיים דגל איכות נתונים. אין להסיק מסקנות ללא בדיקה ידנית מול המקורות הרשמיים.</div>', unsafe_allow_html=True)

    main_col, units_col, scores_col = st.columns(3)
    with main_col:
        st.markdown("#### פרטים מרכזיים")
        for label, col in [("עיר", "city"), ("שם מתחם", "complex_name"), ("מספר מתחם מקור", "source_record_id"), ("מספר תוכנית", "plan_number"), ("מסלול", "renewal_type"), ("סטטוס תכנוני ציבורי", "planning_status_raw"), ("סטטוס מנורמל", "planning_status_normalized")]:
            render_field(label, row.get(col))
    with units_col:
        st.markdown("#### יחידות והיתרים")
        render_field("יח״ד קיימות", metric_value(row.get("existing_units"), 0))
        render_field("יח״ד תוספתיות", metric_value(row.get("additional_units"), 0))
        render_field("יח״ד מוצעות", metric_value(row.get("proposed_units"), 0))
        render_field("יחס מוצע/קיים", metric_value(row.get("proposed_to_existing_ratio"), 2))
        render_field("היתרים", metric_value(row.get("permits_total"), 0))
    with scores_col:
        st.markdown("#### ניקוד")
        render_field("Project Momentum Score", metric_value(row.get("project_momentum_score"), 1))
        render_field("רמת בשלות אינדיקטיבית", row.get("project_momentum_label"))
        render_field("Planning Maturity Score", metric_value(row.get("planning_maturity_score"), 1))
        render_field("Data Confidence Score", metric_value(row.get("data_confidence_score"), 1))
        render_field("Source Strength Score", metric_value(row.get("source_strength_score"), 1))
        render_field("Indicative Advancement Score", metric_value(row.get("ml_advancement_score"), 1))
        render_field("רמת התקדמות אינדיקטיבית", row.get("ml_advancement_label"))
        st.caption("ציון אינדיקטיבי בלבד — אינו מהווה תחזית מחייבת.")

    note_col, link_col = st.columns([2, 1])
    with note_col:
        st.markdown("#### הערות")
        st.info(safe_text(row.get("lawyer_note"), "אין הערת בדיקה."))
        st.write(safe_text(row.get("dashboard_note"), "אין הערת מערכת."))
        st.caption(safe_text(row.get("project_momentum_explanation"), "אין הסבר ניקוד."))
        st.warning(safe_text(row.get("ml_score_warning"), ML_DISCLAIMER))
        render_field("דגל איכות נתונים", row.get("data_quality_flag"))
    with link_col:
        st.markdown("#### קישורים")
        render_link("פתח במבא״ת", row.get("mavat_url"))
        render_link("פתח מפה", row.get("map_url"))
        render_link("מקור מידע", row.get("source_url"))
        st.markdown("#### פתיחת מקורות")
        render_source_action("פתח מבא״ת", row.get("mavat_url"))
        render_source_action("פתח מפה", row.get("map_url"))
        render_source_action("פתח מקור מידע", row.get("source_url"))

    record_id = record_identifier(row)
    if st.button("הוסף לתיק בדיקה זמני", key=f"add_to_review_basket_{record_id or 'unknown_record'}"):
        if record_id:
            if "review_basket_ids" not in st.session_state:
                st.session_state["review_basket_ids"] = []
            if record_id not in st.session_state["review_basket_ids"]:
                st.session_state["review_basket_ids"].append(record_id)
            st.success("המתחם נוסף לתיק הבדיקה הזמני.")
        else:
            st.warning("לא נמצא record_id למתחם הזה, ולכן אי אפשר להוסיף אותו לתיק הבדיקה הזמני.")

    render_review_basket(current_df)
    render_plan_review_checklist(row)
    render_case_summary(row)

try:
    base_df = load_scored_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"שגיאה בטעינת הנתונים: {type(exc).__name__}: {exc}")
    st.stop()

if "dashboard_df" not in st.session_state:
    st.session_state["dashboard_df"] = base_df.copy()
if "uploaded_session_active" not in st.session_state:
    st.session_state["uploaded_session_active"] = False
if "last_uploaded_file_signature" not in st.session_state:
    st.session_state["last_uploaded_file_signature"] = None
if "review_basket_ids" not in st.session_state:
    st.session_state["review_basket_ids"] = []

st.markdown('<div class="main-title">Urban Renewal Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-line">ברוכים הבאים, צוות צמח המרמן</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Public-data exploration tool for Israeli urban renewal projects.</div>', unsafe_allow_html=True)
st.markdown('<div class="context-caption">מערכת חקירה מבוססת מידע ציבורי לבחינה ראשונית של מתחמי התחדשות עירונית.</div>', unsafe_allow_html=True)
st.markdown('<div class="creator-line">פותח על ידי גיא לוזון | אבטיפוס מודיעין נדל״ני מבוסס מידע ציבורי</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="professional-card">
    <h3>כיצד המערכת יכולה לסייע לאנשי נדל״ן?</h3>
    <p>
        המערכת מרכזת מידע ציבורי על מתחמי התחדשות עירונית ומאפשרת לבצע סינון ראשוני של אזורים,
        מתחמים ותוכניות לפי סטטוס תכנוני, מספר תוכנית, סוג התחדשות, מקור מידע וציון בשלות אינדיקטיבי.
    </p>
    <p>
        עבור אנשי נדל״ן, המערכת יכולה לסייע בזיהוי ראשוני של אזורים עם פעילות תכנונית,
        איתור מתחמים בעלי אינדיקציות להתקדמות, השוואה בין פרויקטים לפי רמת בשלות ציבורית,
        והכוונת בדיקות המשך מול מקורות רשמיים כגון מסמכי תכנון, מערכות מינהל התכנון,
        רשויות מקומיות וגורמי מקצוע.
    </p>
    <ul>
        <li>סינון ראשוני של מתחמים ואזורים</li>
        <li>איתור תוכניות בעלות אינדיקציה להתקדמות</li>
        <li>השוואה בין רשומות לפי סטטוס וציון בשלות</li>
        <li>קישור מהיר למקורות מידע ציבוריים</li>
        <li>הכנת רשימת פרויקטים לבדיקה מקצועית מעמיקה</li>
    </ul>
    <p>
        המערכת אינה מחליפה בדיקת היתכנות, בדיקה משפטית, בדיקה שמאית או בדיקה תכנונית,
        אלא נועדה לשמש כשכבת סינון ומודיעין ראשונית המבוססת על מידע ציבורי.
    </p>
</div>
    """,
    unsafe_allow_html=True,
)
st.warning(FULL_DISCLAIMER)
st.markdown(f'<div class="rights-card">{USAGE_RIGHTS_TOP_HTML}</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-callout">להעלאת JSON חדש ממקור ציבורי: פתח את הלשונית הראשונה "הוספת נתונים" או השתמש בכפתור ההעלאה שבסרגל הצד.</div>', unsafe_allow_html=True)

tab_upload, tab_search, tab_overview, tab_city, tab_scoring, tab_about = st.tabs(["הוספת נתונים", "חיפוש מתחמים", "תמונת מצב", "סיכום לפי עיר", "הסבר ניקוד ו־ML", "אודות והבהרות"])

with tab_upload:
    st.markdown("### הוספת נתונים זמנית לסשן")
    st.write("""
כאן אפשר להעלות קובץ JSON רשמי מ־data.gov.il כדי לעדכן את המערכת בסשן הנוכחי.

חשוב: ההעלאה אינה נשמרת קבוע. אם מרעננים את הדף, סוגרים את הדפדפן או שהאפליקציה מתאפסת — יהיה צורך להעלות את הקובץ שוב.

העלאה זו מיועדת לבדיקת מידע ציבורי עדכני בסשן הנוכחי בלבד, ואינה מחליפה תהליך קליטת נתונים קבוע.
""")
    st.caption("הנתונים הבסיסיים של המערכת נטענים אוטומטית. ההעלאה כאן מיועדת למקרה שבו יש קובץ חדש או מעודכן לבדיקה בסשן הנוכחי.")
    with st.expander("איך מכינים קובץ להעלאה?", expanded=False):
        st.markdown("""
1. נכנסים לקישור API רשמי של data.gov.il בפורמט:
   https://data.gov.il/api/3/action/datastore_search?resource_id=<RESOURCE_ID>&limit=50000

2. דוגמה למשאב שעבד בפרויקט:
   https://data.gov.il/api/3/action/datastore_search?resource_id=f65a0daf-f737-49c5-9424-d378d52104f5&limit=50000

3. מוודאים שבדף מופיע: "success": true

4. שומרים את הדף כקובץ JSON:
   - אפשרות א: Ctrl + S ושמירה כקובץ עם סיומת .json
   - אפשרות ב: Ctrl + A, Ctrl + C, לפתוח קובץ חדש בשם data_update.json, להדביק ולשמור

5. חוזרים למערכת ומעלים את הקובץ בטופס למטה.

6. לאחר ההעלאה, המערכת תחבר את הרשומות לנתונים הקיימים ותעדכן את הטבלאות, המסננים והגרפים.

7. אם רוצים לשמור את התוצאה, לוחצים: הורד CSV מעודכן
""")
        st.warning("יש להעלות רק קבצי JSON רשמיים ממקור ציבורי. לא להעלות קבצים פרטיים, חסויים או קבצים שלא ברור מקורם.")

    uploaded_file = st.file_uploader("העלה קובץ JSON רשמי מ־data.gov.il", type=["json"])
    st.info("אם קיים מודל ML שמור בסביבת הפרויקט, המערכת תחשב גם Indicative Advancement Score לרשומות שהועלו. המדד אינדיקטיבי בלבד ואינו תחזית.")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_signature = f"{uploaded_file.name}:{len(file_bytes)}:{hash(file_bytes)}"
        if file_signature == st.session_state.get("last_uploaded_file_signature"):
            st.info("הקובץ כבר נטען בסשן הנוכחי.")
        else:
            raw_uploaded_df, resource_id, total, records_count, error_message = parse_ckan_datastore_json(uploaded_file)
            if error_message or raw_uploaded_df is None:
                st.error(error_message or "הקובץ לא נראה כמו JSON תקין של data.gov.il CKAN datastore_search.")
            else:
                uploaded_standardized_df = standardize_uploaded_urban_renewal_records(raw_uploaded_df, resource_id)
                uploaded_scored_df = score_uploaded_records(uploaded_standardized_df)
                uploaded_scored_df = apply_saved_ml_similarity_model(uploaded_scored_df)
                current_df = ensure_columns(st.session_state["dashboard_df"])
                existing_ids = set(current_df["record_id"].dropna().astype(str))
                uploaded_ids = set(uploaded_scored_df["record_id"].dropna().astype(str))
                duplicates_replaced = len(existing_ids & uploaded_ids)
                merged_df = pd.concat([current_df, uploaded_scored_df], ignore_index=True)
                merged_df["record_id"] = merged_df["record_id"].astype("string")
                merged_df = merged_df.drop_duplicates(subset=["record_id"], keep="last")
                st.session_state["dashboard_df"] = ensure_columns(merged_df)
                st.session_state["uploaded_session_active"] = True
                st.session_state["last_uploaded_file_signature"] = file_signature
                st.success(f"נוספו {len(uploaded_scored_df):,} רשומות למערכת בסשן הנוכחי.")
                if duplicates_replaced:
                    st.info(f"הוחלפו {duplicates_replaced:,} רשומות קיימות לפי record_id.")
                st.warning("העדכון זמני בלבד. בכניסה הבאה יהיה צורך להעלות שוב את הקובץ.")
                st.caption(f"resource_id: {resource_id or 'לא זמין'} · total לפי API: {total if total is not None else 'לא זמין'} · records בקובץ: {records_count:,}")

    if st.button("איפוס לנתונים המקוריים"):
        st.session_state["dashboard_df"] = base_df.copy()
        st.session_state["uploaded_session_active"] = False
        st.session_state["last_uploaded_file_signature"] = None
        st.success("המערכת אופסה לנתונים המקוריים.")

    st.download_button(
        "הורד CSV מעודכן",
        data=filtered_csv_bytes(ensure_columns(st.session_state["dashboard_df"])),
        file_name="urban_renewal_scored_session_updated.csv",
        mime="text/csv",
    )

# From here onward, every dashboard component uses the current session dataframe.
df = ensure_columns(st.session_state["dashboard_df"])

with st.sidebar.expander("הוספת נתונים", expanded=False):
    st.caption("העלה JSON רשמי מ־data.gov.il לעדכון זמני של הסשן.")
    sidebar_uploaded_file = st.file_uploader(
        "העלה JSON",
        type=["json"],
        key="sidebar_json_upload",
    )
    process_uploaded_json_file(sidebar_uploaded_file)

df = ensure_columns(st.session_state["dashboard_df"])

st.sidebar.markdown("## סינון מתחמים")
st.sidebar.caption("סינון ראשוני לפי מאפייני הפרויקט, סטטוס תכנוני, רמת אמינות נתונים ומקורות מידע.")
city_options = options_from_column(df, "city")
status_col = "planning_status_normalized" if "planning_status_normalized" in df.columns else "planning_status_raw"
status_options = options_from_column(df, status_col)
renewal_options = options_from_column(df, "renewal_type")
momentum_options = options_from_column(df, "project_momentum_label")
confidence_options = options_from_column(df, "confidence_level")

selected_cities = st.sidebar.multiselect("עיר", city_options, default=city_options)
selected_statuses = st.sidebar.multiselect("סטטוס תכנוני", status_options, default=status_options)
selected_renewal = st.sidebar.multiselect("מסלול / סוג התחדשות", renewal_options, default=renewal_options)
selected_momentum = st.sidebar.multiselect("רמת בשלות אינדיקטיבית", momentum_options, default=momentum_options)
selected_confidence = st.sidebar.multiselect("רמת אמינות נתונים", confidence_options, default=confidence_options)
only_quality_issues = st.sidebar.checkbox("הצג רק רשומות עם דגלי איכות נתונים")
only_mavat = st.sidebar.checkbox("הצג רק רשומות עם קישור למבא״ת")
only_plan_number = st.sidebar.checkbox("הצג רק רשומות עם מספר תוכנית")
score_range = st.sidebar.slider("טווח Project Momentum Score", 0.0, 100.0, (0.0, 100.0), 1.0)
search_text = st.sidebar.text_input("חיפוש חופשי", placeholder="עיר, מתחם, תוכנית, סטטוס...")

filtered_df = df.copy()
filtered_df = apply_multiselect_filter(filtered_df, "city", selected_cities)
filtered_df = apply_multiselect_filter(filtered_df, status_col, selected_statuses)
filtered_df = apply_multiselect_filter(filtered_df, "renewal_type", selected_renewal)
filtered_df = apply_multiselect_filter(filtered_df, "project_momentum_label", selected_momentum)
filtered_df = apply_multiselect_filter(filtered_df, "confidence_level", selected_confidence)
filtered_df = filtered_df[filtered_df["project_momentum_score"].fillna(-1).between(score_range[0], score_range[1])]
if only_quality_issues:
    filtered_df = filtered_df[filtered_df["has_quality_issue"]]
if only_mavat:
    filtered_df = filtered_df[filtered_df["has_mavat_url"]]
if only_plan_number:
    filtered_df = filtered_df[filtered_df["has_plan_number"]]
if search_text.strip():
    needle = search_text.strip().lower()
    search_cols = ["city", "complex_name", "plan_number", "source_record_id", "planning_status_raw", "renewal_type", "street_or_area"]
    existing_search_cols = [col for col in search_cols if col in filtered_df.columns]
    if existing_search_cols:
        haystack = filtered_df[existing_search_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered_df = filtered_df[haystack.str.contains(needle, regex=False, na=False)]

if filtered_df.empty:
    st.warning("לא נמצאו מתחמים לפי הסינון הנוכחי. ניתן להרחיב את תנאי החיפוש או לנקות חלק מהמסננים.")

known_status_count = int(filtered_df["planning_status_raw"].map(has_value).sum()) if "planning_status_raw" in filtered_df.columns else 0
official_source_count = int(
    (
        filtered_df["has_mavat_url"].fillna(False).astype(bool)
        | filtered_df["source_url"].map(lambda value: bool(safe_url(value)))
    ).sum()
) if "source_url" in filtered_df.columns else int(filtered_df["has_mavat_url"].fillna(False).astype(bool).sum())

metric_cols = st.columns(5)
metric_cols[0].metric("סה״כ רשומות", f"{len(filtered_df):,}")
metric_cols[1].metric("מספר ערים", f"{filtered_df['city'].nunique(dropna=True):,}")
metric_cols[2].metric("עם מספר תוכנית", f"{int(filtered_df['has_plan_number'].sum()):,}")
metric_cols[3].metric("עם סטטוס תכנוני ידוע", f"{known_status_count:,}")
metric_cols[4].metric("עם מקור מידע רשמי", f"{official_source_count:,}")

st.download_button("הורד CSV מסונן", data=filtered_csv_bytes(filtered_df), file_name="urban_renewal_filtered.csv", mime="text/csv")
st.download_button("הורד CSV מלא / סשן נוכחי", data=filtered_csv_bytes(df), file_name="urban_renewal_scored_current_session.csv", mime="text/csv")


with tab_search:
    st.markdown("### טבלת פרויקטים")
    table_df = display_table(filtered_df, TABLE_COLUMNS)
    column_config = {
        "Project Momentum Score": st.column_config.ProgressColumn("Project Momentum Score", min_value=0, max_value=100, format="%.1f"),
        "Indicative Advancement Score": st.column_config.NumberColumn("Indicative Advancement Score", min_value=0, max_value=100, format="%.1f"),
    }
    st.dataframe(table_df, use_container_width=True, hide_index=True, column_config=column_config)
    st.markdown("### בחירת פרויקט לבדיקה")
    if not filtered_df.empty:
        selected_idx = st.selectbox("בחר מתחם", filtered_df.index.tolist(), format_func=lambda idx: selected_option_label(filtered_df.loc[idx]))
        render_record_detail(filtered_df.loc[selected_idx], df)
    else:
        st.caption("אין רשומה זמינה לבחירה לפי הסינון הנוכחי.")

with tab_overview:
    st.markdown("### תמונת מצב כללית")
    col_a, col_b = st.columns(2)
    with col_a:
        city_counts = filtered_df["city"].fillna("לא ידוע").value_counts().head(15).reset_index()
        city_counts.columns = ["עיר", "מספר מתחמים"]
        render_bar_chart(city_counts, "עיר", "מספר מתחמים", "מתחמים לפי עיר — 15 המובילות")
    with col_b:
        status_counts = filtered_df["planning_status_normalized"].fillna("UNKNOWN").value_counts().reset_index()
        status_counts.columns = ["סטטוס", "מספר מתחמים"]
        render_bar_chart(status_counts, "סטטוס", "מספר מתחמים", "מתחמים לפי סטטוס מנורמל")
    col_c, col_d = st.columns(2)
    with col_c:
        city_avg = filtered_df.groupby("city", dropna=False).agg(num_records=("record_id", "size"), avg_score=("project_momentum_score", "mean")).reset_index()
        city_avg = city_avg[city_avg["num_records"] >= 3].sort_values("avg_score", ascending=False).head(15).rename(columns={"city": "עיר", "avg_score": "ציון ממוצע"})
        render_bar_chart(city_avg, "עיר", "ציון ממוצע", "ממוצע Project Momentum Score לפי עיר")
    with col_d:
        momentum_counts = filtered_df["project_momentum_label"].fillna("UNKNOWN").value_counts().reset_index()
        momentum_counts.columns = ["רמת בשלות אינדיקטיבית", "מספר מתחמים"]
        render_bar_chart(momentum_counts, "רמת בשלות אינדיקטיבית", "מספר מתחמים", "התפלגות רמת בשלות אינדיקטיבית")
    ml_counts = filtered_df["ml_advancement_label"].fillna("ML_NOT_AVAILABLE").value_counts().reset_index()
    ml_counts.columns = ["רמת התקדמות אינדיקטיבית", "מספר מתחמים"]
    render_bar_chart(ml_counts, "רמת התקדמות אינדיקטיבית", "מספר מתחמים", "התפלגות Indicative Advancement Score")

with tab_city:
    if st.session_state.get("uploaded_session_active"):
        summary_df = compute_city_summary(df)
        st.caption("הסיכום מחושב מהנתונים המעודכנים של הסשן הנוכחי.")
    else:
        summary_df = load_city_summary()
        if summary_df is None:
            summary_df = compute_city_summary(df)
    for col in ["num_with_plan_number", "num_with_mavat_url", "num_with_quality_issues"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0).astype(int)
    st.markdown("### סיכום לפי עיר")
    city_select_options = options_from_column(summary_df, "city")
    selected_city = st.selectbox("בחר עיר לסיכום מהיר", city_select_options if city_select_options else ["לא זמין"])
    if selected_city != "לא זמין" and "city" in summary_df.columns:
        city_row = summary_df[summary_df["city"].astype(str) == selected_city].head(1)
        if not city_row.empty:
            r = city_row.iloc[0]
            st.markdown('<div class="soft-card">תמונת עיר לפי הרשומות המסוננות. מומלץ לאמת את הנתונים מול המקורות הרשמיים לפני קבלת החלטות.</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("מספר מתחמים", metric_value(r.get("num_records"), 0))
            c2.metric("ממוצע Project Momentum Score", metric_value(r.get("avg_project_momentum_score"), 1))
            c3.metric("עם מבא״ת", metric_value(r.get("num_with_mavat_url"), 0))
            c4.metric("דגלי איכות נתונים", metric_value(r.get("num_with_quality_issues"), 0))
    summary_display_cols = [col for col in SUMMARY_COLUMNS if col in summary_df.columns]
    st.dataframe(summary_df[summary_display_cols].rename(columns=SUMMARY_HEBREW_LABELS), use_container_width=True, hide_index=True)

with tab_scoring:
    st.markdown("### Project Momentum Score")
    st.write("מדד אינדיקטיבי שמחבר בין סטטוס תכנוני ציבורי, חוזק מקור וקישורים, רמת אמינות נתונים, היקף הפרויקט ודגלי איכות נתונים. המדד המרכזי הוא Rule-Based ושקוף, אך אינו תחזית מחייבת.")
    st.markdown("- **סטטוס תכנוני ציבורי** — הרכיב המרכזי בציון.\n- **חוזק מקור** — מספר תוכנית, קישור למבא״ת, מפה ומזהה מקור.\n- **רמת אמינות נתונים** — איכות ושלמות שדות מרכזיים.\n- **היקף פרויקט** — אינדיקציה זהירה לפי יח״ד, לא ציון כלכלי.\n- **דגלי איכות נתונים** — מפחיתים מהציון ולא מוחקים רשומות.")
    st.caption("ציון אינדיקטיבי בלבד — אינו מהווה תחזית מחייבת.")
    with st.expander("מילון סטטוסים קצר", expanded=False):
        st.write("תכנון ראשוני — אינדיקציה מוקדמת, נדרש בירור במסמכים.")
        st.write("תכנון סטטוטורי — הליך תכנוני רשמי, לא בהכרח אישור.")
        st.write("תכנית מאושרת לפני מימוש — התכנית אושרה אך טרם מימוש/רישוי מלא.")
        st.write("תכנית מאושרת - אחרי רישוי — אינדיקציה לשלב מתקדם יותר; יש לבדוק היתרים ומסמכים.")
        st.write("תכנית מאושרת במימוש — אינדיקציה לביצוע/מימוש; עדיין מומלץ לבדוק מקור רשמי.")
        st.caption("סטטוס ציבורי רחב אינו מחליף בדיקת החלטות ועדה, תנאים, התנגדויות או מסמכי תכנית.")
    st.markdown("### Indicative Advancement Score")
    st.write("מדד דמיון אינדיקטיבי למתחמים שמופיעים במידע הציבורי כמתקדמים יותר. המדד אינו תחזית, אינו הסתברות לאישור ואינו תחליף לבדיקה מקצועית.")
    st.info("כאשר המדד זמין, יש להתייחס אליו ככלי עזר למיון ותיעדוף בלבד.")
    st.warning(ML_DISCLAIMER)
    st.caption("רשומות שהועלו ידנית בסשן מקבלות Project Momentum Score, ואם קיים מודל שמור בסביבת הפרויקט הן מקבלות גם Indicative Advancement Score.")

with tab_about:
    st.markdown("### אודות הכלי")
    st.write("המערכת מאפשרת חיפוש, סינון וסקירה ראשונית של מתחמי התחדשות עירונית בישראל על בסיס מידע ציבורי מעובד.")
    st.write("המטרה היא לתמוך באיתור פרויקטים לבדיקה, השוואת סטטוסים ציבוריים, פתיחת מקורות מידע ותיעדוף בדיקות המשך.")
    st.markdown("### הבהרת שימוש")
    st.markdown(USAGE_RIGHTS_FULL_MARKDOWN)
    st.markdown("### הבהרה מקצועית")
    st.warning(FULL_DISCLAIMER)
    st.warning(ML_DISCLAIMER)
    st.caption("המערכת אינה מחליפה בדיקה מקצועית במסמכי התכנון, במבא״ת, במערכות העירוניות ובמקורות הרשמיים.")

st.markdown(f'<div class="rights-footer">{USAGE_RIGHTS_FOOTER_HTML}</div>', unsafe_allow_html=True)
