"""
CSV/Excel Data Quality & Profiling Tool
========================================
INPUT:  A CSV or Excel file (e.g., customers.xlsx, sales.csv)
OUTPUT: A comprehensive data quality report in JSON format, including:
        - Quality score (0-100)
        - Missing value analysis
        - Duplicate detection
        - Email/date/numeric validation
        - Column type inference
        - Distribution statistics
        - Outlier detection
        - Actionable recommendations
        - Optionally: a cleaned dataset (CSV)

PURPOSE: Accepts tabular data files and produces enterprise-grade data quality
         reports to help data teams identify issues, validate data integrity,
         and clean datasets before analysis or migration.
"""

import os
import re
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)

# Columns whose names suggest they are ID columns
ID_COLUMN_PATTERNS = re.compile(
    r"^(id|.*_id|.*id$|key|.*_key|uuid|guid)", re.IGNORECASE
)

# Columns whose names suggest they contain email addresses
EMAIL_COLUMN_PATTERNS = re.compile(
    r"(email|e_mail|e-mail|mail|email_address)", re.IGNORECASE
)

# Columns whose names suggest they contain dates
DATE_COLUMN_PATTERNS = re.compile(
    r"(date|datetime|timestamp|created|updated|modified|_at$|_on$|_dt$|dob|birth)",
    re.IGNORECASE,
)

# Columns whose names suggest they contain phone numbers
PHONE_COLUMN_PATTERNS = re.compile(
    r"(phone|mobile|cell|tel|fax|contact_number)", re.IGNORECASE
)

# Maximum rows to sample for heavy operations on very large files
MAX_SAMPLE_ROWS = 500_000


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame.

    Supports .csv, .xlsx, and .xls extensions. Raises ValueError for
    unsupported formats or missing files.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, engine="openpyxl" if ext == ".xlsx" else None)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv, .xlsx, or .xls")

    if df.empty:
        logger.warning("Loaded file is empty (0 rows).")

    return df


def _is_email_column(series: pd.Series, col_name: str) -> bool:
    """Heuristic: decide if a column likely contains email addresses."""
    # Check column name first
    if EMAIL_COLUMN_PATTERNS.search(col_name):
        return True
    # If the column is object/string, sample values and check pattern
    if series.dtype == object:
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        match_ratio = sample.apply(lambda v: bool(EMAIL_REGEX.match(str(v)))).mean()
        return match_ratio > 0.5
    return False


def _is_date_column(series: pd.Series, col_name: str) -> bool:
    """Heuristic: decide if a column likely contains date/datetime values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if DATE_COLUMN_PATTERNS.search(col_name):
        return True
    return False


def _is_phone_column(col_name: str) -> bool:
    """Heuristic: decide if a column likely contains phone numbers."""
    return bool(PHONE_COLUMN_PATTERNS.search(col_name))


def _is_id_column(col_name: str) -> bool:
    """Heuristic: decide if a column likely represents an ID/key."""
    return bool(ID_COLUMN_PATTERNS.search(col_name))


def _infer_column_type(series: pd.Series, col_name: str) -> str:
    """Infer a human-readable type label for a column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_float_dtype(series):
            return "float"
        return "integer"
    # String-based heuristics
    if _is_email_column(series, col_name):
        return "email"
    if _is_date_column(series, col_name):
        return "datetime (string)"
    if _is_phone_column(col_name):
        return "phone"
    # Cardinality-based: categorical vs free-text
    nunique = series.nunique()
    total = len(series.dropna())
    if total > 0 and nunique / total < 0.05:
        return "categorical"
    return "text"


def _validate_emails(series: pd.Series) -> dict[str, Any]:
    """Validate email values in a series. Returns count of invalid emails and samples."""
    non_null = series.dropna().astype(str)
    invalid_mask = ~non_null.apply(lambda v: bool(EMAIL_REGEX.match(v)))
    invalid_count = int(invalid_mask.sum())
    invalid_samples = non_null[invalid_mask].head(10).tolist()
    return {"count": invalid_count, "samples": invalid_samples}


def _validate_dates(series: pd.Series) -> dict[str, Any]:
    """Attempt to parse a string column as dates and report failures."""
    if pd.api.types.is_datetime64_any_dtype(series):
        # Already parsed — check for NaT
        nat_count = int(series.isna().sum())
        return {"count": nat_count, "samples": []}

    non_null = series.dropna().astype(str)
    invalid_indices: list[str] = []
    for val in non_null.head(MAX_SAMPLE_ROWS):
        try:
            pd.to_datetime(val)
        except (ValueError, TypeError):
            invalid_indices.append(val)
            if len(invalid_indices) >= 10:
                break

    # Full count via vectorised attempt
    parsed = pd.to_datetime(non_null, errors="coerce")
    invalid_count = int(parsed.isna().sum())
    return {"count": invalid_count, "samples": invalid_indices[:10]}


def _detect_invalid_numerics(series: pd.Series) -> dict[str, Any]:
    """For columns that *should* be numeric but contain non-numeric strings."""
    if pd.api.types.is_numeric_dtype(series):
        return {"count": 0, "samples": []}

    non_null = series.dropna().astype(str)
    coerced = pd.to_numeric(non_null, errors="coerce")
    invalid_mask = coerced.isna() & non_null.notna()
    invalid_count = int(invalid_mask.sum())
    invalid_samples = non_null[invalid_mask].head(10).tolist()
    return {"count": invalid_count, "samples": invalid_samples}


def _compute_distribution(series: pd.Series) -> dict[str, Any]:
    """Compute descriptive statistics for a numeric series."""
    clean = series.dropna()
    if len(clean) == 0:
        return {}
    return {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "q1": round(float(clean.quantile(0.25)), 4),
        "q3": round(float(clean.quantile(0.75)), 4),
    }


def _detect_outliers_iqr(series: pd.Series) -> dict[str, Any]:
    """Detect outliers using the IQR (Inter-Quartile Range) method."""
    clean = series.dropna()
    if len(clean) < 4:
        return {"count": 0, "lower_bound": None, "upper_bound": None, "samples": []}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (clean < lower) | (clean > upper)
    outlier_count = int(outlier_mask.sum())
    outlier_samples = clean[outlier_mask].head(10).tolist()

    return {
        "count": outlier_count,
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "samples": [round(float(v), 4) for v in outlier_samples],
    }


def _detect_suspicious_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag columns that are constant, mostly null, or have very high cardinality."""
    suspicious: list[dict[str, Any]] = []
    total_rows = len(df)
    if total_rows == 0:
        return suspicious

    for col in df.columns:
        reasons: list[str] = []
        null_pct = df[col].isna().mean() * 100

        # Mostly null (>80 %)
        if null_pct > 80:
            reasons.append(f"{null_pct:.1f}% missing values")

        # Constant value
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1 and null_pct < 100:
            reasons.append("constant or single-value column")

        # Very high cardinality (>95 % unique among non-null)
        non_null_count = df[col].notna().sum()
        if non_null_count > 0 and nunique / non_null_count > 0.95 and nunique > 50:
            reasons.append(f"very high cardinality ({nunique} unique values)")

        if reasons:
            suspicious.append({"column": col, "reasons": reasons})

    return suspicious


def _calculate_quality_score(
    total_rows: int,
    total_cells: int,
    missing_total: int,
    duplicate_rows: int,
    duplicate_id_count: int,
    invalid_email_count: int,
    invalid_date_count: int,
    invalid_numeric_count: int,
    outlier_total: int,
    suspicious_col_count: int,
    total_columns: int,
) -> float:
    """Calculate an overall quality score from 0 to 100.

    The score starts at 100 and is reduced by weighted penalties for each
    category of issue found.
    """
    if total_rows == 0 or total_cells == 0:
        return 0.0

    score = 100.0

    # Missing values penalty (max -25 points)
    missing_ratio = missing_total / total_cells
    score -= min(25.0, missing_ratio * 100)

    # Duplicate rows penalty (max -15 points)
    dup_ratio = duplicate_rows / total_rows
    score -= min(15.0, dup_ratio * 100)

    # Duplicate IDs penalty (max -10 points)
    if total_rows > 0:
        dup_id_ratio = duplicate_id_count / total_rows
        score -= min(10.0, dup_id_ratio * 100)

    # Invalid emails penalty (max -10 points)
    if total_cells > 0:
        email_ratio = invalid_email_count / total_cells
        score -= min(10.0, email_ratio * 200)

    # Invalid dates penalty (max -10 points)
    if total_cells > 0:
        date_ratio = invalid_date_count / total_cells
        score -= min(10.0, date_ratio * 200)

    # Invalid numerics penalty (max -10 points)
    if total_cells > 0:
        num_ratio = invalid_numeric_count / total_cells
        score -= min(10.0, num_ratio * 200)

    # Outliers penalty (max -10 points)
    if total_cells > 0:
        outlier_ratio = outlier_total / total_cells
        score -= min(10.0, outlier_ratio * 100)

    # Suspicious columns penalty (max -10 points)
    if total_columns > 0:
        sus_ratio = suspicious_col_count / total_columns
        score -= min(10.0, sus_ratio * 30)

    return round(max(0.0, score), 2)


def _generate_recommendations(
    missing_by_column: dict[str, dict[str, Any]],
    duplicate_rows: int,
    duplicate_ids: dict[str, Any],
    invalid_emails: dict[str, dict[str, Any]],
    invalid_dates: dict[str, dict[str, Any]],
    invalid_numerics: dict[str, dict[str, Any]],
    outliers: dict[str, dict[str, Any]],
    suspicious_columns: list[dict[str, Any]],
) -> list[str]:
    """Generate human-readable, actionable recommendations."""
    recs: list[str] = []

    # Missing values
    for col, info in missing_by_column.items():
        if info["count"] > 0:
            recs.append(
                f"Column '{col}' has {info['count']} missing values "
                f"({info['percentage']}%). Consider imputation or removal."
            )

    # Duplicate rows
    if duplicate_rows > 0:
        recs.append(
            f"Dataset contains {duplicate_rows} exact duplicate rows. "
            "Consider deduplication."
        )

    # Duplicate IDs
    if duplicate_ids.get("count", 0) > 0:
        recs.append(
            f"Column '{duplicate_ids['column']}' has {duplicate_ids['count']} "
            "duplicate values. ID columns should be unique."
        )

    # Invalid emails
    for col, info in invalid_emails.items():
        if info["count"] > 0:
            recs.append(
                f"Column '{col}' contains {info['count']} invalid email addresses. "
                "Validate and correct email formats."
            )

    # Invalid dates
    for col, info in invalid_dates.items():
        if info["count"] > 0:
            recs.append(
                f"Column '{col}' contains {info['count']} invalid or unparseable dates. "
                "Standardise date formats."
            )

    # Invalid numerics
    for col, info in invalid_numerics.items():
        if info["count"] > 0:
            recs.append(
                f"Column '{col}' contains {info['count']} non-numeric values in a "
                "numeric context. Clean or convert these values."
            )

    # Outliers
    for col, info in outliers.items():
        if info["count"] > 0:
            recs.append(
                f"Column '{col}' has {info['count']} outliers "
                f"(values outside IQR range [{info['lower_bound']}, {info['upper_bound']}]). "
                "Review for data entry errors."
            )

    # Suspicious columns
    for entry in suspicious_columns:
        reasons = "; ".join(entry["reasons"])
        recs.append(
            f"Column '{entry['column']}' is suspicious: {reasons}. "
            "Consider dropping or investigating."
        )

    return recs


def _generate_cleaned_file(df: pd.DataFrame, input_file: str) -> str:
    """Produce a cleaned CSV: drop exact duplicates, forward-fill missing values.

    Returns the path to the cleaned file.
    """
    cleaned = df.copy()

    # Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates()

    # For numeric columns, fill missing with median; for others, leave as-is
    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            median_val = cleaned[col].median()
            if pd.notna(median_val):
                cleaned[col] = cleaned[col].fillna(median_val)
        elif cleaned[col].dtype == object:
            # Fill categorical/text with mode if available
            mode_vals = cleaned[col].mode()
            if len(mode_vals) > 0:
                cleaned[col] = cleaned[col].fillna(mode_vals.iloc[0])

    # Determine output path
    base, _ = os.path.splitext(input_file)
    output_path = f"{base}_cleaned.csv"
    cleaned.to_csv(output_path, index=False)
    logger.info("Cleaned file written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# MCP Server & FastAPI application
# ---------------------------------------------------------------------------

mcp = FastMCP("CSV Data Quality Profiler")
app = FastAPI(title="CSV Data Quality & Profiling Tool")


@mcp.tool()
async def profile_dataset(
    input_file: str,
    output_format: str = "json",
    generate_cleaned_file: bool = True,
) -> dict:
    """Profile a CSV or Excel dataset and return a comprehensive data quality report.

    Args:
        input_file: Path to the CSV or Excel (.xlsx/.xls) file to analyse.
        output_format: Output format for the report (currently only "json").
        generate_cleaned_file: If True, produce a cleaned CSV alongside the report.

    Returns:
        A dictionary containing quality score, issue counts, distributions,
        column profiles, recommendations, and optionally the cleaned file path.
    """
    logger.info("Starting profiling for: %s", input_file)

    # ------------------------------------------------------------------
    # 1. Load the file
    # ------------------------------------------------------------------
    try:
        df = _load_dataframe(input_file)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected error loading file")
        return {"error": f"Failed to load file: {exc}"}

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    if total_rows == 0:
        return {
            "quality_score": 0,
            "rows": 0,
            "columns": total_columns,
            "column_profiles": [],
            "issues": {},
            "distributions": {},
            "suspicious_columns": [],
            "recommendations": ["The dataset is empty. No profiling possible."],
            "cleaned_file": None,
        }

    # ------------------------------------------------------------------
    # 2. Missing value analysis
    # ------------------------------------------------------------------
    missing_by_column: dict[str, dict[str, Any]] = {}
    missing_total = 0
    for col in df.columns:
        col_missing = int(df[col].isna().sum())
        missing_total += col_missing
        missing_by_column[col] = {
            "count": col_missing,
            "percentage": round(col_missing / total_rows * 100, 2),
        }

    # ------------------------------------------------------------------
    # 3. Duplicate row detection
    # ------------------------------------------------------------------
    duplicate_rows = int(df.duplicated().sum())

    # ------------------------------------------------------------------
    # 4. Duplicate ID detection
    # ------------------------------------------------------------------
    duplicate_ids: dict[str, Any] = {"column": None, "count": 0}
    for col in df.columns:
        if _is_id_column(col):
            dup_count = int(df[col].dropna().duplicated().sum())
            if dup_count > duplicate_ids["count"]:
                duplicate_ids = {"column": col, "count": dup_count}

    # ------------------------------------------------------------------
    # 5-8. Column profiling: type inference, email/date/numeric validation
    # ------------------------------------------------------------------
    column_profiles: list[dict[str, Any]] = []
    invalid_emails: dict[str, dict[str, Any]] = {}
    invalid_dates: dict[str, dict[str, Any]] = {}
    invalid_numerics: dict[str, dict[str, Any]] = {}
    distributions: dict[str, dict[str, Any]] = {}
    outliers: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        inferred_type = _infer_column_type(series, col)

        profile: dict[str, Any] = {
            "column": col,
            "inferred_type": inferred_type,
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        # Email validation
        if inferred_type == "email" or _is_email_column(series, col):
            result = _validate_emails(series)
            if result["count"] > 0:
                invalid_emails[col] = result
            profile["email_validation"] = result

        # Date validation
        if inferred_type in ("datetime", "datetime (string)") or _is_date_column(series, col):
            result = _validate_dates(series)
            if result["count"] > 0:
                invalid_dates[col] = result
            profile["date_validation"] = result

        # Numeric validation for object columns that might be numeric
        if series.dtype == object and inferred_type not in ("email", "phone", "categorical"):
            result = _detect_invalid_numerics(series)
            if result["count"] > 0:
                invalid_numerics[col] = result

        # ------------------------------------------------------------------
        # 9-10. Distribution & outlier detection for numeric columns
        # ------------------------------------------------------------------
        if pd.api.types.is_numeric_dtype(series):
            dist = _compute_distribution(series)
            if dist:
                distributions[col] = dist
                profile["distribution"] = dist

            outlier_info = _detect_outliers_iqr(series)
            if outlier_info["count"] > 0:
                outliers[col] = outlier_info
                profile["outliers"] = outlier_info

        column_profiles.append(profile)

    # ------------------------------------------------------------------
    # 11. Suspicious column detection
    # ------------------------------------------------------------------
    suspicious_columns = _detect_suspicious_columns(df)

    # ------------------------------------------------------------------
    # 12. Quality score
    # ------------------------------------------------------------------
    outlier_total = sum(info["count"] for info in outliers.values())
    invalid_email_total = sum(info["count"] for info in invalid_emails.values())
    invalid_date_total = sum(info["count"] for info in invalid_dates.values())
    invalid_numeric_total = sum(info["count"] for info in invalid_numerics.values())

    quality_score = _calculate_quality_score(
        total_rows=total_rows,
        total_cells=total_cells,
        missing_total=missing_total,
        duplicate_rows=duplicate_rows,
        duplicate_id_count=duplicate_ids.get("count", 0),
        invalid_email_count=invalid_email_total,
        invalid_date_count=invalid_date_total,
        invalid_numeric_count=invalid_numeric_total,
        outlier_total=outlier_total,
        suspicious_col_count=len(suspicious_columns),
        total_columns=total_columns,
    )

    # ------------------------------------------------------------------
    # 13. Recommendations
    # ------------------------------------------------------------------
    recommendations = _generate_recommendations(
        missing_by_column=missing_by_column,
        duplicate_rows=duplicate_rows,
        duplicate_ids=duplicate_ids,
        invalid_emails=invalid_emails,
        invalid_dates=invalid_dates,
        invalid_numerics=invalid_numerics,
        outliers=outliers,
        suspicious_columns=suspicious_columns,
    )

    # ------------------------------------------------------------------
    # 15. Cleaned dataset (optional)
    # ------------------------------------------------------------------
    cleaned_file_path: str | None = None
    if generate_cleaned_file:
        try:
            cleaned_file_path = _generate_cleaned_file(df, input_file)
        except Exception as exc:
            logger.exception("Failed to generate cleaned file")
            recommendations.append(f"Could not generate cleaned file: {exc}")

    # ------------------------------------------------------------------
    # 14. Assemble the final report
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "quality_score": quality_score,
        "rows": total_rows,
        "columns": total_columns,
        "column_profiles": column_profiles,
        "issues": {
            "missing_values": {
                "total": missing_total,
                "by_column": missing_by_column,
            },
            "duplicate_rows": duplicate_rows,
            "duplicate_ids": duplicate_ids,
            "invalid_emails": invalid_emails,
            "invalid_dates": invalid_dates,
            "invalid_numerics": invalid_numerics,
            "outliers": outliers,
        },
        "distributions": distributions,
        "suspicious_columns": suspicious_columns,
        "recommendations": recommendations,
        "cleaned_file": cleaned_file_path,
    }

    logger.info(
        "Profiling complete — quality score: %.2f, %d recommendations",
        quality_score,
        len(recommendations),
    )

    return report


# ---------------------------------------------------------------------------
# Mount MCP server as ASGI sub-application
# ---------------------------------------------------------------------------

app.mount("/mcp", mcp.streamable_http_app())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
