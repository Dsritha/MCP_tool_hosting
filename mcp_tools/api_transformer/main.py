"""
API Response Transformer & Enricher
=====================================
INPUT:  A raw API response JSON file (e.g., salesforce_accounts_raw.json) containing
        deeply nested, verbose data from enterprise systems (Salesforce, HubSpot, SAP,
        Stripe, etc.), along with an optional transformation config (YAML/JSON) that
        defines field mappings and enrichment sources.
OUTPUT: Transformed, flattened, analysis-ready data in CSV or JSON format, including:
        - Nested JSON flattened to tabular structure
        - Field renaming and type casting per configuration
        - Null/missing value handling
        - Data enrichment from configurable external sources
        - Schema validation report
        - Transformation summary with statistics

PURPOSE: Enterprise APIs (Salesforce, HubSpot, SAP, Stripe) return complex, deeply
         nested JSON responses that are unusable for direct analysis. This tool
         flattens, transforms, enriches, and standardizes API responses into clean,
         analysis-ready tabular formats for data analysts, BI tools, and downstream
         data pipelines.
"""

import json
import os
import re
import time
import csv
import io
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastmcp import FastMCP

# =============================================================================
# Built-in Enrichment Data
# =============================================================================

# Free email provider domains for domain_info enricher
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "zoho.com", "protonmail.com", "yandex.com",
    "gmx.com", "gmx.net", "live.com", "msn.com", "inbox.com",
    "fastmail.com", "tutanota.com", "hushmail.com", "mailfence.com",
    "disroot.org", "riseup.net", "posteo.de", "mailbox.org",
    "yahoo.co.uk", "yahoo.co.in", "hotmail.co.uk", "live.co.uk",
    "rediffmail.com", "qq.com", "163.com", "126.com",
}

# Country code to continent/region mapping (40+ countries)
COUNTRY_GEO_MAP = {
    "US": {"continent": "North America", "region": "Northern America"},
    "CA": {"continent": "North America", "region": "Northern America"},
    "MX": {"continent": "North America", "region": "Central America"},
    "BR": {"continent": "South America", "region": "South America"},
    "AR": {"continent": "South America", "region": "South America"},
    "CL": {"continent": "South America", "region": "South America"},
    "CO": {"continent": "South America", "region": "South America"},
    "PE": {"continent": "South America", "region": "South America"},
    "GB": {"continent": "Europe", "region": "Northern Europe"},
    "UK": {"continent": "Europe", "region": "Northern Europe"},
    "DE": {"continent": "Europe", "region": "Western Europe"},
    "FR": {"continent": "Europe", "region": "Western Europe"},
    "IT": {"continent": "Europe", "region": "Southern Europe"},
    "ES": {"continent": "Europe", "region": "Southern Europe"},
    "PT": {"continent": "Europe", "region": "Southern Europe"},
    "NL": {"continent": "Europe", "region": "Western Europe"},
    "BE": {"continent": "Europe", "region": "Western Europe"},
    "CH": {"continent": "Europe", "region": "Western Europe"},
    "AT": {"continent": "Europe", "region": "Western Europe"},
    "SE": {"continent": "Europe", "region": "Northern Europe"},
    "NO": {"continent": "Europe", "region": "Northern Europe"},
    "DK": {"continent": "Europe", "region": "Northern Europe"},
    "FI": {"continent": "Europe", "region": "Northern Europe"},
    "IE": {"continent": "Europe", "region": "Northern Europe"},
    "PL": {"continent": "Europe", "region": "Eastern Europe"},
    "CZ": {"continent": "Europe", "region": "Eastern Europe"},
    "RO": {"continent": "Europe", "region": "Eastern Europe"},
    "HU": {"continent": "Europe", "region": "Eastern Europe"},
    "RU": {"continent": "Europe", "region": "Eastern Europe"},
    "UA": {"continent": "Europe", "region": "Eastern Europe"},
    "GR": {"continent": "Europe", "region": "Southern Europe"},
    "TR": {"continent": "Asia", "region": "Western Asia"},
    "CN": {"continent": "Asia", "region": "Eastern Asia"},
    "JP": {"continent": "Asia", "region": "Eastern Asia"},
    "KR": {"continent": "Asia", "region": "Eastern Asia"},
    "IN": {"continent": "Asia", "region": "Southern Asia"},
    "PK": {"continent": "Asia", "region": "Southern Asia"},
    "BD": {"continent": "Asia", "region": "Southern Asia"},
    "ID": {"continent": "Asia", "region": "South-Eastern Asia"},
    "TH": {"continent": "Asia", "region": "South-Eastern Asia"},
    "VN": {"continent": "Asia", "region": "South-Eastern Asia"},
    "MY": {"continent": "Asia", "region": "South-Eastern Asia"},
    "SG": {"continent": "Asia", "region": "South-Eastern Asia"},
    "PH": {"continent": "Asia", "region": "South-Eastern Asia"},
    "AE": {"continent": "Asia", "region": "Western Asia"},
    "SA": {"continent": "Asia", "region": "Western Asia"},
    "IL": {"continent": "Asia", "region": "Western Asia"},
    "AU": {"continent": "Oceania", "region": "Australia and New Zealand"},
    "NZ": {"continent": "Oceania", "region": "Australia and New Zealand"},
    "ZA": {"continent": "Africa", "region": "Southern Africa"},
    "NG": {"continent": "Africa", "region": "Western Africa"},
    "KE": {"continent": "Africa", "region": "Eastern Africa"},
    "EG": {"continent": "Africa", "region": "Northern Africa"},
    "MA": {"continent": "Africa", "region": "Northern Africa"},
    "GH": {"continent": "Africa", "region": "Western Africa"},
    "TZ": {"continent": "Africa", "region": "Eastern Africa"},
    "ET": {"continent": "Africa", "region": "Eastern Africa"},
}

# Country name to code mapping for reverse lookups
COUNTRY_NAME_TO_CODE = {
    "united states": "US", "usa": "US", "united states of america": "US",
    "canada": "CA", "mexico": "MX", "brazil": "BR", "argentina": "AR",
    "chile": "CL", "colombia": "CO", "peru": "PE",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "england": "GB",
    "germany": "DE", "france": "FR", "italy": "IT", "spain": "ES",
    "portugal": "PT", "netherlands": "NL", "belgium": "BE", "switzerland": "CH",
    "austria": "AT", "sweden": "SE", "norway": "NO", "denmark": "DK",
    "finland": "FI", "ireland": "IE", "poland": "PL", "czech republic": "CZ",
    "czechia": "CZ", "romania": "RO", "hungary": "HU", "russia": "RU",
    "ukraine": "UA", "greece": "GR", "turkey": "TR", "china": "CN",
    "japan": "JP", "south korea": "KR", "korea": "KR", "india": "IN",
    "pakistan": "PK", "bangladesh": "BD", "indonesia": "ID", "thailand": "TH",
    "vietnam": "VN", "malaysia": "MY", "singapore": "SG", "philippines": "PH",
    "united arab emirates": "AE", "uae": "AE", "saudi arabia": "SA",
    "israel": "IL", "australia": "AU", "new zealand": "NZ",
    "south africa": "ZA", "nigeria": "NG", "kenya": "KE", "egypt": "EG",
    "morocco": "MA", "ghana": "GH", "tanzania": "TZ", "ethiopia": "ET",
}

# Common API metadata fields to auto-exclude in default mode
DEFAULT_METADATA_FIELDS = {
    "attributes.type", "attributes.url", "attributes.referenceId",
    "links", "self", "next", "previous", "meta",
    "pagination", "cursor", "nextPageToken", "prevPageToken",
    "totalSize", "done", "nextRecordsUrl",
    "@odata.context", "@odata.type", "@odata.id",
    "_links", "_embedded", "_metadata",
}

# PII detection patterns
PII_PATTERNS = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "phone": re.compile(r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}$"),
    "ssn": re.compile(r"^\d{3}-?\d{2}-?\d{4}$"),
    "credit_card": re.compile(r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$"),
    "ip_address": re.compile(
        r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def flatten_json(
    nested: Any,
    prefix: str = "",
    separator: str = ".",
    max_depth: int = 10,
    current_depth: int = 0,
) -> Dict[str, Any]:
    """
    Recursively flatten a nested JSON object into dot-notation keys.

    Args:
        nested: The JSON object/value to flatten.
        prefix: Current key prefix for recursion.
        separator: Separator for nested keys (default: dot).
        max_depth: Maximum recursion depth.
        current_depth: Current recursion depth tracker.

    Returns:
        A flat dictionary with dot-notation keys.
    """
    result = {}

    if current_depth >= max_depth:
        # At max depth, serialize remaining structure as JSON string
        result[prefix] = json.dumps(nested) if isinstance(nested, (dict, list)) else nested
        return result

    if isinstance(nested, dict):
        for key, value in nested.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            if isinstance(value, dict):
                result.update(
                    flatten_json(value, new_key, separator, max_depth, current_depth + 1)
                )
            elif isinstance(value, list):
                # Handle arrays: if list of primitives, serialize; if list of dicts, expand
                if len(value) > 0 and isinstance(value[0], dict):
                    # Serialize complex arrays as JSON strings for tabular output
                    result[new_key] = json.dumps(value)
                else:
                    # Serialize primitive arrays as JSON strings
                    result[new_key] = json.dumps(value) if value else "[]"
            else:
                result[new_key] = value
    elif isinstance(nested, list):
        # Top-level list handling — serialize
        result[prefix] = json.dumps(nested) if nested else "[]"
    else:
        result[prefix] = nested

    return result


def auto_detect_records(data: Any) -> List[Dict[str, Any]]:
    """
    Auto-detect the record structure from raw JSON data.

    Handles:
    - JSON array of records: [{...}, {...}]
    - JSON object with a data key: {"records": [...], "data": [...], "results": [...]}
    - Single record: {...} (wrapped in array)

    Args:
        data: Parsed JSON data.

    Returns:
        A list of record dictionaries.
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Check for common data wrapper keys
        data_keys = ["records", "data", "results", "items", "entries", "rows",
                      "hits", "documents", "objects", "values", "elements",
                      "content", "payload", "response", "body"]
        for key in data_keys:
            if key in data and isinstance(data[key], list):
                return data[key]

        # Check for any key that contains a list of dicts
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                return value

        # Single record — wrap in array
        return [data]

    # Fallback: wrap primitive in a record
    return [{"value": data}]


def load_transformation_config(config_path: str) -> Dict[str, Any]:
    """
    Load a transformation configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")

    if config_path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content) or {}
    elif config_path.suffix == ".json":
        return json.loads(content)
    else:
        # Try YAML first, then JSON
        try:
            return yaml.safe_load(content) or {}
        except yaml.YAMLError:
            return json.loads(content)


def get_default_config() -> Dict[str, Any]:
    """
    Return a sensible default transformation configuration.

    This auto-detects common API patterns and applies standard transformations.
    """
    return {
        "field_mappings": {},
        "type_casts": {},
        "exclude_fields": [],
        "include_only": [],
        "null_handling": {
            "strategy": "fill",
            "fill_value": "",
            "numeric_fill": 0,
        },
    }


def apply_default_field_cleanup(columns: List[str]) -> Dict[str, str]:
    """
    Auto-detect and rename common API patterns for default mode.

    Strips common prefixes like 'attributes.' and converts to snake_case.

    Args:
        columns: List of column names.

    Returns:
        Mapping of original column names to cleaned names.
    """
    mappings = {}
    for col in columns:
        cleaned = col

        # Remove common prefixes
        for prefix in ["attributes.", "properties.", "fields.", "data.", "record."]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]

        # Convert CamelCase to snake_case
        cleaned = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", cleaned)
        cleaned = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", cleaned)
        cleaned = cleaned.lower()

        # Replace dots and special chars with underscores
        cleaned = re.sub(r"[.\-\s]+", "_", cleaned)

        # Remove leading/trailing underscores and collapse multiples
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")

        if cleaned != col:
            mappings[col] = cleaned

    return mappings


def apply_field_mappings(df: pd.DataFrame, mappings: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    """
    Rename DataFrame columns based on field mappings.

    Args:
        df: Input DataFrame.
        mappings: Dictionary mapping old column names to new names.

    Returns:
        Tuple of (renamed DataFrame, count of fields renamed).
    """
    # Only rename columns that actually exist in the DataFrame
    valid_mappings = {k: v for k, v in mappings.items() if k in df.columns}
    if valid_mappings:
        df = df.rename(columns=valid_mappings)
    return df, len(valid_mappings)


def apply_type_casts(df: pd.DataFrame, type_casts: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    """
    Cast DataFrame columns to specified types with error handling.

    Supported types: string, int, float, datetime, bool.

    Args:
        df: Input DataFrame.
        type_casts: Dictionary mapping column names to target types.

    Returns:
        Tuple of (cast DataFrame, count of fields cast).
    """
    cast_count = 0
    for col, target_type in type_casts.items():
        if col not in df.columns:
            continue

        try:
            if target_type == "string" or target_type == "str":
                df[col] = df[col].astype(str).replace("nan", "")
            elif target_type == "int" or target_type == "integer":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif target_type == "float" or target_type == "double":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "datetime" or target_type == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target_type == "bool" or target_type == "boolean":
                df[col] = df[col].map(
                    lambda x: True if str(x).lower() in ("true", "1", "yes") else
                    (False if str(x).lower() in ("false", "0", "no") else None)
                )
            cast_count += 1
        except Exception:
            # Silently skip columns that can't be cast
            pass

    return df, cast_count


def apply_null_handling(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, int]:
    """
    Apply null/missing value handling strategy.

    Strategies:
    - fill: Replace nulls with specified fill values.
    - drop_row: Drop rows where any mapped field is null.
    - drop_column: Drop columns that are entirely null.

    Args:
        df: Input DataFrame.
        config: Null handling configuration.

    Returns:
        Tuple of (processed DataFrame, count of nulls handled).
    """
    strategy = config.get("strategy", "fill")
    fill_value = config.get("fill_value", "")
    numeric_fill = config.get("numeric_fill", 0)

    null_count_before = int(df.isnull().sum().sum())

    if strategy == "fill":
        # Fill numeric columns with numeric_fill, others with fill_value
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(numeric_fill)
            else:
                df[col] = df[col].fillna(fill_value)
    elif strategy == "drop_row":
        df = df.dropna()
    elif strategy == "drop_column":
        # Drop columns that are entirely null
        df = df.dropna(axis=1, how="all")

    null_count_after = int(df.isnull().sum().sum())
    nulls_handled = null_count_before - null_count_after

    return df, nulls_handled


def apply_field_filtering(
    df: pd.DataFrame,
    include_only: List[str],
    exclude_fields: List[str],
    is_default: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Filter DataFrame columns based on include/exclude lists.

    Args:
        df: Input DataFrame.
        include_only: Whitelist of columns to keep (empty = keep all).
        exclude_fields: Blacklist of columns to remove.
        is_default: Whether using default config (auto-exclude metadata).

    Returns:
        Tuple of (filtered DataFrame, count of fields excluded).
    """
    original_cols = set(df.columns)

    # Apply include_only whitelist
    if include_only:
        valid_includes = [c for c in include_only if c in df.columns]
        if valid_includes:
            df = df[valid_includes]

    # Apply exclude_fields blacklist
    if exclude_fields:
        cols_to_drop = [c for c in exclude_fields if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # In default mode, auto-exclude known metadata fields
    if is_default:
        for meta_field in DEFAULT_METADATA_FIELDS:
            if meta_field in df.columns:
                df = df.drop(columns=[meta_field], errors="ignore")
            # Also check for flattened versions
            for col in list(df.columns):
                if col.startswith(meta_field) or col == meta_field:
                    df = df.drop(columns=[col], errors="ignore")

    excluded_count = len(original_cols) - len(set(df.columns))
    return df, excluded_count


# =============================================================================
# Enrichment Functions
# =============================================================================


def enrich_domain_info(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Domain info enricher: extract domain from email/website columns.

    Adds columns: domain, domain_tld, is_free_email.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (enriched DataFrame, count of columns added).
    """
    added = 0

    # Find email or website columns
    email_col = None
    website_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "email" in col_lower and email_col is None:
            email_col = col
        elif ("website" in col_lower or "url" in col_lower or "domain" in col_lower == col_lower) and website_col is None:
            website_col = col

    source_col = email_col or website_col
    if source_col is None:
        return df, 0

    def extract_domain(value):
        """Extract domain from email or URL."""
        if pd.isna(value) or not str(value).strip():
            return ""
        value = str(value).strip()
        # Email
        if "@" in value:
            return value.split("@")[-1].lower()
        # URL
        value = re.sub(r"^https?://", "", value)
        value = re.sub(r"^www\.", "", value)
        value = value.split("/")[0].lower()
        return value

    df["domain"] = df[source_col].apply(extract_domain)
    added += 1

    df["domain_tld"] = df["domain"].apply(
        lambda d: d.split(".")[-1] if d and "." in d else ""
    )
    added += 1

    df["is_free_email"] = df["domain"].apply(
        lambda d: str(d in FREE_EMAIL_DOMAINS).lower() if d else "false"
    )
    added += 1

    return df, added


def enrich_geo_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Geo lookup enricher: add continent and region from country data.

    Uses built-in country code/name to continent/region mapping.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (enriched DataFrame, count of columns added).
    """
    added = 0

    # Find country column
    country_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "country" in col_lower:
            country_col = col
            break

    if country_col is None:
        return df, 0

    def lookup_geo(value, field):
        """Look up geographic info from country value."""
        if pd.isna(value) or not str(value).strip():
            return ""
        value = str(value).strip()

        # Try direct code lookup (uppercase)
        code = value.upper()
        if code in COUNTRY_GEO_MAP:
            return COUNTRY_GEO_MAP[code].get(field, "")

        # Try name lookup
        name_lower = value.lower()
        if name_lower in COUNTRY_NAME_TO_CODE:
            code = COUNTRY_NAME_TO_CODE[name_lower]
            return COUNTRY_GEO_MAP.get(code, {}).get(field, "")

        return ""

    df["continent"] = df[country_col].apply(lambda v: lookup_geo(v, "continent"))
    added += 1

    df["region"] = df[country_col].apply(lambda v: lookup_geo(v, "region"))
    added += 1

    return df, added


def enrich_data_classification(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Data classification enricher: detect PII and data types in columns.

    Adds metadata columns: _is_pii, _data_type_detected for columns
    that match PII patterns.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (enriched DataFrame, count of columns added).
    """
    added = 0
    pii_columns = []
    type_info = {}

    for col in df.columns:
        # Skip already-enriched columns
        if col.startswith("_") or col in ("domain", "domain_tld", "is_free_email",
                                           "continent", "region"):
            continue

        # Sample non-null values for pattern detection
        sample = df[col].dropna().astype(str).head(100)
        if sample.empty:
            continue

        # Check each PII pattern
        for pii_type, pattern in PII_PATTERNS.items():
            match_rate = sample.apply(lambda v: bool(pattern.match(str(v)))).mean()
            if match_rate > 0.5:  # More than 50% match
                pii_columns.append(col)
                type_info[col] = pii_type
                break

    if pii_columns:
        # Add a summary column indicating which columns contain PII
        pii_summary = {col: type_info.get(col, "unknown") for col in pii_columns}
        df["_pii_columns_detected"] = json.dumps(pii_summary)
        added += 1

        # Add per-column PII flags
        for col in pii_columns:
            flag_col = f"_{col}_is_pii"
            df[flag_col] = "true"
            df[f"_{col}_pii_type"] = type_info.get(col, "unknown")
            added += 2

    return df, added


# Enricher registry
ENRICHERS = {
    "domain_info": enrich_domain_info,
    "geo_lookup": enrich_geo_lookup,
    "data_classification": enrich_data_classification,
}


# =============================================================================
# Schema Validation
# =============================================================================


def validate_schema(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate the transformed DataFrame against expected schema.

    Checks:
    - Expected columns present (from field mappings)
    - Null rate acceptable (< 50% per column)
    - Type consistency

    Args:
        df: Transformed DataFrame.
        config: Transformation configuration.

    Returns:
        Schema validation report dictionary.
    """
    warnings = []
    expected_columns_present = True
    null_rate_acceptable = True
    type_consistency = True

    # Check expected columns from field mappings
    field_mappings = config.get("field_mappings", {})
    if field_mappings:
        expected_cols = set(field_mappings.values())
        actual_cols = set(df.columns)
        missing = expected_cols - actual_cols
        if missing:
            expected_columns_present = False
            warnings.append(f"Missing expected columns: {', '.join(sorted(missing))}")

    # Check null rates
    for col in df.columns:
        null_rate = df[col].isnull().mean()
        if null_rate > 0.5:
            null_rate_acceptable = False
            warnings.append(
                f"Column '{col}' has high null rate: {null_rate:.1%}"
            )

    # Check type consistency
    type_casts = config.get("type_casts", {})
    for col, expected_type in type_casts.items():
        if col not in df.columns:
            continue
        try:
            if expected_type in ("int", "integer"):
                # Check if numeric
                non_null = df[col].dropna()
                if not non_null.empty and not pd.api.types.is_numeric_dtype(non_null):
                    type_consistency = False
                    warnings.append(
                        f"Column '{col}' expected type '{expected_type}' but contains non-numeric values"
                    )
            elif expected_type in ("float", "double"):
                non_null = df[col].dropna()
                if not non_null.empty and not pd.api.types.is_numeric_dtype(non_null):
                    type_consistency = False
                    warnings.append(
                        f"Column '{col}' expected type '{expected_type}' but contains non-numeric values"
                    )
        except Exception:
            pass

    # Determine overall status
    all_passed = expected_columns_present and null_rate_acceptable and type_consistency
    status = "passed" if all_passed else ("warnings" if warnings else "passed")

    return {
        "status": status,
        "checks": {
            "expected_columns_present": expected_columns_present,
            "null_rate_acceptable": null_rate_acceptable,
            "type_consistency": type_consistency,
        },
        "warnings": warnings,
    }


# =============================================================================
# Output Writers
# =============================================================================


def write_csv_output(df: pd.DataFrame, output_path: str) -> str:
    """
    Write DataFrame to CSV with UTF-8 BOM encoding for Excel compatibility.

    Args:
        df: DataFrame to write.
        output_path: Output file path.

    Returns:
        The output file path.
    """
    df.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return output_path


def write_json_output(df: pd.DataFrame, output_path: str) -> str:
    """
    Write DataFrame to JSON as an array of flat records.

    Args:
        df: DataFrame to write.
        output_path: Output file path.

    Returns:
        The output file path.
    """
    # Convert to records and handle NaN/NaT values
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return output_path


def write_excel_output(df: pd.DataFrame, output_path: str) -> str:
    """
    Write DataFrame to Excel with auto-sized columns.

    Args:
        df: DataFrame to write.
        output_path: Output file path.

    Returns:
        The output file path.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transformed Data")

        # Auto-size columns
        worksheet = writer.sheets["Transformed Data"]
        for col_idx, col_name in enumerate(df.columns, 1):
            # Calculate max width from header and data
            max_len = len(str(col_name))
            for value in df[col_name].head(100):
                val_len = len(str(value)) if pd.notna(value) else 0
                max_len = max(max_len, val_len)
            # Cap at 50 characters and add padding
            adjusted_width = min(max_len + 2, 50)
            col_letter = worksheet.cell(row=1, column=col_idx).column_letter
            worksheet.column_dimensions[col_letter].width = adjusted_width

    return output_path


# =============================================================================
# MCP Server & Tool Definition
# =============================================================================

mcp = FastMCP("API Response Transformer")
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(title="API Response Transformer & Enricher", lifespan=mcp_app.lifespan)


@mcp.tool()
async def transform_and_enrich_api_response(
    input_json_path: str,
    transformation_config: str = None,
    enrichment_sources: str = "[]",
    output_format: str = "csv",
    flatten_depth: int = 10,
) -> dict:
    """
    Transform and enrich enterprise API responses into clean, analysis-ready data.

    Takes a raw API response JSON file (from Salesforce, HubSpot, SAP, Stripe, etc.)
    and transforms it into a flattened, standardized tabular format with optional
    enrichment from built-in data sources.

    Args:
        input_json_path: Path to the input JSON file containing raw API response data.
        transformation_config: Optional path to a YAML/JSON config file defining
            field mappings, type casts, exclusions, and null handling strategy.
            If not provided, sensible defaults are applied.
        enrichment_sources: JSON string array of enrichment source names to apply.
            Available: "domain_info", "geo_lookup", "data_classification".
            Example: '["domain_info", "geo_lookup"]'. Default: "[]" (no enrichment).
        output_format: Output format - "csv" (default), "json", or "xlsx".
        flatten_depth: Maximum depth for recursive JSON flattening (default: 10).

    Returns:
        Dictionary with transformation status, summary statistics, schema validation
        report, and output file path.
    """
    start_time = time.time()

    # -------------------------------------------------------------------------
    # 1. Validate inputs
    # -------------------------------------------------------------------------
    input_path = Path(input_json_path)
    if not input_path.exists():
        return {
            "status": "error",
            "error": f"Input file not found: {input_json_path}",
            "summary": {},
            "schema_validation": {},
            "output_file": "",
            "transformation_config_used": "",
        }

    if output_format not in ("csv", "json", "xlsx"):
        return {
            "status": "error",
            "error": f"Unsupported output format: {output_format}. Use 'csv', 'json', or 'xlsx'.",
            "summary": {},
            "schema_validation": {},
            "output_file": "",
            "transformation_config_used": "",
        }

    # Parse enrichment sources from JSON string
    try:
        enrichment_list = json.loads(enrichment_sources) if enrichment_sources else []
        if not isinstance(enrichment_list, list):
            enrichment_list = []
    except (json.JSONDecodeError, TypeError):
        enrichment_list = []

    # -------------------------------------------------------------------------
    # 2. Load input JSON
    # -------------------------------------------------------------------------
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "error": f"Invalid JSON in input file: {str(e)}",
            "summary": {},
            "schema_validation": {},
            "output_file": "",
            "transformation_config_used": "",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error reading input file: {str(e)}",
            "summary": {},
            "schema_validation": {},
            "output_file": "",
            "transformation_config_used": "",
        }

    # -------------------------------------------------------------------------
    # 3. Auto-detect records
    # -------------------------------------------------------------------------
    records = auto_detect_records(raw_data)
    if not records:
        return {
            "status": "error",
            "error": "No records found in input JSON. The file may be empty or have an unrecognized structure.",
            "summary": {},
            "schema_validation": {},
            "output_file": "",
            "transformation_config_used": "",
        }

    input_record_count = len(records)

    # -------------------------------------------------------------------------
    # 4. Flatten records
    # -------------------------------------------------------------------------
    flattened_records = []
    for record in records:
        try:
            flat = flatten_json(record, max_depth=flatten_depth)
            flattened_records.append(flat)
        except Exception:
            # Skip records that can't be flattened
            continue

    if not flattened_records:
        return {
            "status": "error",
            "error": "All records failed to flatten. Check input data structure.",
            "summary": {},
            "schema_validation": {},
            "output_file": "",
            "transformation_config_used": "",
        }

    # Create DataFrame
    df = pd.DataFrame(flattened_records)
    input_field_count = len(df.columns)

    # -------------------------------------------------------------------------
    # 5. Load transformation config
    # -------------------------------------------------------------------------
    using_default_config = transformation_config is None
    config_label = "default"

    if transformation_config:
        try:
            config = load_transformation_config(transformation_config)
            config_label = str(transformation_config)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error loading transformation config: {str(e)}",
                "summary": {},
                "schema_validation": {},
                "output_file": "",
                "transformation_config_used": "",
            }
    else:
        config = get_default_config()

    # -------------------------------------------------------------------------
    # 6. Apply field filtering (exclude/include)
    # -------------------------------------------------------------------------
    include_only = config.get("include_only", [])
    exclude_fields = config.get("exclude_fields", [])
    df, fields_excluded = apply_field_filtering(
        df, include_only, exclude_fields, is_default=using_default_config
    )

    # -------------------------------------------------------------------------
    # 7. Apply field mappings (renaming)
    # -------------------------------------------------------------------------
    field_mappings = config.get("field_mappings", {})

    if using_default_config:
        # In default mode, auto-generate clean field names
        auto_mappings = apply_default_field_cleanup(list(df.columns))
        field_mappings.update(auto_mappings)

    df, fields_renamed = apply_field_mappings(df, field_mappings)

    # -------------------------------------------------------------------------
    # 8. Apply type casts
    # -------------------------------------------------------------------------
    type_casts = config.get("type_casts", {})
    df, fields_type_cast = apply_type_casts(df, type_casts)

    # -------------------------------------------------------------------------
    # 9. Apply null handling
    # -------------------------------------------------------------------------
    null_config = config.get("null_handling", {
        "strategy": "fill",
        "fill_value": "",
        "numeric_fill": 0,
    })
    df, nulls_handled = apply_null_handling(df, null_config)

    # -------------------------------------------------------------------------
    # 10. Apply enrichment
    # -------------------------------------------------------------------------
    enrichment_columns_added = 0
    for enricher_name in enrichment_list:
        enricher_func = ENRICHERS.get(enricher_name)
        if enricher_func:
            try:
                df, cols_added = enricher_func(df)
                enrichment_columns_added += cols_added
            except Exception:
                # Gracefully skip failed enrichers
                pass

    # -------------------------------------------------------------------------
    # 11. Schema validation
    # -------------------------------------------------------------------------
    schema_validation = validate_schema(df, config)

    # -------------------------------------------------------------------------
    # 12. Write output
    # -------------------------------------------------------------------------
    output_record_count = len(df)
    output_field_count = len(df.columns)

    # Determine output file path
    input_stem = input_path.stem.replace("_raw", "")
    output_dir = input_path.parent

    if output_format == "csv":
        output_filename = f"{input_stem}_transformed.csv"
        output_path = str(output_dir / output_filename)
        write_csv_output(df, output_path)
    elif output_format == "json":
        output_filename = f"{input_stem}_transformed.json"
        output_path = str(output_dir / output_filename)
        write_json_output(df, output_path)
    elif output_format == "xlsx":
        output_filename = f"{input_stem}_transformed.xlsx"
        output_path = str(output_dir / output_filename)
        write_excel_output(df, output_path)
    else:
        output_filename = f"{input_stem}_transformed.csv"
        output_path = str(output_dir / output_filename)
        write_csv_output(df, output_path)

    # -------------------------------------------------------------------------
    # 13. Calculate processing time and build summary
    # -------------------------------------------------------------------------
    processing_time_ms = round((time.time() - start_time) * 1000)

    summary = {
        "input_records": input_record_count,
        "output_records": output_record_count,
        "input_fields": input_field_count,
        "output_fields": output_field_count,
        "fields_renamed": fields_renamed,
        "fields_excluded": fields_excluded,
        "fields_type_cast": fields_type_cast,
        "enrichment_columns_added": enrichment_columns_added,
        "nulls_handled": nulls_handled,
        "processing_time_ms": processing_time_ms,
    }

    return {
        "status": "success",
        "summary": summary,
        "schema_validation": schema_validation,
        "output_file": output_filename,
        "transformation_config_used": config_label,
    }


# =============================================================================
# Mount MCP server as ASGI sub-application
# =============================================================================

app.mount("/mcp", mcp_app)


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
