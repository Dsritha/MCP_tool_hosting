"""
Log Analysis & Incident Report Generator
==========================================
INPUT:  Log files — either a single log file (.log, .txt), multiple log files,
        or a ZIP archive containing log files (e.g., application_logs.zip with
        app.log, api.log, worker.log, database.log)
OUTPUT: Comprehensive incident analysis report including:
        - Time range of analyzed logs
        - Severity classification (LOW / MEDIUM / HIGH / CRITICAL)
        - Total error/warning/info counts
        - Top error patterns with frequencies
        - Suspected incident start time (anomaly spike detection)
        - Affected services identification
        - Error timeline data (CSV)
        - Full incident report (JSON)
        - Optionally: compare two incidents for pattern similarity

PURPOSE: Parses application/server log files, normalizes timestamps and severity
         levels, detects error patterns using regex and frequency analysis,
         identifies anomalies using statistical methods (z-score), and generates
         structured incident reports for DevOps/SRE teams to accelerate
         root cause analysis and incident response.
"""

import os
import re
import csv
import json
import zipfile
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from dateutil import parser as dateutil_parser

from fastapi import FastAPI
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported log file extensions
LOG_EXTENSIONS = {".log", ".txt", ".out", ".err"}

# Severity level ordering (lowest to highest)
SEVERITY_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4, "FATAL": 4, "UNKNOWN": -1}

# Canonical severity names
SEVERITY_ALIASES = {
    "TRACE": "DEBUG",
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "INFORMATION": "INFO",
    "NOTICE": "INFO",
    "WARN": "WARNING",
    "WARNING": "WARNING",
    "ERR": "ERROR",
    "ERROR": "ERROR",
    "CRIT": "CRITICAL",
    "CRITICAL": "CRITICAL",
    "FATAL": "FATAL",
    "EMERG": "FATAL",
    "EMERGENCY": "FATAL",
    "ALERT": "CRITICAL",
    "PANIC": "FATAL",
}

# Default z-score threshold for anomaly detection
Z_SCORE_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Log format regex patterns
# ---------------------------------------------------------------------------

# Pattern 1: "2026-08-19 10:07:14 ERROR [service-name] Message"
PATTERN_STANDARD = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(?P<severity>[A-Z]+)\s+"
    r"\[(?P<service>[^\]]+)\]\s+"
    r"(?P<message>.+)$"
)

# Pattern 2: "[2026-08-19T10:07:14Z] ERROR service-name: Message"
PATTERN_BRACKET_ISO = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?(?:[+-]\d{2}:?\d{2})?)\]\s+"
    r"(?P<severity>[A-Z]+)\s+"
    r"(?P<service>[\w\-\.]+):\s+"
    r"(?P<message>.+)$"
)

# Pattern 3: Syslog — "Aug 19 10:07:14 hostname service[pid]: Message"
PATTERN_SYSLOG = re.compile(
    r"^(?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<hostname>[\w\-\.]+)\s+"
    r"(?P<service>[\w\-\.]+)(?:\[(?P<pid>\d+)\])?:\s+"
    r"(?P<message>.+)$"
)

# Pattern 4: Apache/Nginx combined log format
PATTERN_APACHE = re.compile(
    r"^(?P<client_ip>\S+)\s+\S+\s+\S+\s+"
    r"\[(?P<timestamp>\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]\s+"
    r"\"(?P<request>[^\"]*)\"\s+"
    r"(?P<status>\d{3})\s+"
    r"(?P<size>\S+)"
    r"(?:\s+\"(?P<referer>[^\"]*)\"\s+\"(?P<user_agent>[^\"]*)\")?"
)

# Pattern 5: JSON structured log line
PATTERN_JSON_LINE = re.compile(r"^\s*\{.*\}\s*$")

# Pattern 6: ISO timestamp with severity (flexible)
PATTERN_ISO_FLEX = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?P<severity>[A-Z]+)\s+"
    r"(?:(?:\[(?P<service>[^\]]+)\]|(?P<service2>[\w\-\.]+):)\s+)?"
    r"(?P<message>.+)$"
)

# ---------------------------------------------------------------------------
# Normalization regex patterns (for error pattern grouping)
# ---------------------------------------------------------------------------

# Replace IPv4 addresses
RE_IPV4 = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
# Replace IPv6 addresses
RE_IPV6 = re.compile(r"\b[0-9a-fA-F]{1,4}(:[0-9a-fA-F]{1,4}){7}\b")
# Replace UUIDs
RE_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
# Replace hex strings (8+ chars)
RE_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
# Replace port numbers (after colon)
RE_PORT = re.compile(r":(\d{2,5})\b")
# Replace standalone numbers (but not in words)
RE_NUMBERS = re.compile(r"\b\d+\b")
# Replace file paths
RE_FILEPATH = re.compile(r"(/[\w\-\.]+)+/?")
# Replace timestamps embedded in messages
RE_EMBEDDED_TS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
# Replace email addresses
RE_EMAIL = re.compile(r"\b[\w\.\-]+@[\w\.\-]+\.\w+\b")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def normalize_severity(raw: str) -> str:
    """Normalize a raw severity string to a canonical level."""
    upper = raw.strip().upper()
    return SEVERITY_ALIASES.get(upper, "UNKNOWN")


def normalize_timestamp(raw: str) -> Optional[datetime]:
    """
    Parse a raw timestamp string into a timezone-aware UTC datetime.
    Returns None if parsing fails.
    """
    if not raw:
        return None
    try:
        dt = dateutil_parser.parse(raw, fuzzy=True)
        # If naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except (ValueError, OverflowError):
        return None


def normalize_error_pattern(message: str) -> str:
    """
    Normalize variable parts of an error message to create a groupable pattern.
    Replaces IPs, UUIDs, numbers, paths, etc. with placeholders.
    """
    pattern = message
    pattern = RE_EMBEDDED_TS.sub("{timestamp}", pattern)
    pattern = RE_EMAIL.sub("{email}", pattern)
    pattern = RE_UUID.sub("{uuid}", pattern)
    pattern = RE_IPV4.sub("{ip}", pattern)
    pattern = RE_IPV6.sub("{ipv6}", pattern)
    pattern = RE_HEX.sub("{hex}", pattern)
    pattern = RE_PORT.sub(":{port}", pattern)
    pattern = RE_FILEPATH.sub("{path}", pattern)
    pattern = RE_NUMBERS.sub("{n}", pattern)
    return pattern


def is_binary_file(filepath: str, sample_size: int = 8192) -> bool:
    """Check if a file appears to be binary by reading a sample."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(sample_size)
        # If more than 10% of bytes are non-text, consider it binary
        text_chars = set(range(32, 127)) | {9, 10, 13}  # printable + tab, newline, CR
        non_text = sum(1 for b in chunk if b not in text_chars)
        return non_text > len(chunk) * 0.10
    except Exception:
        return True


def severity_meets_threshold(severity: str, threshold: str) -> bool:
    """Check if a severity level meets or exceeds the threshold."""
    sev_val = SEVERITY_ORDER.get(severity, -1)
    thr_val = SEVERITY_ORDER.get(normalize_severity(threshold), 0)
    return sev_val >= thr_val


def parse_time_range(time_range: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse a time range specification.
    Supports:
      - ISO interval: "2026-08-19T10:00:00Z/2026-08-19T11:00:00Z"
      - Relative: "last_1h", "last_24h", "last_7d", "last_30m"
    Returns (start, end) datetimes in UTC.
    """
    if not time_range:
        return None, None

    # Check for ISO interval (two timestamps separated by /)
    if "/" in time_range and not time_range.startswith("last_"):
        parts = time_range.split("/", 1)
        start = normalize_timestamp(parts[0])
        end = normalize_timestamp(parts[1])
        return start, end

    # Check for relative time ranges
    relative_match = re.match(r"last_(\d+)([mhd])", time_range.lower())
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        now = datetime.now(timezone.utc)
        if unit == "m":
            delta = timedelta(minutes=amount)
        elif unit == "h":
            delta = timedelta(hours=amount)
        elif unit == "d":
            delta = timedelta(days=amount)
        else:
            return None, None
        return now - delta, now

    return None, None


def parse_log_line(line: str, source_file: str = "") -> Optional[Dict[str, Any]]:
    """
    Attempt to parse a single log line using multiple format patterns.
    Returns a dict with keys: timestamp, severity, service, message, raw
    or None if the line is empty.
    """
    line = line.rstrip("\n\r")
    if not line.strip():
        return None

    # Try JSON structured log
    if PATTERN_JSON_LINE.match(line):
        try:
            data = json.loads(line)
            ts_raw = data.get("timestamp") or data.get("time") or data.get("@timestamp") or data.get("ts", "")
            sev_raw = (
                data.get("level") or data.get("severity") or data.get("loglevel") or data.get("log_level", "INFO")
            )
            service = (
                data.get("service") or data.get("logger") or data.get("app") or data.get("source", "")
            )
            message = (
                data.get("message") or data.get("msg") or data.get("text", json.dumps(data))
            )
            return {
                "timestamp": normalize_timestamp(str(ts_raw)),
                "severity": normalize_severity(str(sev_raw)),
                "service": str(service),
                "message": str(message),
                "raw": line,
            }
        except (json.JSONDecodeError, TypeError):
            pass  # Fall through to other patterns

    # Try Pattern 1: Standard format
    m = PATTERN_STANDARD.match(line)
    if m:
        return {
            "timestamp": normalize_timestamp(m.group("timestamp")),
            "severity": normalize_severity(m.group("severity")),
            "service": m.group("service").strip(),
            "message": m.group("message").strip(),
            "raw": line,
        }

    # Try Pattern 2: Bracket ISO format
    m = PATTERN_BRACKET_ISO.match(line)
    if m:
        return {
            "timestamp": normalize_timestamp(m.group("timestamp")),
            "severity": normalize_severity(m.group("severity")),
            "service": m.group("service").strip(),
            "message": m.group("message").strip(),
            "raw": line,
        }

    # Try Pattern 6: ISO flexible format
    m = PATTERN_ISO_FLEX.match(line)
    if m:
        service = m.group("service") or m.group("service2") or ""
        return {
            "timestamp": normalize_timestamp(m.group("timestamp")),
            "severity": normalize_severity(m.group("severity")),
            "service": service.strip(),
            "message": m.group("message").strip(),
            "raw": line,
        }

    # Try Pattern 4: Apache/Nginx format
    m = PATTERN_APACHE.match(line)
    if m:
        status_code = int(m.group("status"))
        # Derive severity from HTTP status code
        if status_code >= 500:
            severity = "ERROR"
        elif status_code >= 400:
            severity = "WARNING"
        else:
            severity = "INFO"
        return {
            "timestamp": normalize_timestamp(m.group("timestamp")),
            "severity": severity,
            "service": "httpd",
            "message": f'{m.group("request")} {m.group("status")} {m.group("size")}',
            "raw": line,
        }

    # Try Pattern 3: Syslog format
    m = PATTERN_SYSLOG.match(line)
    if m:
        message = m.group("message").strip()
        # Try to extract severity from the message itself
        sev_match = re.match(r"^(DEBUG|INFO|WARN(?:ING)?|ERR(?:OR)?|CRIT(?:ICAL)?|FATAL|EMERG)\b[:\s]*(.+)", message, re.IGNORECASE)
        if sev_match:
            severity = normalize_severity(sev_match.group(1))
            message = sev_match.group(2).strip()
        else:
            severity = "INFO"
        return {
            "timestamp": normalize_timestamp(m.group("timestamp")),
            "severity": severity,
            "service": m.group("service").strip(),
            "message": message,
            "raw": line,
        }

    # Fallback: treat as raw text with UNKNOWN severity
    # Try to extract any severity keyword from the line
    sev_found = "UNKNOWN"
    for keyword in ["FATAL", "CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG"]:
        if keyword in line.upper():
            sev_found = normalize_severity(keyword)
            break

    # Try to extract any service name from brackets
    service = ""
    bracket_match = re.search(r"\[([a-zA-Z][\w\-\.]*)\]", line)
    if bracket_match:
        service = bracket_match.group(1)

    # Derive service from source filename if not found
    if not service and source_file:
        base = os.path.splitext(os.path.basename(source_file))[0]
        service = base

    return {
        "timestamp": None,
        "severity": sev_found,
        "service": service,
        "message": line.strip(),
        "raw": line,
    }


def collect_log_files(input_path: str) -> Tuple[List[str], Optional[str]]:
    """
    Given an input path, return a list of log file paths to process.
    If the input is a ZIP, extract to a temp directory and return log files within.
    Returns (list_of_file_paths, temp_dir_or_None).
    """
    temp_dir = None

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Handle ZIP archives
    if zipfile.is_zipfile(input_path):
        temp_dir = tempfile.mkdtemp(prefix="log_analyzer_")
        try:
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(temp_dir)
        except zipfile.BadZipFile:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Corrupted ZIP archive: {input_path}")

        # Collect all log files from extracted contents
        log_files = []
        for root, _dirs, files in os.walk(temp_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext in LOG_EXTENSIONS and not is_binary_file(fpath):
                    log_files.append(fpath)
        if not log_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError("ZIP archive contains no readable log files (.log, .txt, .out, .err)")
        return log_files, temp_dir

    # Handle single file
    if os.path.isfile(input_path):
        if is_binary_file(input_path):
            raise ValueError(f"Input file appears to be binary: {input_path}")
        return [input_path], None

    # Handle directory
    if os.path.isdir(input_path):
        log_files = []
        for root, _dirs, files in os.walk(input_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext in LOG_EXTENSIONS and not is_binary_file(fpath):
                    log_files.append(fpath)
        if not log_files:
            raise ValueError(f"Directory contains no readable log files: {input_path}")
        return log_files, None

    raise ValueError(f"Unsupported input type: {input_path}")


def compute_anomaly_detection(timeline_data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    Perform z-score based anomaly detection on per-minute error counts.
    Returns anomaly detection results including suspected incident start time.
    """
    if not timeline_data:
        return {
            "suspected_incident_start": None,
            "anomaly_score": 0.0,
            "baseline_error_rate_per_min": 0.0,
            "peak_error_rate_per_min": 0,
        }

    # Sort buckets chronologically
    sorted_buckets = sorted(timeline_data.keys())
    error_counts = [timeline_data[b].get("error", 0) for b in sorted_buckets]

    if not error_counts or len(error_counts) < 2:
        peak = max(error_counts) if error_counts else 0
        return {
            "suspected_incident_start": sorted_buckets[0] if sorted_buckets else None,
            "anomaly_score": 0.0,
            "baseline_error_rate_per_min": float(np.mean(error_counts)) if error_counts else 0.0,
            "peak_error_rate_per_min": peak,
        }

    error_array = np.array(error_counts, dtype=float)
    mean_val = np.mean(error_array)
    std_val = np.std(error_array)

    # Compute z-scores
    if std_val > 0:
        z_scores = (error_array - mean_val) / std_val
    else:
        z_scores = np.zeros_like(error_array)

    # Find first bucket where z-score exceeds threshold
    suspected_start = None
    max_z = 0.0
    for i, z in enumerate(z_scores):
        if z > max_z:
            max_z = z
        if z > Z_SCORE_THRESHOLD and suspected_start is None:
            suspected_start = sorted_buckets[i]

    return {
        "suspected_incident_start": suspected_start,
        "anomaly_score": round(float(max_z), 2),
        "baseline_error_rate_per_min": round(float(mean_val), 2),
        "peak_error_rate_per_min": int(max(error_counts)),
    }


def classify_severity(
    peak_error_rate: int,
    total_errors: int,
    timeline_data: Dict[str, Dict[str, int]],
    has_fatal: bool,
) -> str:
    """
    Classify overall incident severity based on error rates and patterns.
    - CRITICAL: >1000 errors/min or FATAL errors present
    - HIGH: >100 errors/min or sustained errors >10min
    - MEDIUM: >10 errors/min
    - LOW: <10 errors/min
    """
    if has_fatal or peak_error_rate > 1000:
        return "CRITICAL"

    if peak_error_rate > 100:
        return "HIGH"

    # Check for sustained errors > 10 minutes
    sustained_minutes = sum(
        1 for bucket_data in timeline_data.values() if bucket_data.get("error", 0) > 0
    )
    if sustained_minutes > 10:
        return "HIGH"

    if peak_error_rate > 10:
        return "MEDIUM"

    return "LOW"


def generate_recommendations(
    top_errors: List[Dict[str, Any]],
    anomaly: Dict[str, Any],
    affected_services: List[Dict[str, Any]],
    severity: str,
) -> List[str]:
    """Generate actionable recommendations based on analysis results."""
    recommendations = []

    # Recommendations based on top error patterns
    for err in top_errors[:5]:
        pattern = err["pattern"]
        count = err["count"]

        if "ConnectionTimeout" in pattern or "connection" in pattern.lower() and "timeout" in pattern.lower():
            recommendations.append(
                f"ConnectionTimeout errors ({count} occurrences) suggest network connectivity issues — "
                f"check network routes, firewall rules, and target service health"
            )
        elif "NullPointer" in pattern or "NoneType" in pattern or "null" in pattern.lower():
            recommendations.append(
                f"Null reference errors ({count} occurrences) in {', '.join(err.get('affected_services', []))} — "
                f"review recent code deployments for missing null checks"
            )
        elif "OutOfMemory" in pattern or "MemoryError" in pattern or "heap" in pattern.lower():
            recommendations.append(
                f"Memory errors ({count} occurrences) — consider increasing heap size, "
                f"checking for memory leaks, or scaling horizontally"
            )
        elif "Timeout" in pattern or "timed out" in pattern.lower():
            recommendations.append(
                f"Timeout errors ({count} occurrences) — review service SLAs, "
                f"increase timeout thresholds, or investigate downstream latency"
            )
        elif "Permission" in pattern or "Forbidden" in pattern or "403" in pattern:
            recommendations.append(
                f"Permission/authorization errors ({count} occurrences) — "
                f"verify IAM policies, API keys, and service account permissions"
            )
        elif "disk" in pattern.lower() or "storage" in pattern.lower() or "ENOSPC" in pattern:
            recommendations.append(
                f"Storage-related errors ({count} occurrences) — "
                f"check disk usage, clean up old logs/temp files, or expand storage"
            )
        elif count > 1000:
            recommendations.append(
                f"High-frequency error pattern ({count} occurrences): \"{pattern[:80]}\" — "
                f"investigate root cause urgently"
            )

    # Recommendations based on anomaly detection
    if anomaly.get("suspected_incident_start"):
        recommendations.append(
            f"Incident started at {anomaly['suspected_incident_start']} — "
            f"check infrastructure changes, deployments, or external events around that time"
        )

    # Recommendations based on affected services
    if affected_services:
        top_services = [s["name"] for s in affected_services[:3]]
        recommendations.append(
            f"{', '.join(top_services)} {'are' if len(top_services) > 1 else 'is'} most affected — "
            f"prioritize {'these services' if len(top_services) > 1 else 'this service'} for investigation"
        )

    # Severity-based recommendations
    if severity == "CRITICAL":
        recommendations.append(
            "CRITICAL severity — consider activating incident response procedures and notifying on-call teams"
        )
    elif severity == "HIGH":
        recommendations.append(
            "HIGH severity — escalate to engineering leads and monitor for further degradation"
        )

    # Deduplicate while preserving order
    seen = set()
    unique_recs = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    return unique_recs


# ---------------------------------------------------------------------------
# MCP Server & FastAPI Setup
# ---------------------------------------------------------------------------

mcp = FastMCP("Log Analyzer & Incident Reporter")
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(title="Log Analysis & Incident Report Generator", lifespan=mcp_app.lifespan)


# ---------------------------------------------------------------------------
# Tool: analyze_logs
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_logs(
    input_file: str,
    environment: str = "production",
    time_range: str = None,
    severity_threshold: str = "WARNING",
) -> dict:
    """
    Analyze log files and generate a comprehensive incident report.

    Args:
        input_file: Path to a log file (.log, .txt), directory of log files,
                     or ZIP archive containing log files.
        environment: Deployment environment label (e.g., production, staging, dev).
        time_range: Optional time range filter. Supports ISO interval
                     (e.g., "2026-08-19T10:00:00Z/2026-08-19T11:00:00Z")
                     or relative (e.g., "last_1h", "last_24h", "last_30m").
        severity_threshold: Minimum severity level to include in analysis.
                            One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns:
        dict: Comprehensive incident analysis report with error patterns,
              anomaly detection, affected services, and recommendations.
    """
    temp_dir = None
    try:
        # -----------------------------------------------------------------
        # Step 1: Collect log files
        # -----------------------------------------------------------------
        try:
            log_files, temp_dir = collect_log_files(input_file)
        except FileNotFoundError as e:
            return {"error": str(e), "status": "failed"}
        except ValueError as e:
            return {"error": str(e), "status": "failed"}

        # -----------------------------------------------------------------
        # Step 2: Parse time range filter
        # -----------------------------------------------------------------
        filter_start, filter_end = parse_time_range(time_range) if time_range else (None, None)

        # -----------------------------------------------------------------
        # Step 3: Parse all log lines
        # -----------------------------------------------------------------
        parsed_entries: List[Dict[str, Any]] = []
        parse_failures = 0
        total_lines = 0

        for log_file in log_files:
            try:
                # Attempt to read with UTF-8, fallback to latin-1
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    with open(log_file, "r", encoding="latin-1") as f:
                        lines = f.readlines()

                for line in lines:
                    total_lines += 1
                    entry = parse_log_line(line, source_file=log_file)
                    if entry is None:
                        continue  # Skip empty lines

                    # Track parse failures (no timestamp and unknown severity)
                    if entry["timestamp"] is None and entry["severity"] == "UNKNOWN":
                        parse_failures += 1

                    # Derive service from filename if not extracted from log line
                    if not entry["service"]:
                        base = os.path.splitext(os.path.basename(log_file))[0]
                        entry["service"] = base

                    parsed_entries.append(entry)

            except Exception as e:
                # Skip files that can't be read (permissions, encoding issues)
                parse_failures += 1
                continue

        if not parsed_entries:
            return {
                "error": "No log entries could be parsed from the input",
                "total_lines_scanned": total_lines,
                "parse_failures": parse_failures,
                "status": "failed",
            }

        # -----------------------------------------------------------------
        # Step 4: Apply time range filter
        # -----------------------------------------------------------------
        if filter_start or filter_end:
            filtered = []
            for entry in parsed_entries:
                ts = entry["timestamp"]
                if ts is None:
                    continue  # Skip entries without timestamps when filtering
                if filter_start and ts < filter_start:
                    continue
                if filter_end and ts > filter_end:
                    continue
                filtered.append(entry)
            parsed_entries = filtered

            if not parsed_entries:
                return {
                    "error": "No log entries found within the specified time range",
                    "time_range_filter": time_range,
                    "status": "failed",
                }

        # -----------------------------------------------------------------
        # Step 5: Compute statistics
        # -----------------------------------------------------------------
        severity_counts = Counter()
        service_error_counts = defaultdict(int)
        has_fatal = False

        # Collect timestamps for time range
        timestamps = []

        for entry in parsed_entries:
            sev = entry["severity"]
            severity_counts[sev] += 1

            if sev == "FATAL":
                has_fatal = True

            if sev in ("ERROR", "CRITICAL", "FATAL"):
                service_error_counts[entry["service"]] += 1

            if entry["timestamp"]:
                timestamps.append(entry["timestamp"])

        # Determine time range
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
        else:
            min_ts = None
            max_ts = None

        total_errors = severity_counts.get("ERROR", 0) + severity_counts.get("CRITICAL", 0) + severity_counts.get("FATAL", 0)
        total_warnings = severity_counts.get("WARNING", 0)
        total_info = severity_counts.get("INFO", 0) + severity_counts.get("DEBUG", 0)

        # -----------------------------------------------------------------
        # Step 6: Error pattern grouping
        # -----------------------------------------------------------------
        error_patterns: Dict[str, Dict[str, Any]] = {}

        for entry in parsed_entries:
            if entry["severity"] not in ("ERROR", "CRITICAL", "FATAL"):
                continue

            normalized = normalize_error_pattern(entry["message"])
            if normalized not in error_patterns:
                error_patterns[normalized] = {
                    "pattern": normalized,
                    "count": 0,
                    "first_seen": entry["timestamp"],
                    "last_seen": entry["timestamp"],
                    "affected_services": set(),
                    "sample_message": entry["message"],
                }

            ep = error_patterns[normalized]
            ep["count"] += 1

            if entry["service"]:
                ep["affected_services"].add(entry["service"])

            if entry["timestamp"]:
                if ep["first_seen"] is None or entry["timestamp"] < ep["first_seen"]:
                    ep["first_seen"] = entry["timestamp"]
                if ep["last_seen"] is None or entry["timestamp"] > ep["last_seen"]:
                    ep["last_seen"] = entry["timestamp"]

        # Sort by count descending
        sorted_patterns = sorted(error_patterns.values(), key=lambda x: x["count"], reverse=True)

        # Format top errors for output
        top_errors = []
        for ep in sorted_patterns[:20]:  # Top 20 patterns
            top_errors.append({
                "pattern": ep["pattern"],
                "count": ep["count"],
                "first_seen": ep["first_seen"].isoformat() if ep["first_seen"] else None,
                "last_seen": ep["last_seen"].isoformat() if ep["last_seen"] else None,
                "affected_services": sorted(ep["affected_services"]),
                "sample_message": ep["sample_message"],
            })

        # -----------------------------------------------------------------
        # Step 7: Build per-minute timeline
        # -----------------------------------------------------------------
        timeline_data: Dict[str, Dict[str, int]] = {}

        for entry in parsed_entries:
            ts = entry["timestamp"]
            if ts is None:
                continue
            # Bucket by minute
            bucket = ts.strftime("%Y-%m-%dT%H:%M:00Z")
            if bucket not in timeline_data:
                timeline_data[bucket] = {"total": 0, "error": 0, "warning": 0, "info": 0}

            timeline_data[bucket]["total"] += 1
            sev = entry["severity"]
            if sev in ("ERROR", "CRITICAL", "FATAL"):
                timeline_data[bucket]["error"] += 1
            elif sev == "WARNING":
                timeline_data[bucket]["warning"] += 1
            elif sev in ("INFO", "DEBUG"):
                timeline_data[bucket]["info"] += 1

        # -----------------------------------------------------------------
        # Step 8: Anomaly detection
        # -----------------------------------------------------------------
        anomaly = compute_anomaly_detection(timeline_data)

        # -----------------------------------------------------------------
        # Step 9: Severity classification
        # -----------------------------------------------------------------
        overall_severity = classify_severity(
            peak_error_rate=anomaly["peak_error_rate_per_min"],
            total_errors=total_errors,
            timeline_data=timeline_data,
            has_fatal=has_fatal,
        )

        # -----------------------------------------------------------------
        # Step 10: Affected services
        # -----------------------------------------------------------------
        affected_services = [
            {"name": svc, "error_count": cnt}
            for svc, cnt in sorted(service_error_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # -----------------------------------------------------------------
        # Step 11: Generate recommendations
        # -----------------------------------------------------------------
        recommendations = generate_recommendations(top_errors, anomaly, affected_services, overall_severity)

        # -----------------------------------------------------------------
        # Step 12: Write output files
        # -----------------------------------------------------------------
        # Determine output directory (same as input file's directory)
        if os.path.isfile(input_file):
            output_dir = os.path.dirname(os.path.abspath(input_file))
        elif os.path.isdir(input_file):
            output_dir = os.path.abspath(input_file)
        else:
            output_dir = os.path.dirname(os.path.abspath(input_file))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- Error Summary CSV ---
        error_summary_path = os.path.join(output_dir, "error_summary.csv")
        try:
            with open(error_summary_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["error_pattern", "count", "first_seen", "last_seen", "affected_services"])
                for ep in sorted_patterns:
                    writer.writerow([
                        ep["pattern"],
                        ep["count"],
                        ep["first_seen"].isoformat() if ep["first_seen"] else "",
                        ep["last_seen"].isoformat() if ep["last_seen"] else "",
                        ";".join(sorted(ep["affected_services"])),
                    ])
        except Exception as e:
            error_summary_path = f"Error writing error_summary.csv: {e}"

        # --- Timeline CSV ---
        timeline_path = os.path.join(output_dir, "timeline.csv")
        try:
            with open(timeline_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp_bucket", "total_count", "error_count", "warning_count", "info_count"])
                for bucket in sorted(timeline_data.keys()):
                    d = timeline_data[bucket]
                    writer.writerow([bucket, d["total"], d["error"], d["warning"], d["info"]])
        except Exception as e:
            timeline_path = f"Error writing timeline.csv: {e}"

        # --- Incident Report JSON ---
        report = {
            "environment": environment,
            "time_range": {
                "start": min_ts.isoformat() if min_ts else None,
                "end": max_ts.isoformat() if max_ts else None,
            },
            "severity": overall_severity,
            "summary": {
                "total_lines_parsed": total_lines,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "total_info": total_info,
                "parse_failures": parse_failures,
            },
            "top_errors": top_errors,
            "anomaly_detection": anomaly,
            "affected_services": affected_services,
            "recommendations": recommendations,
            "output_files": {
                "incident_report": os.path.basename(os.path.join(output_dir, "incident_report.json")),
                "error_summary": os.path.basename(error_summary_path) if not error_summary_path.startswith("Error") else error_summary_path,
                "timeline": os.path.basename(timeline_path) if not timeline_path.startswith("Error") else timeline_path,
            },
        }

        report_path = os.path.join(output_dir, "incident_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            report["output_files"]["incident_report"] = f"Error writing incident_report.json: {e}"

        return report

    except Exception as e:
        return {
            "error": f"Unexpected error during log analysis: {str(e)}",
            "status": "failed",
        }
    finally:
        # Clean up temp directory if we extracted a ZIP
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tool: compare_incidents
# ---------------------------------------------------------------------------


@mcp.tool()
async def compare_incidents(
    incident_a: str,
    incident_b: str,
) -> dict:
    """
    Compare two incident reports for pattern similarity.

    Args:
        incident_a: Path to the first incident report JSON file.
        incident_b: Path to the second incident report JSON file.

    Returns:
        dict: Comparison report with common patterns, unique patterns,
              similarity score, and assessment.
    """
    try:
        # -----------------------------------------------------------------
        # Step 1: Load both incident reports
        # -----------------------------------------------------------------
        for path, label in [(incident_a, "incident_a"), (incident_b, "incident_b")]:
            if not os.path.exists(path):
                return {"error": f"File not found: {path} ({label})", "status": "failed"}

        try:
            with open(incident_a, "r", encoding="utf-8") as f:
                report_a = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"error": f"Failed to parse incident_a ({incident_a}): {str(e)}", "status": "failed"}

        try:
            with open(incident_b, "r", encoding="utf-8") as f:
                report_b = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"error": f"Failed to parse incident_b ({incident_b}): {str(e)}", "status": "failed"}

        # -----------------------------------------------------------------
        # Step 2: Extract error patterns and services
        # -----------------------------------------------------------------
        patterns_a = {e["pattern"] for e in report_a.get("top_errors", [])}
        patterns_b = {e["pattern"] for e in report_b.get("top_errors", [])}

        services_a = {s["name"] for s in report_a.get("affected_services", [])}
        services_b = {s["name"] for s in report_b.get("affected_services", [])}

        # -----------------------------------------------------------------
        # Step 3: Compute similarity
        # -----------------------------------------------------------------
        # Pattern similarity (Jaccard index)
        common_patterns = patterns_a & patterns_b
        all_patterns = patterns_a | patterns_b
        pattern_similarity = (len(common_patterns) / len(all_patterns) * 100) if all_patterns else 0.0

        # Service similarity (Jaccard index)
        common_services = services_a & services_b
        all_services = services_a | services_b
        service_similarity = (len(common_services) / len(all_services) * 100) if all_services else 0.0

        # Severity comparison
        severity_a = report_a.get("severity", "UNKNOWN")
        severity_b = report_b.get("severity", "UNKNOWN")
        same_severity = severity_a == severity_b

        # Overall similarity score (weighted)
        # 60% pattern similarity + 30% service similarity + 10% severity match
        similarity_score = round(
            pattern_similarity * 0.6 + service_similarity * 0.3 + (100 if same_severity else 0) * 0.1,
            1,
        )

        # -----------------------------------------------------------------
        # Step 4: Build comparison details
        # -----------------------------------------------------------------
        # Get pattern details with counts for common patterns
        common_pattern_details = []
        for pattern in common_patterns:
            count_a = next((e["count"] for e in report_a.get("top_errors", []) if e["pattern"] == pattern), 0)
            count_b = next((e["count"] for e in report_b.get("top_errors", []) if e["pattern"] == pattern), 0)
            common_pattern_details.append({
                "pattern": pattern,
                "count_in_a": count_a,
                "count_in_b": count_b,
            })

        unique_to_a = []
        for pattern in (patterns_a - patterns_b):
            count = next((e["count"] for e in report_a.get("top_errors", []) if e["pattern"] == pattern), 0)
            unique_to_a.append({"pattern": pattern, "count": count})

        unique_to_b = []
        for pattern in (patterns_b - patterns_a):
            count = next((e["count"] for e in report_b.get("top_errors", []) if e["pattern"] == pattern), 0)
            unique_to_b.append({"pattern": pattern, "count": count})

        # -----------------------------------------------------------------
        # Step 5: Generate assessment
        # -----------------------------------------------------------------
        if similarity_score >= 80:
            assessment = (
                "HIGH SIMILARITY — These incidents are very likely related or caused by the same root issue. "
                "Consider investigating them as a single recurring problem."
            )
        elif similarity_score >= 50:
            assessment = (
                "MODERATE SIMILARITY — These incidents share some common patterns and may be partially related. "
                "Investigate shared error patterns for potential common root causes."
            )
        elif similarity_score >= 20:
            assessment = (
                "LOW SIMILARITY — These incidents have limited overlap. They are likely separate issues "
                "but may share some infrastructure dependencies."
            )
        else:
            assessment = (
                "MINIMAL SIMILARITY — These incidents appear to be unrelated. "
                "Investigate them independently."
            )

        # -----------------------------------------------------------------
        # Step 6: Build comparison report
        # -----------------------------------------------------------------
        comparison = {
            "similarity_score": similarity_score,
            "assessment": assessment,
            "incident_a": {
                "file": incident_a,
                "environment": report_a.get("environment", "unknown"),
                "time_range": report_a.get("time_range", {}),
                "severity": severity_a,
                "total_errors": report_a.get("summary", {}).get("total_errors", 0),
                "num_error_patterns": len(patterns_a),
                "num_affected_services": len(services_a),
            },
            "incident_b": {
                "file": incident_b,
                "environment": report_b.get("environment", "unknown"),
                "time_range": report_b.get("time_range", {}),
                "severity": severity_b,
                "total_errors": report_b.get("summary", {}).get("total_errors", 0),
                "num_error_patterns": len(patterns_b),
                "num_affected_services": len(services_b),
            },
            "pattern_comparison": {
                "common_patterns": common_pattern_details,
                "unique_to_a": unique_to_a,
                "unique_to_b": unique_to_b,
                "pattern_similarity_pct": round(pattern_similarity, 1),
            },
            "service_comparison": {
                "common_services": sorted(common_services),
                "unique_to_a": sorted(services_a - services_b),
                "unique_to_b": sorted(services_b - services_a),
                "service_similarity_pct": round(service_similarity, 1),
            },
            "severity_comparison": {
                "incident_a": severity_a,
                "incident_b": severity_b,
                "same_severity": same_severity,
            },
        }

        return comparison

    except Exception as e:
        return {
            "error": f"Unexpected error during incident comparison: {str(e)}",
            "status": "failed",
        }


# ---------------------------------------------------------------------------
# Mount MCP server as ASGI sub-application
# ---------------------------------------------------------------------------

app.mount("/mcp", mcp_app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
