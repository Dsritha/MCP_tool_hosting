"""
PDF Invoice → Structured Financial Data Extractor
===================================================
INPUT:  A PDF invoice file (e.g., invoice.pdf)
OUTPUT: Structured financial data extracted from the invoice, including:
        - Invoice number, date, due date
        - Vendor and buyer information (name, address, tax ID)
        - Currency, subtotal, tax, total
        - Line items (description, quantity, unit price, total)
        - Validation report (totals reconciliation, tax checks, field completeness)
        - Output in JSON format, with optional Excel (.xlsx) export

PURPOSE: Parses PDF invoices using text extraction, identifies key financial fields
         via pattern matching and heuristics, validates arithmetic consistency
         (subtotal + tax = total, line items sum = subtotal), and produces
         enterprise-grade structured output for accounting and ERP integration.
"""

# =============================================================================
# Standard library imports
# =============================================================================
import os
import re
import json
import logging
from datetime import datetime, date
from typing import Optional
import shutil
from fastapi import UploadFile, File

# Add this endpoint directly below your app = FastAPI(...) initialization


# =============================================================================
# Third-party imports
# =============================================================================
import pdfplumber
import pandas as pd
from dateutil import parser as dateutil_parser
from fastapi import FastAPI
from fastmcp import FastMCP

# =============================================================================
# Logging configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pdf_invoice_extractor")

# =============================================================================
# In-memory duplicate invoice registry (session-scoped)
# =============================================================================
_processed_invoice_numbers: set = set()

# =============================================================================
# MCP + FastAPI setup
# =============================================================================
mcp = FastMCP("PDF Invoice Extractor")
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(title="PDF Invoice → Structured Financial Data", lifespan=mcp_app.lifespan)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accepts local files, saves them to Render's temporary disk, and returns the path."""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"server_file_path": temp_path}

# =============================================================================
# Constants — regex patterns for field extraction
# =============================================================================

# Invoice number patterns (e.g., INV-2026-00921, Invoice #12345, Invoice No. 12345)
INVOICE_NUMBER_PATTERNS = [
    re.compile(r"(?:Invoice\s*#|Invoice\s*No\.?\s*|INV[-–])\s*([A-Za-z0-9\-]+)", re.IGNORECASE),
    re.compile(r"(?:Bill\s*#|Bill\s*No\.?\s*)\s*([A-Za-z0-9\-]+)", re.IGNORECASE),
    re.compile(r"(?:Receipt\s*#|Receipt\s*No\.?\s*)\s*([A-Za-z0-9\-]+)", re.IGNORECASE),
    re.compile(r"\b(INV[-–]\d{4}[-–]\d{3,})\b", re.IGNORECASE),
    re.compile(r"(?:Invoice\s*Number\s*[:\-]?\s*)([A-Za-z0-9\-]+)", re.IGNORECASE),
]

# Date patterns — multiple common formats
DATE_PATTERNS = [
    # ISO: 2026-08-14
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    # US: 08/14/2026 or 08-14-2026
    re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"),
    # Verbose: 14 Aug 2026, August 14, 2026
    re.compile(
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

# Date label patterns — to associate dates with their roles
INVOICE_DATE_LABELS = [
    re.compile(r"(?:Invoice\s*Date|Date\s*of\s*Invoice|Issue\s*Date|Issued\s*On|Date)\s*[:\-]?\s*", re.IGNORECASE),
]
DUE_DATE_LABELS = [
    re.compile(r"(?:Due\s*Date|Payment\s*Due|Pay\s*By|Due\s*On|Due)\s*[:\-]?\s*", re.IGNORECASE),
]

# Currency symbols and ISO codes
CURRENCY_SYMBOLS = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
    "₹": "INR", "₩": "KRW", "₽": "RUB", "R$": "BRL",
    "CHF": "CHF", "A$": "AUD", "C$": "CAD", "kr": "SEK",
}
CURRENCY_ISO_PATTERN = re.compile(r"\b(USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|SEK|NOK|DKK|BRL|KRW|RUB|CNY|HKD|SGD|NZD|MXN|ZAR)\b", re.IGNORECASE)

# Amount patterns
SUBTOTAL_PATTERNS = [
    re.compile(r"(?:Sub\s*[-]?\s*Total|Subtotal)\s*[:\-]?\s*[^\d]*([\d,]+\.?\d*)", re.IGNORECASE),
]
TAX_PATTERNS = [
    re.compile(r"(?:Tax|VAT|GST|Sales\s*Tax|HST)\s*(?:\(?\d+\.?\d*%?\)?)?\s*[:\-]?\s*[^\d]*([\d,]+\.?\d*)", re.IGNORECASE),
]
TOTAL_PATTERNS = [
    re.compile(r"(?:Grand\s*Total|Total\s*Due|Total\s*Amount|Amount\s*Due|Balance\s*Due|Total)\s*[:\-]?\s*[^\d]*([\d,]+\.?\d*)", re.IGNORECASE),
]

# Tax ID / VAT patterns
TAX_ID_PATTERNS = [
    re.compile(r"(?:Tax\s*ID|TIN|VAT\s*(?:No|Number|ID|Reg)|EIN|ABN|GST\s*(?:No|Number|Reg))\s*[:\-]?\s*([A-Za-z0-9\-\.]+)", re.IGNORECASE),
    re.compile(r"\b([A-Z]{2}[-]?\d{2}[-]?\d{7,})\b"),
]


# =============================================================================
# Helper functions
# =============================================================================

def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file using pdfplumber.
    Falls back to empty string if the PDF is scanned-only (image-based).
    """
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    except Exception as exc:
        logger.error("Failed to extract text from PDF '%s': %s", file_path, exc)
        raise ValueError(f"Could not read PDF file: {exc}") from exc

    if not full_text.strip():
        logger.warning("No extractable text found in '%s'. The PDF may be scanned/image-only.", file_path)

    return full_text


def _extract_tables_from_pdf(file_path: str) -> list[list[list[Optional[str]]]]:
    """
    Extract tables from each page of the PDF using pdfplumber's table extraction.
    Returns a list of tables, where each table is a list of rows (list of cell strings).
    """
    all_tables: list[list[list[Optional[str]]]] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
    except Exception as exc:
        logger.warning("Table extraction failed for '%s': %s", file_path, exc)
    return all_tables


def _find_first_match(text: str, patterns: list[re.Pattern]) -> Optional[str]:
    """Return the first regex group match from a list of compiled patterns."""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _parse_date_safe(date_str: str) -> Optional[str]:
    """
    Attempt to parse a date string into ISO format (YYYY-MM-DD).
    Returns None if parsing fails.
    """
    if not date_str:
        return None
    try:
        parsed = dateutil_parser.parse(date_str, dayfirst=False, fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def _extract_labeled_date(text: str, label_patterns: list[re.Pattern]) -> Optional[str]:
    """
    Find a date that appears immediately after a label (e.g., 'Invoice Date:').
    """
    for label_pat in label_patterns:
        match = label_pat.search(text)
        if match:
            # Grab the text after the label and try to find a date in it
            after_label = text[match.end(): match.end() + 60]
            for date_pat in DATE_PATTERNS:
                date_match = date_pat.search(after_label)
                if date_match:
                    return _parse_date_safe(date_match.group(1))
    return None


def _extract_invoice_number(text: str) -> Optional[str]:
    """Extract invoice number from text using multiple patterns."""
    return _find_first_match(text, INVOICE_NUMBER_PATTERNS)


def _extract_dates(text: str) -> dict:
    """Extract invoice date and due date from text."""
    invoice_date = _extract_labeled_date(text, INVOICE_DATE_LABELS)
    due_date = _extract_labeled_date(text, DUE_DATE_LABELS)

    # Fallback: if no labeled invoice date found, grab the first date in the document
    if not invoice_date:
        for pat in DATE_PATTERNS:
            match = pat.search(text)
            if match:
                invoice_date = _parse_date_safe(match.group(1))
                if invoice_date:
                    break

    return {"invoice_date": invoice_date, "due_date": due_date}


def _detect_currency(text: str) -> str:
    """Detect currency from symbols or ISO codes in the text."""
    # Check ISO codes first
    iso_match = CURRENCY_ISO_PATTERN.search(text)
    if iso_match:
        return iso_match.group(1).upper()

    # Check currency symbols
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code

    return "USD"  # Default fallback


def _parse_amount(raw: Optional[str]) -> Optional[float]:
    """Parse a raw amount string (e.g., '12,000.00') into a float."""
    if not raw:
        return None
    try:
        cleaned = raw.replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_amounts(text: str) -> dict:
    """Extract subtotal, tax, and total amounts from text."""
    subtotal_raw = _find_first_match(text, SUBTOTAL_PATTERNS)
    tax_raw = _find_first_match(text, TAX_PATTERNS)
    total_raw = _find_first_match(text, TOTAL_PATTERNS)

    subtotal = _parse_amount(subtotal_raw)
    tax = _parse_amount(tax_raw)
    total = _parse_amount(total_raw)

    # Attempt to infer missing values
    if subtotal is not None and tax is not None and total is None:
        total = round(subtotal + tax, 2)
    elif total is not None and tax is not None and subtotal is None:
        subtotal = round(total - tax, 2)
    elif total is not None and subtotal is not None and tax is None:
        tax = round(total - subtotal, 2)

    return {"subtotal": subtotal, "tax": tax, "total": total}


def _extract_tax_id(text: str) -> Optional[str]:
    """Extract tax ID / VAT number from text."""
    return _find_first_match(text, TAX_ID_PATTERNS)


def _extract_vendor_buyer(text: str) -> dict:
    """
    Extract vendor and buyer information from the invoice text.
    Uses heuristic section detection based on common invoice layouts.
    """
    vendor = {"name": None, "address": None, "tax_id": None}
    buyer = {"name": None, "address": None}

    lines = text.split("\n")

    # --- Vendor extraction ---
    # Look for "From:", "Seller:", "Vendor:", "Bill From:", "Sold By:" sections
    vendor_section_patterns = [
        re.compile(r"(?:From|Seller|Vendor|Bill\s*From|Sold\s*By|Supplier)\s*[:\-]?\s*", re.IGNORECASE),
    ]
    # --- Buyer extraction ---
    buyer_section_patterns = [
        re.compile(r"(?:To|Buyer|Bill\s*To|Sold\s*To|Ship\s*To|Customer|Client)\s*[:\-]?\s*", re.IGNORECASE),
    ]

    def _extract_section_info(section_patterns: list[re.Pattern], max_lines: int = 5) -> dict:
        """Extract name and address from a labeled section."""
        info = {"name": None, "address": None}
        for i, line in enumerate(lines):
            for pat in section_patterns:
                match = pat.search(line)
                if match:
                    # The name might be on the same line after the label, or the next line
                    remainder = line[match.end():].strip()
                    section_lines = []
                    if remainder:
                        section_lines.append(remainder)
                    # Collect subsequent lines (up to max_lines)
                    for j in range(i + 1, min(i + 1 + max_lines, len(lines))):
                        next_line = lines[j].strip()
                        # Stop if we hit another section label or empty line
                        if not next_line:
                            break
                        # Stop if we hit a known label
                        if re.match(r"(?:From|To|Seller|Buyer|Bill|Sold|Ship|Invoice|Date|Due|Tax|Sub|Total|Item|Qty|Description)", next_line, re.IGNORECASE):
                            break
                        section_lines.append(next_line)

                    if section_lines:
                        info["name"] = section_lines[0]
                        if len(section_lines) > 1:
                            info["address"] = ", ".join(section_lines[1:])
                    return info
        return info

    vendor_info = _extract_section_info(vendor_section_patterns)
    buyer_info = _extract_section_info(buyer_section_patterns)

    vendor["name"] = vendor_info.get("name")
    vendor["address"] = vendor_info.get("address")
    vendor["tax_id"] = _extract_tax_id(text)

    buyer["name"] = buyer_info.get("name")
    buyer["address"] = buyer_info.get("address")

    # Fallback: if vendor name not found, use the first non-empty line as vendor name
    if not vendor["name"]:
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3 and not re.match(r"^\d", stripped):
                vendor["name"] = stripped
                break

    return {"vendor": vendor, "buyer": buyer}


def _extract_line_items(text: str, tables: list[list[list[Optional[str]]]]) -> list[dict]:
    """
    Extract line items from PDF tables or from text using heuristics.
    Each line item should have: description, quantity, unit_price, total.
    """
    line_items: list[dict] = []

    # --- Strategy 1: Use extracted tables ---
    if tables:
        for table in tables:
            if not table or len(table) < 2:
                continue

            # Try to identify header row
            header_row = table[0]
            if header_row is None:
                continue

            # Normalize headers
            headers = [str(h).strip().lower() if h else "" for h in header_row]

            # Map column indices
            desc_idx = _find_column_index(headers, ["description", "item", "product", "service", "particulars", "detail"])
            qty_idx = _find_column_index(headers, ["qty", "quantity", "units", "count", "no"])
            price_idx = _find_column_index(headers, ["unit price", "price", "rate", "unit cost", "unit_price"])
            total_idx = _find_column_index(headers, ["total", "amount", "line total", "ext", "extended", "line_total"])

            if desc_idx is None and total_idx is None:
                continue  # Not a line-item table

            # Parse data rows
            for row in table[1:]:
                if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                    continue

                item = {
                    "description": _safe_cell(row, desc_idx),
                    "quantity": _parse_amount(_safe_cell(row, qty_idx)),
                    "unit_price": _parse_amount(_safe_cell(row, price_idx)),
                    "total": _parse_amount(_safe_cell(row, total_idx)),
                }

                # Skip rows that look like subtotal/total summary rows
                desc_lower = (item["description"] or "").lower()
                if any(kw in desc_lower for kw in ["subtotal", "sub total", "total", "tax", "vat", "gst", "grand total", "discount"]):
                    continue

                # Infer missing values where possible
                if item["quantity"] is not None and item["unit_price"] is not None and item["total"] is None:
                    item["total"] = round(item["quantity"] * item["unit_price"], 2)
                elif item["total"] is not None and item["quantity"] is not None and item["unit_price"] is None and item["quantity"] != 0:
                    item["unit_price"] = round(item["total"] / item["quantity"], 2)

                # Only add if we have at least a description or a total
                if item["description"] or item["total"] is not None:
                    line_items.append(item)

            if line_items:
                return line_items  # Use the first valid table

    # --- Strategy 2: Regex-based line item extraction from text ---
    # Pattern: description followed by quantity, unit price, total on the same line
    line_item_pattern = re.compile(
        r"^(.+?)\s+(\d+(?:\.\d+)?)\s+[\$€£¥₹]?([\d,]+\.?\d*)\s+[\$€£¥₹]?([\d,]+\.?\d*)\s*$",
        re.MULTILINE,
    )
    for match in line_item_pattern.finditer(text):
        desc = match.group(1).strip()
        # Skip summary rows
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in ["subtotal", "sub total", "total", "tax", "vat", "gst", "grand total", "discount"]):
            continue
        qty = _parse_amount(match.group(2))
        unit_price = _parse_amount(match.group(3))
        total = _parse_amount(match.group(4))
        line_items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit_price,
            "total": total,
        })

    return line_items


def _find_column_index(headers: list[str], candidates: list[str]) -> Optional[int]:
    """Find the index of a column header matching any of the candidate names."""
    for i, header in enumerate(headers):
        for candidate in candidates:
            if candidate in header:
                return i
    return None


def _safe_cell(row: list, idx: Optional[int]) -> Optional[str]:
    """Safely retrieve a cell value from a row by index."""
    if idx is None or idx >= len(row):
        return None
    val = row[idx]
    if val is None:
        return None
    return str(val).strip()


# =============================================================================
# Validation functions
# =============================================================================

def _validate_invoice(invoice_data: dict, line_items: list[dict]) -> dict:
    """
    Run all validation checks on the extracted invoice data.
    Returns a validation report with pass/fail status for each check.
    """
    checks = {}
    errors = []
    warnings = []

    # ---- 1. Invoice number present ----
    inv_num = invoice_data.get("invoice_number")
    if inv_num:
        checks["invoice_number_present"] = {"status": "passed"}
    else:
        checks["invoice_number_present"] = {"status": "failed", "detail": "Invoice number not found"}
        errors.append("Invoice number could not be extracted")

    # ---- 2. Date validity ----
    inv_date_str = invoice_data.get("invoice_date")
    if inv_date_str:
        try:
            inv_date = datetime.strptime(inv_date_str, "%Y-%m-%d").date()
            if inv_date > date.today():
                checks["date_valid"] = {"status": "warning", "detail": f"Invoice date {inv_date_str} is in the future"}
                warnings.append(f"Invoice date {inv_date_str} is in the future")
            else:
                checks["date_valid"] = {"status": "passed"}
        except ValueError:
            checks["date_valid"] = {"status": "failed", "detail": f"Could not parse date: {inv_date_str}"}
            errors.append(f"Invalid invoice date format: {inv_date_str}")
    else:
        checks["date_valid"] = {"status": "failed", "detail": "Invoice date not found"}
        errors.append("Invoice date could not be extracted")

    # ---- 3. Vendor present ----
    vendor_name = invoice_data.get("vendor", {}).get("name")
    if vendor_name:
        checks["vendor_present"] = {"status": "passed"}
    else:
        checks["vendor_present"] = {"status": "failed", "detail": "Vendor name not found"}
        warnings.append("Vendor name could not be extracted")

    # ---- 4. Totals reconciliation ----
    subtotal = invoice_data.get("subtotal")
    tax = invoice_data.get("tax")
    total = invoice_data.get("total")

    if subtotal is not None and tax is not None and total is not None:
        expected_total = round(subtotal + tax, 2)
        tolerance = 0.02  # Allow 2 cents tolerance for rounding
        if abs(expected_total - total) <= tolerance:
            checks["totals_reconcile"] = {
                "status": "passed",
                "detail": f"subtotal({subtotal}) + tax({tax}) = total({total})",
            }
        else:
            checks["totals_reconcile"] = {
                "status": "failed",
                "detail": f"subtotal({subtotal}) + tax({tax}) = {expected_total}, but total is {total}",
            }
            errors.append(f"Totals do not reconcile: {subtotal} + {tax} = {expected_total} ≠ {total}")
    else:
        missing = []
        if subtotal is None:
            missing.append("subtotal")
        if tax is None:
            missing.append("tax")
        if total is None:
            missing.append("total")
        checks["totals_reconcile"] = {
            "status": "skipped",
            "detail": f"Missing values: {', '.join(missing)}",
        }
        warnings.append(f"Cannot verify totals — missing: {', '.join(missing)}")

    # ---- 5. Line items reconciliation ----
    if line_items and subtotal is not None:
        line_items_sum = round(sum(item.get("total", 0) or 0 for item in line_items), 2)
        tolerance = 0.05  # Allow small rounding tolerance
        if abs(line_items_sum - subtotal) <= tolerance:
            checks["line_items_reconcile"] = {
                "status": "passed",
                "detail": f"sum of line items({line_items_sum}) = subtotal({subtotal})",
            }
        else:
            checks["line_items_reconcile"] = {
                "status": "failed",
                "detail": f"sum of line items({line_items_sum}) ≠ subtotal({subtotal})",
            }
            warnings.append(f"Line items sum ({line_items_sum}) does not match subtotal ({subtotal})")
    elif not line_items:
        checks["line_items_reconcile"] = {
            "status": "skipped",
            "detail": "No line items extracted",
        }
    else:
        checks["line_items_reconcile"] = {
            "status": "skipped",
            "detail": "Subtotal not available for comparison",
        }

    # ---- 6. Tax plausibility ----
    if tax is not None and subtotal is not None and subtotal > 0:
        tax_rate = round((tax / subtotal) * 100, 2)
        if 0 <= tax_rate <= 30:
            checks["tax_plausible"] = {
                "status": "passed",
                "detail": f"tax rate {tax_rate}% is within 0-30% range",
            }
        else:
            checks["tax_plausible"] = {
                "status": "warning",
                "detail": f"tax rate {tax_rate}% is outside typical 0-30% range",
            }
            warnings.append(f"Tax rate {tax_rate}% seems unusual (outside 0-30% range)")
    else:
        checks["tax_plausible"] = {
            "status": "skipped",
            "detail": "Tax or subtotal not available",
        }

    # ---- 7. Duplicate invoice check ----
    if inv_num:
        if inv_num in _processed_invoice_numbers:
            checks["duplicate_check"] = {
                "status": "warning",
                "detail": f"Invoice number '{inv_num}' has been seen before in this session",
            }
            warnings.append(f"Possible duplicate: invoice '{inv_num}' was already processed")
        else:
            checks["duplicate_check"] = {
                "status": "passed",
                "detail": "No duplicate invoice numbers found",
            }
            _processed_invoice_numbers.add(inv_num)
    else:
        checks["duplicate_check"] = {
            "status": "skipped",
            "detail": "No invoice number to check",
        }

    # ---- 8. Field completeness ----
    required_fields = {
        "invoice_number": inv_num,
        "invoice_date": inv_date_str,
        "vendor_name": vendor_name,
        "total": total,
    }
    missing_fields = [k for k, v in required_fields.items() if not v]
    if not missing_fields:
        checks["field_completeness"] = {"status": "passed", "detail": "All required fields present"}
    else:
        checks["field_completeness"] = {
            "status": "failed",
            "detail": f"Missing required fields: {', '.join(missing_fields)}",
        }
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")

    # Determine overall status
    statuses = [c["status"] for c in checks.values()]
    if "failed" in statuses:
        overall = "failed"
    elif "warning" in statuses:
        overall = "passed_with_warnings"
    else:
        overall = "passed"

    return {
        "overall_status": overall,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


# =============================================================================
# Excel export
# =============================================================================

def _export_to_excel(invoice_data: dict, line_items: list[dict], output_path: str) -> str:
    """
    Export invoice data and line items to an Excel file with separate sheets.
    Returns the path to the generated Excel file.
    """
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: Invoice Summary
            summary_records = []
            flat_fields = [
                ("Invoice Number", invoice_data.get("invoice_number")),
                ("Invoice Date", invoice_data.get("invoice_date")),
                ("Due Date", invoice_data.get("due_date")),
                ("Vendor Name", invoice_data.get("vendor", {}).get("name")),
                ("Vendor Address", invoice_data.get("vendor", {}).get("address")),
                ("Vendor Tax ID", invoice_data.get("vendor", {}).get("tax_id")),
                ("Buyer Name", invoice_data.get("buyer", {}).get("name")),
                ("Buyer Address", invoice_data.get("buyer", {}).get("address")),
                ("Currency", invoice_data.get("currency")),
                ("Subtotal", invoice_data.get("subtotal")),
                ("Tax", invoice_data.get("tax")),
                ("Tax Rate (%)", invoice_data.get("tax_rate_percent")),
                ("Total", invoice_data.get("total")),
            ]
            for field_name, field_value in flat_fields:
                summary_records.append({"Field": field_name, "Value": field_value})

            df_summary = pd.DataFrame(summary_records)
            df_summary.to_excel(writer, sheet_name="Invoice Summary", index=False)

            # Sheet 2: Line Items
            if line_items:
                df_items = pd.DataFrame(line_items)
                df_items.to_excel(writer, sheet_name="Line Items", index=False)
            else:
                # Write an empty sheet with headers
                df_empty = pd.DataFrame(columns=["description", "quantity", "unit_price", "total"])
                df_empty.to_excel(writer, sheet_name="Line Items", index=False)

        logger.info("Excel file exported to '%s'", output_path)
        return output_path
    except Exception as exc:
        logger.error("Failed to export Excel file: %s", exc)
        raise


# =============================================================================
# Main MCP Tool
# =============================================================================

@mcp.tool()
async def process_invoice(
    input_file: str,
    extract_line_items: bool = True,
    validate_totals: bool = True,
    output_format: str = "json",
) -> dict:
    """
    Process a PDF invoice and extract structured financial data.

    Args:
        input_file: Path to the PDF invoice file.
        extract_line_items: Whether to extract individual line items from the invoice.
        validate_totals: Whether to run validation checks (totals reconciliation, tax plausibility, etc.).
        output_format: Output format — "json" for JSON only, "xlsx" for JSON + Excel export.

    Returns:
        A dictionary containing extracted invoice data, validation report, and output file paths.
    """
    result_errors: list[str] = []
    result_warnings: list[str] = []
    output_files: list[str] = []

    # -------------------------------------------------------------------------
    # Step 0: Validate input file
    # -------------------------------------------------------------------------
    if not input_file:
        return {
            "status": "error",
            "invoice": {},
            "validation_report": {},
            "errors": ["No input file specified"],
            "warnings": [],
            "output_files": [],
        }

    if not os.path.isfile(input_file):
        return {
            "status": "error",
            "invoice": {},
            "validation_report": {},
            "errors": [f"File not found: {input_file}"],
            "warnings": [],
            "output_files": [],
        }

    if not input_file.lower().endswith(".pdf"):
        result_warnings.append("File does not have a .pdf extension — attempting to process anyway")

    # -------------------------------------------------------------------------
    # Step 1: Extract text from PDF
    # -------------------------------------------------------------------------
    try:
        full_text = _extract_text_from_pdf(input_file)
    except ValueError as exc:
        return {
            "status": "error",
            "invoice": {},
            "validation_report": {},
            "errors": [str(exc)],
            "warnings": [],
            "output_files": [],
        }

    if not full_text.strip():
        return {
            "status": "error",
            "invoice": {},
            "validation_report": {},
            "errors": ["No text could be extracted from the PDF. It may be a scanned/image-only document."],
            "warnings": ["Consider using OCR (e.g., Tesseract) for scanned PDFs"],
            "output_files": [],
        }

    # -------------------------------------------------------------------------
    # Step 2: Extract tables (for line items)
    # -------------------------------------------------------------------------
    tables = _extract_tables_from_pdf(input_file) if extract_line_items else []

    # -------------------------------------------------------------------------
    # Step 3: Extract invoice fields
    # -------------------------------------------------------------------------
    invoice_number = _extract_invoice_number(full_text)
    dates = _extract_dates(full_text)
    currency = _detect_currency(full_text)
    amounts = _extract_amounts(full_text)
    vendor_buyer = _extract_vendor_buyer(full_text)

    # Calculate tax rate
    tax_rate_percent = None
    if amounts["tax"] is not None and amounts["subtotal"] is not None and amounts["subtotal"] > 0:
        tax_rate_percent = round((amounts["tax"] / amounts["subtotal"]) * 100, 2)

    # -------------------------------------------------------------------------
    # Step 4: Extract line items
    # -------------------------------------------------------------------------
    line_items: list[dict] = []
    if extract_line_items:
        line_items = _extract_line_items(full_text, tables)
        if not line_items:
            result_warnings.append("No line items could be extracted from the invoice")

    # -------------------------------------------------------------------------
    # Step 5: Build invoice data structure
    # -------------------------------------------------------------------------
    invoice_data = {
        "invoice_number": invoice_number,
        "invoice_date": dates["invoice_date"],
        "due_date": dates["due_date"],
        "vendor": vendor_buyer["vendor"],
        "buyer": vendor_buyer["buyer"],
        "currency": currency,
        "subtotal": amounts["subtotal"],
        "tax": amounts["tax"],
        "tax_rate_percent": tax_rate_percent,
        "total": amounts["total"],
        "line_items": line_items if extract_line_items else [],
    }

    # -------------------------------------------------------------------------
    # Step 6: Run validation
    # -------------------------------------------------------------------------
    validation_report: dict = {}
    if validate_totals:
        validation_result = _validate_invoice(invoice_data, line_items)
        validation_report = {
            "overall_status": validation_result["overall_status"],
            "checks": validation_result["checks"],
        }
        result_errors.extend(validation_result["errors"])
        result_warnings.extend(validation_result["warnings"])

    # -------------------------------------------------------------------------
    # Step 7: Determine overall status
    # -------------------------------------------------------------------------
    if result_errors:
        overall_status = "validation_failed"
    elif result_warnings:
        overall_status = "validation_passed_with_warnings"
    else:
        overall_status = "validation_passed"

    # -------------------------------------------------------------------------
    # Step 8: Generate output files
    # -------------------------------------------------------------------------
    # Determine output directory (same as input file directory)
    output_dir = os.path.dirname(os.path.abspath(input_file))
    safe_inv_num = re.sub(r"[^\w\-]", "_", invoice_number) if invoice_number else "unknown"

    # JSON output
    json_filename = f"invoice_{safe_inv_num}.json"
    json_path = os.path.join(output_dir, json_filename)

    output_payload = {
        "status": overall_status,
        "invoice": invoice_data,
        "validation_report": validation_report,
        "errors": result_errors,
        "warnings": result_warnings,
        "output_files": [],  # Will be updated below
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False, default=str)
        output_files.append(json_filename)
        logger.info("JSON output written to '%s'", json_path)
    except Exception as exc:
        logger.error("Failed to write JSON output: %s", exc)
        result_errors.append(f"Failed to write JSON output: {exc}")

    # Excel output (if requested)
    if output_format.lower() == "xlsx":
        xlsx_filename = f"invoice_{safe_inv_num}.xlsx"
        xlsx_path = os.path.join(output_dir, xlsx_filename)
        try:
            _export_to_excel(invoice_data, line_items, xlsx_path)
            output_files.append(xlsx_filename)
        except Exception as exc:
            result_errors.append(f"Failed to generate Excel file: {exc}")

    # Update output_files in the payload
    output_payload["output_files"] = output_files
    output_payload["errors"] = result_errors
    output_payload["warnings"] = result_warnings

    # Re-write JSON with updated output_files list
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass  # Non-critical — the return value still has the correct data

    return output_payload


# =============================================================================
# Mount MCP server as ASGI sub-application
# =============================================================================
app.mount("/mcp", mcp_app)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting PDF Invoice Extractor MCP server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
