from __future__ import annotations

import base64
import json
import uuid
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

from camel.agents import ChatAgent

from src.agents.factory import create_haiku
from src.schemas import CompanyProfile, FinancialProfile, OrganizationStructure

CLASSIFY_PROMPT = """\
You classify financial/business documents. Given the document content, output ONLY valid JSON:
{
  "category": "income_statement|balance_sheet|cash_flow|10k_annual|10q_quarterly|budget|org_chart|pitch_deck|custom",
  "confidence": 0.95,
  "period": "2025-Q4 or 2025-FY or unknown",
  "summary": "One-line description of what this document contains"
}
"""

EXTRACT_PROMPT = """\
You extract structured financial data from documents. Given a classified document, \
extract all relevant data points and output ONLY valid JSON matching the requested schema.

Be precise with numbers. Convert all values to raw numbers (not thousands/millions notation). \
If a value is ambiguous or unclear, omit it rather than guess.
"""

EXTRACTION_SCHEMAS: dict[str, str] = {
    "income_statement": """{
  "annual_revenue": number,
  "cogs": number,
  "gross_margin": number (0-1),
  "operating_expenses": {"sga": number, "rd": number, "depreciation": number},
  "net_income": number,
  "ebitda": number,
  "revenue_growth_rate": number (0-1)
}""",
    "balance_sheet": """{
  "total_assets": number,
  "total_debt": number,
  "cash": number,
  "equity": number,
  "debt_to_equity": number
}""",
    "cash_flow": """{
  "capex": number,
  "net_income": number,
  "dso": number,
  "dio": number,
  "dpo": number
}""",
    "10k_annual": """{
  "annual_revenue": number,
  "cogs": number,
  "gross_margin": number,
  "net_income": number,
  "total_assets": number,
  "total_debt": number,
  "cash": number,
  "equity": number,
  "ebitda": number,
  "capex": number,
  "rd_spend": number,
  "revenue_growth_rate": number
}""",
    "budget": """{
  "departments": [{"name": str, "budget": number, "headcount": number}],
  "total_budget": number
}""",
    "org_chart": """{
  "total_headcount": number,
  "departments": [{"name": str, "headcount": number, "function": str}],
  "key_roles": [{"title": str, "department": str}]
}""",
}


class DocumentCategory(str, Enum):
    income_statement = "income_statement"
    balance_sheet = "balance_sheet"
    cash_flow = "cash_flow"
    ten_k_annual = "10k_annual"
    ten_q_quarterly = "10q_quarterly"
    budget = "budget"
    org_chart = "org_chart"
    pitch_deck = "pitch_deck"
    custom = "custom"


class ProcessingJob:
    def __init__(self, file_name: str, content_type: str, raw_data: bytes):
        self.id = str(uuid.uuid4())
        self.file_name = file_name
        self.content_type = content_type
        self.raw_data = raw_data
        self.status: str = "pending"  # pending | classifying | extracting | done | error
        self.category: str | None = None
        self.extraction: dict | None = None
        self.error: str | None = None
        self.profile_fields: dict | None = None


# In-memory job store
_jobs: dict[str, ProcessingJob] = {}


def _prepare_content(job: ProcessingJob) -> str:
    """Prepare document content as text for the LLM."""
    ct = job.content_type.lower()

    if "pdf" in ct:
        # Return base64 for Claude's native PDF support
        b64 = base64.b64encode(job.raw_data).decode()
        return f"[PDF document, base64 encoded: {b64[:200]}... ({len(job.raw_data)} bytes)]"

    if "spreadsheet" in ct or "excel" in ct or job.file_name.endswith((".xlsx", ".xls")):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(BytesIO(job.raw_data), read_only=True, data_only=True)
            lines = []
            for sheet in wb.worksheets[:3]:  # First 3 sheets
                lines.append(f"=== Sheet: {sheet.title} ===")
                for row in sheet.iter_rows(max_row=100, values_only=True):
                    cells = [str(c) if c is not None else "" for c in row]
                    lines.append("\t".join(cells))
            return "\n".join(lines)
        except Exception as e:
            return f"[Excel file, could not parse: {e}]"

    if "csv" in ct or job.file_name.endswith(".csv"):
        text = job.raw_data.decode("utf-8", errors="replace")
        # Truncate to ~5000 chars
        return text[:5000]

    if "image" in ct or job.file_name.endswith((".png", ".jpg", ".jpeg")):
        b64 = base64.b64encode(job.raw_data).decode()
        return f"[Image document, base64: {b64[:200]}... ({len(job.raw_data)} bytes)]"

    # Default: try to decode as text
    try:
        return job.raw_data.decode("utf-8")[:5000]
    except Exception:
        return f"[Binary file: {job.file_name}, {len(job.raw_data)} bytes]"


def classify_document(job: ProcessingJob) -> str:
    """Classify a document using Haiku. Returns category string."""
    job.status = "classifying"
    model = create_haiku()
    agent = ChatAgent(system_message=CLASSIFY_PROMPT, model=model)

    content = _prepare_content(job)
    response = agent.step(f"Classify this document:\n\nFilename: {job.file_name}\n\n{content}")
    raw = response.msgs[0].content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
        category = result.get("category", "custom")
    except json.JSONDecodeError:
        category = "custom"

    job.category = category
    return category


def extract_financial_data(job: ProcessingJob) -> dict:
    """Extract structured data from a classified document."""
    job.status = "extracting"
    model = create_haiku()

    schema = EXTRACTION_SCHEMAS.get(job.category or "", "{}")
    prompt = f"{EXTRACT_PROMPT}\n\nExpected output schema:\n{schema}"
    agent = ChatAgent(system_message=prompt, model=model)

    content = _prepare_content(job)
    response = agent.step(
        f"Document type: {job.category}\nFilename: {job.file_name}\n\nContent:\n{content}\n\n"
        f"Extract the data matching the schema."
    )
    raw = response.msgs[0].content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        extraction = json.loads(raw)
    except json.JSONDecodeError:
        extraction = {}

    job.extraction = extraction
    job.status = "done"
    return extraction


def map_extraction_to_profile(category: str, extraction: dict) -> dict:
    """Map extracted data to CompanyProfile field paths."""
    mapping: dict[str, Any] = {}

    if category in ("income_statement", "10k_annual", "10q_quarterly"):
        fin = {}
        for key in (
            "annual_revenue", "cogs", "gross_margin", "net_income", "ebitda",
            "revenue_growth_rate", "total_assets", "total_debt", "cash",
            "equity", "capex", "rd_spend",
        ):
            if key in extraction and extraction[key] is not None:
                fin[key] = extraction[key]
        if "operating_expenses" in extraction:
            fin["operating_expenses"] = extraction["operating_expenses"]
        if "debt_to_equity" in extraction:
            fin["debt_to_equity"] = extraction["debt_to_equity"]
        if fin:
            mapping["financials"] = fin

    elif category == "balance_sheet":
        fin = {}
        for key in ("total_assets", "total_debt", "cash", "equity", "debt_to_equity"):
            if key in extraction and extraction[key] is not None:
                fin[key] = extraction[key]
        if fin:
            mapping["financials"] = fin

    elif category == "cash_flow":
        fin = {}
        for key in ("capex", "net_income", "dso", "dio", "dpo"):
            if key in extraction and extraction[key] is not None:
                fin[key] = extraction[key]
        if fin:
            mapping["financials"] = fin

    elif category == "budget":
        org: dict[str, Any] = {}
        if "departments" in extraction:
            org["departments"] = extraction["departments"]
        if org:
            mapping["organization"] = org

    elif category == "org_chart":
        org = {}
        if "total_headcount" in extraction:
            org["total_headcount"] = extraction["total_headcount"]
        if "departments" in extraction:
            org["departments"] = extraction["departments"]
        if "key_roles" in extraction:
            org["key_roles"] = extraction["key_roles"]
        if org:
            mapping["organization"] = org

    return mapping


def merge_into_profile(
    profile: CompanyProfile,
    category: str,
    extraction: dict,
) -> CompanyProfile:
    """Merge extracted document data into an existing CompanyProfile.

    Document data takes precedence over questionnaire estimates.
    """
    field_mapping = map_extraction_to_profile(category, extraction)
    profile_data = profile.model_dump()

    for section, fields in field_mapping.items():
        if section not in profile_data:
            continue
        if isinstance(fields, dict):
            for key, value in fields.items():
                if value is not None and value != 0 and value != "":
                    if isinstance(profile_data[section], dict):
                        profile_data[section][key] = value

    return CompanyProfile.model_validate(profile_data)


def create_job(file_name: str, content_type: str, raw_data: bytes) -> ProcessingJob:
    job = ProcessingJob(file_name, content_type, raw_data)
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> ProcessingJob | None:
    return _jobs.get(job_id)
