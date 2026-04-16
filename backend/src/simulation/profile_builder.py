"""AI-powered business profile builder — generates IndustrySpec from user interview."""

from __future__ import annotations

import json
import logging
import re
import tempfile
import uuid
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.simulation.ceo_agent import CEO_MODEL, _get_client
from src.simulation.config_loader import IndustrySpec, LocationDefaults, _INDUSTRY_DIR

log = logging.getLogger(__name__)

# ── Session model ──


class InterviewSession(BaseModel):
    id: str
    transcript: list[dict] = []  # [{role: "assistant"/"user", content: "..."}]
    document_summaries: list[dict] = []  # [{filename: str, summary: str}]
    generated_spec: dict | None = None
    status: str = "interviewing"  # "interviewing" | "generating" | "complete" | "error"
    error: str | None = None


# In-memory sessions (fine for local-run open-source project)
_sessions: dict[str, InterviewSession] = {}


def get_session(session_id: str) -> InterviewSession | None:
    return _sessions.get(session_id)


# ── Interview system prompt ──

_INTERVIEW_SYSTEM = """You are a business analyst conducting an interview to build a simulation profile.
Your goal is to understand the user's business well enough to generate a complete
industry simulation configuration.

You need to gather information about:
1. BUSINESS TYPE: What the company does, what it sells, the industry
2. REVENUE MODEL: How money is made (physical sales, subscriptions, services, etc.)
3. ECONOMICS: Pricing, variable costs per unit, fixed costs, supply chain
4. LOCATIONS/UNITS: What a "location" means for this business (store, office, product line, etc.)
5. ORGANIZATIONAL STRUCTURE: Key roles as the company grows (first hires, department heads, etc.)
6. GROWTH MILESTONES: What triggers growth (revenue thresholds, headcount, location count)
7. EXTERNAL DEPENDENCIES: Suppliers, partners, investors
8. REVENUE STREAMS: Beyond the core product (consulting, licensing, etc.)

RULES:
- Ask ONE focused question at a time
- Be conversational but efficient — aim for 8-12 questions total
- Adapt follow-up questions based on answers
- After gathering enough info, say EXACTLY "INTERVIEW_COMPLETE" on its own line,
  followed by a brief summary of what you learned
- Do NOT generate YAML or technical output — just interview naturally
- For the FIRST message, introduce yourself briefly and ask what the business does"""


def start_session() -> InterviewSession:
    """Create a new interview session and generate the first question."""
    session = InterviewSession(id=str(uuid.uuid4()))
    _sessions[session.id] = session
    return session


async def get_first_question(session: InterviewSession) -> str:
    """Get the opening question from Claude."""
    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=300,
        system=_INTERVIEW_SYSTEM,
        messages=[{"role": "user", "content": "Start the interview."}],
    )
    question = response.content[0].text
    session.transcript.append({"role": "assistant", "content": question})
    return question


async def process_answer(session: InterviewSession, answer: str) -> dict:
    """Process a user answer and get the next question or complete the interview.

    Returns: {next_question: str, progress: float, is_complete: bool}
    """
    session.transcript.append({"role": "user", "content": answer})

    # Build messages for Claude from transcript
    messages = []
    for entry in session.transcript:
        messages.append({"role": entry["role"], "content": entry["content"]})

    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=400,
        system=_INTERVIEW_SYSTEM,
        messages=messages,
    )
    reply = response.content[0].text
    session.transcript.append({"role": "assistant", "content": reply})

    # Check if interview is complete
    is_complete = "INTERVIEW_COMPLETE" in reply
    if is_complete:
        session.status = "generating"

    # Estimate progress (rough: count Q&A pairs, target ~10)
    qa_count = sum(1 for t in session.transcript if t["role"] == "user")
    progress = min(0.95, qa_count / 10.0)
    if is_complete:
        progress = 1.0

    return {
        "next_question": reply,
        "progress": round(progress, 2),
        "is_complete": is_complete,
    }


# ── YAML generation ──

_REFERENCE_YAML = (_INDUSTRY_DIR / "restaurant.yaml").read_text()

_GENERATION_SYSTEM = f"""You are a simulation configuration generator. Given an interview transcript
about a business, generate a complete IndustrySpec YAML configuration that can
power a business simulation.

The YAML must follow this exact structure (reference example below).
Every field shown is required unless noted.

KEY RULES:
- meta.slug must be lowercase with underscores only
- roles.location_type and roles.founder_type MUST exist as keys in the nodes dict
- Every trigger node_type MUST exist as a key in the nodes dict
- roles.supplier_types entries MUST exist as keys in the nodes dict
- Exactly ONE trigger must have is_location_expansion: true
- economics_model must be "physical", "subscription", or "service"
- For subscription businesses: set capacity_decay_rate to 0, set churn_rate > 0
- For service businesses: set capacity_decay_rate low (bench time)
- For physical businesses: set capacity_decay_rate > 0 (spoilage/waste)
- Node categories must be: location, corporate, external, or revenue
- Bridge quality_modifier_keys should reference keys that appear in node revenue_modifiers
- Trigger conditions use this DSL:
  Simple: {{"monthly_revenue": {{">": 15000}}}}
  AND: {{"all": [...]}}
  OR: {{"any": [...]}}
  Has node: {{"has_node": "node_type_key"}}

Output ONLY valid YAML — no markdown fences, no explanation, just the YAML.

REFERENCE EXAMPLE (restaurant industry):
```yaml
{_REFERENCE_YAML}
```"""


async def generate_spec(session: InterviewSession) -> dict:
    """Generate an IndustrySpec from the interview transcript + documents.

    Returns the raw dict (YAML-parsed) on success.
    Raises ValueError on failure after retries.
    """
    # Build the user prompt from transcript + docs
    transcript_text = "\n".join(
        f"{'Q' if t['role'] == 'assistant' else 'A'}: {t['content']}"
        for t in session.transcript
    )

    doc_text = ""
    if session.document_summaries:
        doc_text = "\n\nUPLOADED DOCUMENTS:\n"
        for doc in session.document_summaries:
            doc_text += f"\n--- {doc['filename']} ---\n{doc['summary']}\n"

    user_prompt = f"""Based on this business interview, generate a complete IndustrySpec YAML.

INTERVIEW TRANSCRIPT:
{transcript_text}
{doc_text}

Generate the YAML now. Remember: every trigger node_type and role type must exist in the nodes dict."""

    client = _get_client()
    last_error = ""

    for attempt in range(3):
        try:
            retry_hint = ""
            if attempt > 0 and last_error:
                retry_hint = f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{last_error}\nFix the issue and regenerate."

            response = await client.messages.create(
                model=CEO_MODEL,
                max_tokens=4000,
                system=_GENERATION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt + retry_hint}],
            )
            raw_yaml = response.content[0].text

            # Strip markdown fences if present
            raw_yaml = re.sub(r"^```(?:yaml)?\s*\n", "", raw_yaml, flags=re.MULTILINE)
            raw_yaml = re.sub(r"\n```\s*$", "", raw_yaml, flags=re.MULTILINE)

            parsed = yaml.safe_load(raw_yaml)
            if not isinstance(parsed, dict):
                raise ValueError("YAML did not parse to a dict")

            # Validate with Pydantic
            spec = IndustrySpec(**parsed)

            # Cross-reference validation (same as config_loader)
            if spec.roles.location_type not in spec.nodes:
                raise ValueError(
                    f"roles.location_type '{spec.roles.location_type}' not in nodes"
                )
            if spec.roles.founder_type not in spec.nodes:
                raise ValueError(
                    f"roles.founder_type '{spec.roles.founder_type}' not in nodes"
                )
            for st in spec.roles.supplier_types:
                if st not in spec.nodes:
                    raise ValueError(f"supplier_type '{st}' not in nodes")
            for trigger in spec.triggers:
                if trigger.node_type not in spec.nodes:
                    raise ValueError(
                        f"trigger node_type '{trigger.node_type}' not in nodes"
                    )

            # Ensure exactly one expansion trigger
            expansion = [t for t in spec.triggers if t.is_location_expansion]
            if len(expansion) != 1:
                raise ValueError(
                    f"Expected exactly 1 expansion trigger, got {len(expansion)}"
                )

            session.generated_spec = parsed
            session.status = "complete"
            return parsed

        except Exception as e:
            last_error = str(e)
            log.warning(
                "Spec generation attempt %d/3 failed: %s", attempt + 1, last_error
            )

    session.status = "error"
    session.error = f"Failed to generate valid spec after 3 attempts: {last_error}"
    raise ValueError(session.error)


def save_industry(slug: str, spec_dict: dict) -> Path:
    """Save a generated industry spec to the industries directory."""
    filename = f"custom_{slug}.yaml"
    path = _INDUSTRY_DIR / filename
    with open(path, "w") as f:
        yaml.dump(spec_dict, f, default_flow_style=False, sort_keys=False, width=120)
    return path


# ── Document upload + extraction ──

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES_PER_SESSION = 5


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from an uploaded file based on extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith(".txt") or lower.endswith(".csv") or lower.endswith(".md"):
        return file_bytes.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {filename}. Supported: .pdf, .txt, .csv, .md")


async def summarize_document(filename: str, text: str) -> str:
    """Send extracted text to Claude for business-relevant summarization."""
    # Truncate very long documents to fit in context
    if len(text) > 50_000:
        text = text[:50_000] + "\n\n[... truncated ...]"

    client = _get_client()
    response = await client.messages.create(
        model=CEO_MODEL,
        max_tokens=800,
        system=(
            "Extract business-relevant information from this document. Focus on: "
            "revenue model, pricing, cost structure, organizational structure, "
            "growth metrics, competitive landscape, market size, and any financial data. "
            "Be concise — bullet points preferred. If the document is not business-related, "
            "say so briefly."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Document: {filename}\n\n{text}",
            }
        ],
    )
    return response.content[0].text


async def process_upload(
    session: InterviewSession, filename: str, file_bytes: bytes
) -> dict:
    """Process an uploaded file: extract text, summarize, store in session.

    Returns: {filename, summary}
    """
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {len(file_bytes)} bytes (max {MAX_FILE_SIZE})")
    if len(session.document_summaries) >= MAX_FILES_PER_SESSION:
        raise ValueError(f"Max {MAX_FILES_PER_SESSION} files per session")

    text = extract_text(filename, file_bytes)
    if not text.strip():
        raise ValueError("No text could be extracted from the file")

    summary = await summarize_document(filename, text)

    doc_entry = {"filename": filename, "summary": summary}
    session.document_summaries.append(doc_entry)
    return doc_entry
