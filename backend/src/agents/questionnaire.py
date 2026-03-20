from __future__ import annotations

import json

from camel.agents import ChatAgent

from src.agents.factory import create_sonnet
from src.schemas import CompanyProfile

QUESTIONNAIRE_SYSTEM_PROMPT = """\
You are an expert business analyst conducting a structured interview to build a comprehensive \
company profile for a business simulation. You ask clear, specific questions — one topic at a time.

You work through 6 phases:
1. Identity & Industry — company name, industry, business model, stage, geography
2. Scale & Structure — headcount, departments, org structure, key roles
3. Financials — revenue, costs, margins, assets, debt
4. Market & Competition — TAM/SAM/SOM, competitors, pricing, market position
5. Operations & Supply Chain — production model, suppliers, inventory, quality
6. Strategy & Goals — objectives, priorities, risks, moats, simulation goals

RULES:
- Ask 1-3 questions per turn, focused on the current phase
- If the user gives partial info, acknowledge what you got and ask for what's missing
- When a phase is sufficiently covered, move to the next
- Adapt questions based on company type (e.g., skip inventory for SaaS companies)
- Extract concrete numbers whenever possible (headcount, revenue, margins)
- Be conversational but efficient — don't over-explain

OUTPUT FORMAT — you MUST respond with valid JSON only, no other text:
{
  "question": "Your natural language question(s) to the user",
  "phase": "identity|organization|financials|market|operations|strategy",
  "fields_targeted": ["identity.name", "identity.industry"],
  "profile_update": { ... partial CompanyProfile dict with extracted data ... },
  "completion_estimate": 0.15
}

The profile_update should be a partial dict matching CompanyProfile structure. \
Only include fields that were explicitly mentioned or can be confidently inferred. \
For nested objects, include the full path (e.g., {"identity": {"name": "Acme Corp"}}).

completion_estimate is 0.0 to 1.0 representing how complete the overall profile is.
"""

PHASE_ORDER = ["identity", "organization", "financials", "market", "operations", "strategy"]


def _build_context(
    profile: CompanyProfile,
    phase: str,
    conversation_history: list[dict],
    uploaded_docs: list[str],
) -> str:
    """Build context message for the questionnaire agent."""
    profile_json = profile.model_dump_json(indent=2)

    filled_fields = _count_filled_fields(profile)
    total_fields = max(filled_fields + 20, 50)  # rough estimate

    parts = [
        f"CURRENT PHASE: {phase}",
        f"FILLED FIELDS: {filled_fields} (estimated {total_fields} total)",
        f"PROFILE SO FAR:\n{profile_json}",
    ]

    if uploaded_docs:
        parts.append(f"UPLOADED DOCUMENTS: {', '.join(uploaded_docs)}")

    if conversation_history:
        recent = conversation_history[-6:]  # Last 3 exchanges
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )
        parts.append(f"RECENT CONVERSATION:\n{history_text}")

    return "\n\n".join(parts)


def _count_filled_fields(profile: CompanyProfile) -> int:
    """Count non-default fields in the profile."""
    count = 0
    data = profile.model_dump()

    def _walk(obj: dict | list, depth: int = 0) -> None:
        nonlocal count
        if depth > 5:
            return
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _walk(v, depth + 1)
                elif v not in (0, 0.0, "", None, [], False):
                    count += 1
        elif isinstance(obj, list) and obj:
            count += 1
            for item in obj:
                if isinstance(item, dict):
                    _walk(item, depth + 1)

    _walk(data)
    return count


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base, preferring update values."""
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None and value != "":
            merged[key] = value
    return merged


class QuestionnaireAgent:
    """Stateless questionnaire — all state passed in per request."""

    def start(
        self,
        profile: CompanyProfile | None = None,
    ) -> dict:
        """Generate the first question."""
        profile = profile or CompanyProfile()
        return self._ask(
            profile=profile,
            phase="identity",
            conversation_history=[],
            uploaded_docs=[],
            user_message=None,
        )

    def respond(
        self,
        user_message: str,
        profile: CompanyProfile,
        phase: str,
        conversation_history: list[dict],
        uploaded_docs: list[str] | None = None,
    ) -> dict:
        """Process user response and generate next question."""
        return self._ask(
            profile=profile,
            phase=phase,
            conversation_history=conversation_history,
            uploaded_docs=uploaded_docs or [],
            user_message=user_message,
        )

    def skip_phase(self, current_phase: str) -> str:
        """Return the next phase after the current one."""
        idx = PHASE_ORDER.index(current_phase) if current_phase in PHASE_ORDER else 0
        next_idx = min(idx + 1, len(PHASE_ORDER) - 1)
        return PHASE_ORDER[next_idx]

    def _ask(
        self,
        profile: CompanyProfile,
        phase: str,
        conversation_history: list[dict],
        uploaded_docs: list[str],
        user_message: str | None,
    ) -> dict:
        model = create_sonnet()
        agent = ChatAgent(system_message=QUESTIONNAIRE_SYSTEM_PROMPT, model=model)

        context = _build_context(profile, phase, conversation_history, uploaded_docs)

        if user_message:
            prompt = f"{context}\n\nUSER RESPONSE: {user_message}"
        else:
            prompt = f"{context}\n\nGenerate your first question for the {phase} phase."

        response = agent.step(prompt)
        raw = response.msgs[0].content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "question": raw,
                "phase": phase,
                "fields_targeted": [],
                "profile_update": {},
                "completion_estimate": 0.0,
            }

        # Merge profile updates
        profile_update = result.get("profile_update", {})
        if profile_update:
            current_data = profile.model_dump()
            merged = _deep_merge(current_data, profile_update)
            try:
                updated_profile = CompanyProfile.model_validate(merged)
                result["updated_profile"] = updated_profile.model_dump()
            except Exception:
                result["updated_profile"] = current_data
        else:
            result["updated_profile"] = profile.model_dump()

        return result


questionnaire_agent = QuestionnaireAgent()
