from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from src.config import settings
from src.schemas import CompanyGraph

SYSTEM_PROMPT = """\
You are a business structure analyst. Given a natural language description of a company, \
extract a structured company graph.

Output ONLY valid JSON matching this schema:
{
  "name": "Company Name",
  "description": "Brief description",
  "nodes": [
    {
      "id": "unique_id",
      "label": "Display Name",
      "type": "department|team|role|revenue_stream|cost_center|external",
      "metrics": {"headcount": 0, "budget": 0},
      "agent_prompt": "You are the [entity]. Your goal is to..."
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id",
      "relationship": "reports_to|funds|supplies|collaborates|serves",
      "label": "Description"
    }
  ],
  "global_metrics": {"total_headcount": 0, "total_budget": 0, "revenue": 0}
}

Rules:
- Every company must have at least a CEO/leadership node
- Departments should have headcount and budget metrics
- Revenue streams should have revenue metrics
- Create logical reporting relationships
- Generate meaningful agent_prompt for each node describing its role and goals
"""

REFINE_PROMPT = """\
You are a business structure analyst. You have an existing company graph (provided as JSON) \
and the user wants to modify it.

Apply the user's requested changes to the existing graph. Output the COMPLETE updated graph \
as valid JSON matching the same schema. Preserve existing nodes/edges unless the user explicitly \
asks to remove or change them.
"""


def create_parser_agent() -> ChatAgent:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type=ModelType.CLAUDE_3_5_SONNET,
        api_key=settings.anthropic_api_key,
    )
    return ChatAgent(system_message=SYSTEM_PROMPT, model=model)


def create_refine_agent() -> ChatAgent:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type=ModelType.CLAUDE_3_5_SONNET,
        api_key=settings.anthropic_api_key,
    )
    return ChatAgent(system_message=REFINE_PROMPT, model=model)


def parse_company(description: str) -> CompanyGraph:
    agent = create_parser_agent()
    response = agent.step(description)
    raw = response.msgs[0].content
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return CompanyGraph.model_validate_json(raw)


def refine_company(instruction: str, current_graph: CompanyGraph) -> CompanyGraph:
    agent = create_refine_agent()
    prompt = f"Current graph:\n{current_graph.model_dump_json(indent=2)}\n\nUser instruction: {instruction}"
    response = agent.step(prompt)
    raw = response.msgs[0].content
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return CompanyGraph.model_validate_json(raw)
