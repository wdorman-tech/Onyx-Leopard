from __future__ import annotations

from camel.agents import ChatAgent

from src.agents.factory import create_sonnet
from src.schemas import (
    CompanyGraph,
    CompanyProfile,
    Department,
    EdgeData,
    KeyRole,
    NodeData,
    SimulationParameters,
)

AGENT_PROMPT_SYSTEM = """\
You generate short, specific agent_prompt strings for simulation nodes.
Given a node's label, type, and context about the company, write a 1-2 sentence prompt \
that tells the node's AI agent what its goals and constraints are during simulation.
Output ONLY the prompt text, no JSON, no markdown.
"""


def compute_sim_params(profile: CompanyProfile) -> SimulationParameters:
    """Derive economic model parameters from profile data."""
    fin = profile.financials
    org = profile.organization
    mkt = profile.market

    revenue = fin.annual_revenue or 1.0
    headcount = org.total_headcount or 1

    # Cobb-Douglas: labor share ~ payroll/revenue
    total_payroll = org.avg_salary * headcount if org.avg_salary else revenue * 0.4
    labor_share = min(max(total_payroll / revenue, 0.3), 0.9) if revenue > 0 else 0.7
    capital_share = 1.0 - labor_share

    # TFP from labor productivity
    tfp = org.labor_productivity_index if org.labor_productivity_index > 0 else 1.0

    # Cost curves
    fixed_costs = fin.operating_expenses.sga + fin.operating_expenses.depreciation
    variable_total = fin.cogs if fin.cogs > 0 else revenue * (1 - fin.gross_margin) if fin.gross_margin > 0 else revenue * 0.5
    units_proxy = revenue / 100.0 if revenue > 0 else 1.0
    variable_cost_per_unit = variable_total / units_proxy if units_proxy > 0 else 0.0

    # Cournot
    tam = mkt.tam if mkt.tam > 0 else revenue * 10
    market_demand_intercept = tam * 2
    market_demand_slope = tam / (revenue if revenue > 0 else 1.0)
    marginal_cost = variable_cost_per_unit

    # Growth
    depreciation_rate = 0.05
    if fin.total_assets > 0 and fin.operating_expenses.depreciation > 0:
        depreciation_rate = min(fin.operating_expenses.depreciation / fin.total_assets, 0.25)

    reinvestment_rate = 0.3
    if fin.capex > 0 and fin.net_income > 0:
        reinvestment_rate = min(fin.capex / fin.net_income, 0.9)

    # Risk
    revenue_volatility = 0.1
    if fin.revenue_growth_rate != 0:
        revenue_volatility = min(abs(fin.revenue_growth_rate) * 0.5, 0.5)

    return SimulationParameters(
        tfp=round(tfp, 4),
        capital_elasticity=round(capital_share, 4),
        labor_elasticity=round(labor_share, 4),
        fixed_costs=round(fixed_costs, 2),
        variable_cost_per_unit=round(variable_cost_per_unit, 4),
        learning_curve_rate=0.85,
        market_demand_intercept=round(market_demand_intercept, 2),
        market_demand_slope=round(market_demand_slope, 4),
        marginal_cost=round(marginal_cost, 4),
        depreciation_rate=round(depreciation_rate, 4),
        reinvestment_rate=round(reinvestment_rate, 4),
        revenue_volatility=round(revenue_volatility, 4),
        demand_seasonality=[1.0] * 12,
    )


def _generate_agent_prompt_llm(
    label: str, node_type: str, company_name: str, description: str
) -> str:
    """Use Sonnet to generate a contextual agent prompt."""
    model = create_sonnet()
    agent = ChatAgent(system_message=AGENT_PROMPT_SYSTEM, model=model)
    prompt = (
        f"Company: {company_name} — {description}\n"
        f"Node: {label} (type: {node_type})\n"
        f"Write its agent_prompt."
    )
    response = agent.step(prompt)
    return response.msgs[0].content.strip()


def _default_agent_prompt(label: str, node_type: str) -> str:
    """Fallback prompt without LLM call."""
    prompts = {
        "department": f"You are the {label} department. Manage your team and budget to maximize departmental KPIs.",
        "team": f"You are the {label} team. Execute your objectives within budget constraints.",
        "role": f"You are the {label}. Drive strategic decisions and coordinate with your reports.",
        "revenue_stream": f"You are the {label} revenue stream. Grow revenue and optimize margins.",
        "cost_center": f"You are the {label} cost center. Minimize costs while maintaining quality.",
        "external": f"You are {label}, an external entity interacting with the company.",
    }
    return prompts.get(node_type, f"You are {label}.")


def generate_company_graph(
    profile: CompanyProfile,
    use_llm_prompts: bool = False,
) -> CompanyGraph:
    """Generate a CompanyGraph from a CompanyProfile deterministically.

    If use_llm_prompts=True, calls Sonnet to generate agent_prompt for each node.
    Otherwise uses template-based prompts (fast, no API cost).
    """
    nodes: list[NodeData] = []
    edges: list[EdgeData] = []
    identity = profile.identity
    org = profile.organization
    fin = profile.financials
    company_name = identity.name
    company_desc = identity.description

    def make_prompt(label: str, ntype: str) -> str:
        if use_llm_prompts:
            return _generate_agent_prompt_llm(label, ntype, company_name, company_desc)
        return _default_agent_prompt(label, ntype)

    # CEO / leadership node
    ceo_id = "ceo"
    ceo_role: KeyRole | None = None
    for role in org.key_roles:
        if "ceo" in role.title.lower() or "chief executive" in role.title.lower():
            ceo_role = role
            break

    ceo_label = ceo_role.title if ceo_role else "CEO"
    nodes.append(NodeData(
        id=ceo_id,
        label=ceo_label,
        type="role",
        metrics={
            "headcount": org.total_headcount,
            "budget": sum(d.budget for d in org.departments),
        },
        agent_prompt=make_prompt(ceo_label, "role"),
    ))

    # Department nodes
    for dept in org.departments:
        nodes.append(NodeData(
            id=dept.id,
            label=dept.name,
            type="department",
            metrics={
                "headcount": dept.headcount,
                "budget": dept.budget,
            },
            agent_prompt=make_prompt(dept.name, "department"),
        ))
        edges.append(EdgeData(
            source=dept.id,
            target=ceo_id,
            relationship="reports_to",
            label=f"{dept.name} reports to {ceo_label}",
        ))

        # Sub-teams
        for team in dept.sub_teams:
            nodes.append(NodeData(
                id=team.id,
                label=team.name,
                type="team",
                metrics={"headcount": team.headcount, "budget": 0},
                agent_prompt=make_prompt(team.name, "team"),
            ))
            edges.append(EdgeData(
                source=team.id,
                target=dept.id,
                relationship="reports_to",
                label=f"{team.name} reports to {dept.name}",
            ))

    # Key roles (non-CEO)
    for role in org.key_roles:
        if ceo_role and role.id == ceo_role.id:
            continue
        nodes.append(NodeData(
            id=role.id,
            label=role.title,
            type="role",
            metrics={"headcount": 1, "budget": 0},
            agent_prompt=make_prompt(role.title, "role"),
        ))
        target = role.reports_to or role.department_id or ceo_id
        edges.append(EdgeData(
            source=role.id,
            target=target,
            relationship="reports_to",
            label=f"{role.title} reports to {target}",
        ))

    # Revenue stream nodes
    for i, stream in enumerate(fin.revenue_streams):
        rs_id = f"revenue_{i}"
        nodes.append(NodeData(
            id=rs_id,
            label=stream.name,
            type="revenue_stream",
            metrics={
                "revenue": stream.annual_revenue,
                "growth_rate": stream.growth_rate,
                "margin": stream.margin,
            },
            agent_prompt=make_prompt(stream.name, "revenue_stream"),
        ))
        edges.append(EdgeData(
            source=rs_id,
            target=ceo_id,
            relationship="funds",
            label=f"{stream.name} funds company",
        ))

    # Competitor nodes (external)
    for i, comp in enumerate(profile.market.competitors[:3]):
        comp_id = f"competitor_{i}"
        nodes.append(NodeData(
            id=comp_id,
            label=comp.name,
            type="external",
            metrics={
                "revenue": comp.est_revenue,
                "market_share": comp.est_market_share,
            },
            agent_prompt=make_prompt(comp.name, "external"),
        ))

    # Enforce MAX_NODES = 20
    nodes = nodes[:20]
    # Remove edges referencing pruned nodes
    node_ids = {n.id for n in nodes}
    edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

    # Global metrics
    total_revenue = fin.annual_revenue or sum(s.annual_revenue for s in fin.revenue_streams)
    total_budget = sum(d.budget for d in org.departments)
    global_metrics = {
        "total_headcount": float(org.total_headcount),
        "total_budget": total_budget,
        "revenue": total_revenue,
    }

    return CompanyGraph(
        name=company_name,
        description=company_desc,
        nodes=nodes,
        edges=edges,
        global_metrics=global_metrics,
    )
