import type { CompanyGraph } from "./graph";

export type BusinessModel =
  | "b2b" | "b2c" | "b2b2c" | "saas" | "marketplace"
  | "manufacturing" | "services" | "ecommerce" | "other";

export type CompanyStage =
  | "pre_revenue" | "early" | "growth" | "mature" | "turnaround";

export type StructureType =
  | "functional" | "divisional" | "matrix" | "flat" | "hierarchical";

export type CompetitiveLandscape =
  | "monopoly" | "oligopoly" | "monopolistic_competition" | "perfect_competition";

export type ProductionModel =
  | "make_to_order" | "make_to_stock" | "digital_delivery"
  | "service_delivery" | "engineer_to_order";

export type PrimaryObjective =
  | "growth" | "profitability" | "market_share" | "survival"
  | "innovation" | "sustainability";

export type PricingModel =
  | "subscription" | "usage_based" | "per_unit" | "freemium"
  | "tiered" | "custom" | "cost_plus" | "value_based";

export type RevenueStreamType =
  | "recurring" | "transactional" | "licensing" | "services"
  | "advertising" | "other";

// --- Section interfaces ---

export interface CompanyIdentity {
  name: string;
  description: string;
  industry: string;
  naics_code: string;
  business_model: BusinessModel | null;
  company_stage: CompanyStage | null;
  year_founded: number | null;
  headquarters: string;
  geographic_scope: string;
  operating_regions: string[];
}

export interface SubTeam {
  id: string;
  name: string;
  headcount: number;
}

export interface Department {
  id: string;
  name: string;
  headcount: number;
  budget: number;
  function: string;
  sub_teams: SubTeam[];
  kpis: Record<string, number>;
}

export interface KeyRole {
  id: string;
  title: string;
  department_id: string;
  reports_to: string;
}

export interface OrganizationStructure {
  total_headcount: number;
  structure_type: StructureType | null;
  departments: Department[];
  key_roles: KeyRole[];
  avg_salary: number;
  turnover_rate: number;
  hiring_cost: number;
  labor_productivity_index: number;
}

export interface RevenueStream {
  name: string;
  annual_revenue: number;
  growth_rate: number;
  margin: number;
  type: RevenueStreamType;
}

export interface OperatingExpenses {
  sga: number;
  rd: number;
  depreciation: number;
}

export interface FinancialProfile {
  currency: string;
  fiscal_year_end: string;
  annual_revenue: number;
  revenue_growth_rate: number;
  revenue_streams: RevenueStream[];
  cogs: number;
  gross_margin: number;
  operating_expenses: OperatingExpenses;
  total_assets: number;
  total_debt: number;
  cash: number;
  equity: number;
  debt_to_equity: number;
  capex: number;
  rd_spend: number;
  dso: number;
  dio: number;
  dpo: number;
  net_income: number;
  ebitda: number;
  roe: number;
  roa: number;
}

export interface Competitor {
  name: string;
  est_revenue: number;
  est_market_share: number;
  relative_cost: number;
  strengths: string[];
  weaknesses: string[];
}

export interface MarketProfile {
  tam: number;
  sam: number;
  som: number;
  market_share: number;
  market_growth_rate: number;
  competitive_landscape: CompetitiveLandscape | null;
  competitors: Competitor[];
  primary_competition_dimension: string;
  barriers_to_entry: string;
  pricing_model: PricingModel | null;
  price_elasticity_estimate: number;
}

export interface KeyInput {
  name: string;
  annual_cost: number;
  pct_of_cogs: number;
  substitutability: string;
}

export interface OperationsProfile {
  production_model: ProductionModel | null;
  key_inputs: KeyInput[];
  supplier_concentration: number;
  capacity_utilization: number;
  inventory_model: string;
  avg_inventory_value: number;
  inventory_turnover: number;
  lead_time_days: number;
  defect_rate: number;
  customer_satisfaction_score: number;
}

export interface PlannedInitiative {
  name: string;
  description: string;
  priority: number;
}

export interface MajorRisk {
  name: string;
  description: string;
  likelihood: string;
  impact: string;
}

export interface StrategyProfile {
  primary_objective: PrimaryObjective | null;
  strategic_priorities: string[];
  planned_initiatives: PlannedInitiative[];
  major_risks: MajorRisk[];
  moats: string[];
  simulation_objectives: string[];
  time_horizon_weeks: number;
}

export interface SimulationParameters {
  tfp: number;
  capital_elasticity: number;
  labor_elasticity: number;
  fixed_costs: number;
  variable_cost_per_unit: number;
  learning_curve_rate: number;
  market_demand_intercept: number;
  market_demand_slope: number;
  marginal_cost: number;
  depreciation_rate: number;
  reinvestment_rate: number;
  revenue_volatility: number;
  demand_seasonality: number[];
}

export interface ProfileMetadata {
  created_by: string;
  source_documents: string[];
  completeness_score: number;
  last_modified: string | null;
  notes: string;
}

export interface CompanyProfile {
  schema_version: string;
  identity: CompanyIdentity;
  organization: OrganizationStructure;
  financials: FinancialProfile;
  market: MarketProfile;
  operations: OperationsProfile;
  strategy: StrategyProfile;
  sim_params: SimulationParameters;
  metadata: ProfileMetadata;
}

// --- Export / Import ---

export interface SimulationSnapshot {
  tick: number;
  outlook: string;
  global_metrics: Record<string, number>;
  history_summary: Record<string, unknown>[];
}

export interface OleoExport {
  format: string;
  format_version: string;
  exported_at: string;
  profile: CompanyProfile;
  graph: CompanyGraph;
  simulation_snapshot: SimulationSnapshot | null;
  checksum: string;
}
