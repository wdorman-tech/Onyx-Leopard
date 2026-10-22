import { createHash } from "crypto";
import { readdir, readFile } from "fs/promises";
import { join } from "path";

type EconomicsModel = "physical" | "subscription" | "service";
type SeedArchetype = "solo_founder" | "small_team" | "venture_funded" | "enterprise";

interface CompanySeed {
  name: string;
  niche: string;
  archetype: SeedArchetype;
  industry_keywords: string[];
  location_label: string;
  economics_model: EconomicsModel;
  starting_price: number;
  base_unit_cost: number;
  daily_fixed_costs: number;
  starting_cash: number;
  starting_employees: number;
  base_capacity_per_location: number;
  margin_target: number;
  revenue_per_employee_target: number;
  tam: number;
  competitor_density: number;
  market_growth_rate: number;
  customer_unit_label: string;
  seasonality_amplitude: number;
  initial_supplier_types: string[];
  initial_revenue_streams: string[];
  initial_cost_centers: string[];
  initial_locations: number;
  initial_marketing_intensity: number;
  initial_quality_target: number;
  initial_price_position: "discount" | "mid" | "premium";
  initial_capital_runway_months: number;
  initial_hiring_pace: "frozen" | "slow" | "steady" | "aggressive";
  initial_geographic_scope: "local" | "regional" | "national" | "global";
  initial_revenue_concentration: number;
  initial_customer_acquisition_channel:
    | "word_of_mouth"
    | "outbound_sales"
    | "paid_ads"
    | "content"
    | "partnerships";
  cobb_douglas_alpha: number;
  cobb_douglas_beta: number;
}

interface CompanyEntry {
  filename: string;
  name: string;
  industry: string;
  stage: string;
  node_count: number;
  seed: CompanySeed;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function compactKeywords(industry: string): string[] {
  const words = industry
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((word) => word.length > 2);
  return Array.from(new Set(words)).slice(0, 12);
}

function economicsFor(identity: Record<string, unknown>): EconomicsModel {
  const text = `${identity.business_model ?? ""} ${identity.industry ?? ""}`.toLowerCase();
  if (text.includes("saas") || text.includes("software") || text.includes("cloud")) {
    return "subscription";
  }
  if (text.includes("service") || text.includes("consulting")) return "service";
  return "physical";
}

function archetypeFor(headcount: number, stage: string): SeedArchetype {
  if (headcount >= 200 || stage.toLowerCase().includes("mature")) return "enterprise";
  if (stage.toLowerCase().includes("growth") || headcount >= 30) return "venture_funded";
  if (headcount > 3) return "small_team";
  return "solo_founder";
}

function refsFor(economics: EconomicsModel) {
  if (economics === "subscription") {
    return {
      suppliers: ["cloud_provider", "payment_processor"],
      revenue: ["subscription_revenue", "usage_based_billing"],
      costs: ["it_support", "founder_engineer"],
      location: "Product",
      customer: "subscribers",
    };
  }
  if (economics === "service") {
    return {
      suppliers: ["ai_tooling_stack", "ai_platform_partner"],
      revenue: ["professional_services", "retainer_contracts"],
      costs: ["it_support", "training_program"],
      location: "Engagement",
      customer: "engagements",
    };
  }
  return {
    suppliers: ["produce_supplier", "payment_processor"],
    revenue: ["delivery_partnership", "loyalty_program"],
    costs: ["owner_operator", "training_program"],
    location: "Location",
    customer: "customers",
  };
}

function buildSeed(data: Record<string, unknown>): CompanySeed {
  const profile = (data.profile ?? {}) as Record<string, unknown>;
  const identity = (profile.identity ?? {}) as Record<string, unknown>;
  const organization = (profile.organization ?? {}) as Record<string, unknown>;
  const financials = (profile.financials ?? {}) as Record<string, unknown>;
  const market = (profile.market ?? {}) as Record<string, unknown>;

  const name = String(identity.name || "Imported Company");
  const industry = String(identity.industry || "business");
  const stage = String(identity.company_stage || "");
  const economics = economicsFor(identity);
  const refs = refsFor(economics);
  const headcount = Math.round(asNumber(organization.total_headcount, 10));
  const annualRevenue = asNumber(financials.annual_revenue, 1_000_000);
  const growth = asNumber(financials.revenue_growth_rate, asNumber(market.market_growth_rate, 0.08));
  const marketShare = clamp(asNumber(market.market_share, 0.05), 0.001, 0.95);
  const estimatedTam = annualRevenue / marketShare;
  const grossMargin = clamp(asNumber(financials.gross_margin, 0.55), 0.05, 0.85);
  const dailyRevenue = annualRevenue / 365;
  const dailyFixed = clamp(dailyRevenue * (1 - grossMargin) * 0.35, 1, 1_000_000);
  const startingPrice = economics === "service" ? 25_000 : economics === "subscription" ? 120 : 50;

  return {
    name,
    niche: String(identity.description || `${name} in ${industry}`).slice(0, 400),
    archetype: archetypeFor(headcount, stage),
    industry_keywords: compactKeywords(industry).length > 0 ? compactKeywords(industry) : ["business"],
    location_label: refs.location,
    economics_model: economics,
    starting_price: startingPrice,
    base_unit_cost: Number((startingPrice * (1 - grossMargin)).toFixed(2)),
    daily_fixed_costs: Number(dailyFixed.toFixed(2)),
    starting_cash: clamp(annualRevenue * 0.15, 1_000, 1_000_000_000),
    starting_employees: clamp(headcount, 1, 100_000),
    base_capacity_per_location: clamp(Math.round(dailyRevenue / startingPrice), 1, 1_000_000),
    margin_target: grossMargin,
    revenue_per_employee_target: clamp(annualRevenue / Math.max(headcount, 1), 1, 10_000_000),
    tam: clamp(estimatedTam, 1, 1e15),
    competitor_density: clamp(Math.round((1 - marketShare) * 10), 0, 10),
    market_growth_rate: clamp(growth, -0.5, 2.0),
    customer_unit_label: refs.customer,
    seasonality_amplitude: 0.1,
    initial_supplier_types: refs.suppliers,
    initial_revenue_streams: refs.revenue,
    initial_cost_centers: refs.costs,
    initial_locations: clamp(Math.ceil(headcount / 5000), 1, 1000),
    initial_marketing_intensity: clamp(0.2 + growth, 0, 1),
    initial_quality_target: 0.8,
    initial_price_position: grossMargin > 0.6 ? "premium" : "mid",
    initial_capital_runway_months: 18,
    initial_hiring_pace: growth > 0.25 ? "aggressive" : growth > 0.05 ? "steady" : "slow",
    initial_geographic_scope: String(identity.geographic_scope || "").toLowerCase().includes("global")
      ? "global"
      : "national",
    initial_revenue_concentration: 0.25,
    initial_customer_acquisition_channel: economics === "subscription" ? "content" : "partnerships",
    cobb_douglas_alpha: economics === "physical" ? 0.45 : 0.35,
    cobb_douglas_beta: economics === "physical" ? 0.55 : 0.65,
  };
}

// Sorted-key, separator-free JSON.stringify mirrors Python's
// `json.dumps(..., sort_keys=True, separators=(',', ':'))` so the hash
// stays stable regardless of which side wrote the file.
function canonicalize(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(",")}]`;
  }
  const entries = Object.keys(value as Record<string, unknown>).sort();
  const pairs = entries.map(
    (k) => `${JSON.stringify(k)}:${canonicalize((value as Record<string, unknown>)[k])}`,
  );
  return `{${pairs.join(",")}}`;
}

function verifyChecksum(data: unknown): boolean {
  if (data === null || typeof data !== "object") return false;
  const claimed = (data as Record<string, unknown>).checksum;
  if (typeof claimed !== "string" || claimed.length === 0) return false;
  const rest = { ...(data as Record<string, unknown>) };
  delete rest.checksum;
  const computed = createHash("sha256")
    .update(canonicalize(rest))
    .digest("hex");
  return computed === claimed;
}

export async function GET() {
  const dir = join(process.cwd(), "public", "companies");

  let files: string[];
  try {
    files = await readdir(dir);
  } catch {
    return Response.json([]);
  }

  const oleoFiles = files.filter((f) => f.endsWith(".oleo"));
  const entries: CompanyEntry[] = [];

  for (const file of oleoFiles) {
    try {
      const raw = await readFile(join(dir, file), "utf-8");
      const data = JSON.parse(raw);
      if (!verifyChecksum(data)) {
        console.warn(`Skipping ${file}: checksum mismatch`);
        continue;
      }
      const identity = data?.profile?.identity ?? {};
      entries.push({
        filename: file,
        name: identity.name || file.replace(".oleo", ""),
        industry: identity.industry || "",
        stage: identity.company_stage || "",
        node_count: data?.graph?.nodes?.length ?? 0,
        seed: buildSeed(data),
      });
    } catch {
      // skip malformed files
    }
  }

  entries.sort((a, b) => a.name.localeCompare(b.name));
  return Response.json(entries);
}
