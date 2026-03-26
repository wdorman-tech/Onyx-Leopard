"""Industry registry — defines all supported company archetypes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndustryConfig:
    slug: str
    name: str
    description: str
    icon: str
    playable: bool
    total_nodes: int
    growth_stages: int
    key_metrics: tuple[str, ...]
    example_nodes: tuple[str, ...]
    categories: dict[str, int] = field(default_factory=dict)


INDUSTRY_REGISTRY: dict[str, IndustryConfig] = {
    cfg.slug: cfg
    for cfg in [
        IndustryConfig(
            "restaurant", "Restaurant / Food Service",
            "Grow a grilled chicken chain from a single location to a national brand",
            "utensils-crossed", True,
            total_nodes=50, growth_stages=4,
            key_metrics=("Daily Revenue", "Avg Satisfaction", "Food Cost %", "Total Locations"),
            example_nodes=("Restaurant Location", "Commissary Kitchen", "Marketing Dept", "Catering Revenue"),
            categories={"Locations": 8, "Corporate": 16, "External": 6, "Revenue": 6},
        ),
        IndustryConfig(
            "saas-tech-startup", "SaaS / Tech Startup",
            "Build a software company from MVP to enterprise platform",
            "monitor", False,
            total_nodes=85, growth_stages=10,
            key_metrics=("MRR", "Churn Rate", "CAC/LTV Ratio", "Burn Rate"),
            example_nodes=("Engineering Team", "Cloud Infrastructure", "Sales Dev Reps", "Enterprise Tier Revenue"),
            categories={"Locations": 8, "Corporate": 30, "External": 22, "Revenue": 15},
        ),
        IndustryConfig(
            "professional-services", "Professional Services",
            "Scale a law firm or consultancy from solo practice to global firm",
            "briefcase", False,
            total_nodes=82, growth_stages=10,
            key_metrics=("Billable Utilization", "Revenue/Partner", "Realization Rate", "Leverage Ratio"),
            example_nodes=("Practice Group", "Partner Track", "Client Dev Team", "Retainer Revenue"),
            categories={"Locations": 10, "Corporate": 28, "External": 22, "Revenue": 14},
        ),
        IndustryConfig(
            "healthcare-medical-practice", "Healthcare / Medical Practice",
            "Grow a medical practice from single provider to health system",
            "heart-pulse", False,
            total_nodes=88, growth_stages=10,
            key_metrics=("Patient Volume", "Payer Mix", "Collection Rate", "Provider Productivity"),
            example_nodes=("Primary Care Clinic", "Credentialing Dept", "EHR System", "Telehealth Revenue"),
            categories={"Locations": 12, "Corporate": 28, "External": 26, "Revenue": 16},
        ),
        IndustryConfig(
            "real-estate", "Real Estate",
            "Develop and manage properties from first deal to portfolio scale",
            "building-2", False,
            total_nodes=90, growth_stages=10,
            key_metrics=("Cap Rate", "NOI", "AUM", "Occupancy Rate"),
            example_nodes=("Multifamily Property", "Property Management", "Title Company", "Rental Income"),
            categories={"Locations": 14, "Corporate": 26, "External": 28, "Revenue": 16},
        ),
        IndustryConfig(
            "manufacturing", "Manufacturing",
            "Build a manufacturing operation from prototype to mass production",
            "factory", False,
            total_nodes=92, growth_stages=10,
            key_metrics=("Yield Rate", "OEE", "BOM Cost", "Inventory Turns"),
            example_nodes=("Production Line", "Quality Control", "Raw Material Supplier", "OEM Revenue"),
            categories={"Locations": 12, "Corporate": 30, "External": 26, "Revenue": 14},
        ),
        IndustryConfig(
            "retail", "Retail",
            "Expand a brick-and-mortar chain from flagship to national presence",
            "store", False,
            total_nodes=84, growth_stages=10,
            key_metrics=("Same-Store Comps", "Inventory Turnover", "Sales/Sq Ft", "Shrinkage Rate"),
            example_nodes=("Flagship Store", "Distribution Center", "Visual Merchandising", "Private Label Revenue"),
            categories={"Locations": 10, "Corporate": 26, "External": 24, "Revenue": 16},
        ),
        IndustryConfig(
            "ecommerce-dtc", "E-Commerce / DTC",
            "Launch and scale a direct-to-consumer brand",
            "shopping-cart", False,
            total_nodes=82, growth_stages=10,
            key_metrics=("CAC", "LTV", "Return Rate", "AOV"),
            example_nodes=("Fulfillment Center", "Performance Marketing", "3PL Partner", "Subscription Revenue"),
            categories={"Locations": 8, "Corporate": 28, "External": 24, "Revenue": 16},
        ),
        IndustryConfig(
            "construction", "Construction",
            "Grow a general contractor from small jobs to major projects",
            "hard-hat", False,
            total_nodes=90, growth_stages=10,
            key_metrics=("Bonding Capacity", "Backlog", "Bid-Hit Ratio", "Change Order %"),
            example_nodes=("Equipment Yard", "Estimating Dept", "Surety Bond Provider", "T&M Revenue"),
            categories={"Locations": 10, "Corporate": 30, "External": 28, "Revenue": 14},
        ),
        IndustryConfig(
            "financial-services", "Financial Services",
            "Scale a financial advisory from solo RIA to wealth management platform",
            "landmark", False,
            total_nodes=91, growth_stages=10,
            key_metrics=("AUM", "Fee Revenue", "Client Retention", "Compliance Score"),
            example_nodes=("Branch Office", "Compliance Dept", "Custodian Platform", "Advisory Fee Revenue"),
            categories={"Locations": 10, "Corporate": 30, "External": 26, "Revenue": 18},
        ),
        IndustryConfig(
            "media-content", "Media / Content",
            "Build a media company from indie creator to entertainment powerhouse",
            "film", False,
            total_nodes=84, growth_stages=10,
            key_metrics=("Audience Size", "Ad CPM", "IP Library Value", "Subscriber Count"),
            example_nodes=("Production Studio", "Content Team", "Distribution Platform", "Licensing Revenue"),
            categories={"Locations": 10, "Corporate": 28, "External": 24, "Revenue": 16},
        ),
        IndustryConfig(
            "agriculture", "Agriculture",
            "Grow a farming operation from single crop to vertically integrated agribusiness",
            "wheat", False,
            total_nodes=86, growth_stages=10,
            key_metrics=("Yield/Acre", "Cost/Bushel", "Commodity Price Exposure", "Irrigated Acres"),
            example_nodes=("Grain Farm", "Agronomist", "Grain Elevator", "Crop Revenue"),
            categories={"Locations": 12, "Corporate": 26, "External": 26, "Revenue": 14},
        ),
        IndustryConfig(
            "logistics-transportation", "Logistics / Transportation",
            "Build a logistics company from single truck to fleet operation",
            "truck", False,
            total_nodes=82, growth_stages=10,
            key_metrics=("Fleet Utilization", "Revenue/Truck/Week", "Empty Miles %", "On-Time Delivery"),
            example_nodes=("Terminal Facility", "Dispatch Center", "Freight Broker", "LTL Revenue"),
            categories={"Locations": 10, "Corporate": 28, "External": 24, "Revenue": 14},
        ),
        IndustryConfig(
            "hospitality-hotel", "Hospitality / Hotel",
            "Develop a hotel company from single property to hospitality group",
            "hotel", False,
            total_nodes=86, growth_stages=10,
            key_metrics=("RevPAR", "ADR", "Occupancy Rate", "GOP Margin"),
            example_nodes=("Boutique Hotel", "Revenue Management", "Booking.com Partnership", "F&B Revenue"),
            categories={"Locations": 30, "Corporate": 24, "External": 30, "Revenue": 24},
        ),
        IndustryConfig(
            "education-edtech", "Education / EdTech",
            "Build an education venture from single program to learning platform",
            "graduation-cap", False,
            total_nodes=90, growth_stages=10,
            key_metrics=("Enrollment", "Completion Rate", "Revenue/Student", "Accreditation Status"),
            example_nodes=("Campus Location", "Curriculum Development", "LMS Platform", "Tuition Revenue"),
            categories={"Locations": 12, "Corporate": 28, "External": 26, "Revenue": 16},
        ),
        IndustryConfig(
            "energy-utilities", "Energy / Utilities",
            "Develop energy projects from first solar farm to major IPP",
            "zap", False,
            total_nodes=88, growth_stages=10,
            key_metrics=("MW Capacity", "Capacity Factor", "LCOE", "PPA Backlog"),
            example_nodes=("Solar Farm", "Interconnection Team", "Tax Equity Investor", "PPA Revenue"),
            categories={"Locations": 24, "Corporate": 32, "External": 32, "Revenue": 24},
        ),
    ]
}
