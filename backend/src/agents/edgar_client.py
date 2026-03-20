from __future__ import annotations

from src.schemas import CompanyProfile

# XBRL tag → CompanyProfile.financials field name
XBRL_FIELD_MAP = {
    "Revenues": "annual_revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "annual_revenue",
    "CostOfGoodsAndServicesSold": "cogs",
    "CostOfRevenue": "cogs",
    "GrossProfit": "gross_profit_raw",  # Dollar value, converted to ratio later
    "OperatingExpenses": "operating_expenses_total",
    "SellingGeneralAndAdministrativeExpense": "sga",
    "ResearchAndDevelopmentExpense": "rd",
    "DepreciationAndAmortization": "depreciation",
    "NetIncomeLoss": "net_income",
    "Assets": "total_assets",
    "Liabilities": "total_debt",
    "LongTermDebt": "total_debt",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "StockholdersEquity": "equity",
    "PropertyPlantAndEquipmentNet": "ppe_net",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
}


def lookup_company(ticker: str) -> dict | None:
    """Look up a company by stock ticker via edgartools.

    Returns basic company info or None if not found.
    """
    try:
        from edgar import Company
        company = Company(ticker)
        return {
            "name": company.name,
            "ticker": ticker.upper(),
            "cik": str(company.cik),
            "sic": getattr(company, "sic", ""),
            "industry": getattr(company, "industry", ""),
            "state": getattr(company, "state_of_incorporation", ""),
        }
    except Exception:
        return None


def pull_financials(ticker: str) -> dict:
    """Pull latest 10-K financial data from SEC EDGAR.

    Returns a dict of extracted financial fields from XBRL data.
    No LLM needed — XBRL tags map directly to profile fields.
    """
    try:
        from edgar import Company
        company = Company(ticker)

        # Get latest 10-K filing
        filings = company.get_filings(form="10-K")
        if not filings:
            return {"error": "No 10-K filings found"}

        try:
            latest = filings[0]
        except (IndexError, TypeError):
            latest = next(iter(filings), None)
        if latest is None:
            return {"error": "Could not access 10-K filing"}
        xbrl = latest.xbrl()
        if xbrl is None:
            return {"error": "No XBRL data in filing"}

        financials: dict[str, float] = {}
        raw_values: dict[str, float] = {}

        # Extract facts from XBRL
        for tag, field in XBRL_FIELD_MAP.items():
            try:
                facts = xbrl.get_fact(tag)
                if facts is not None:
                    # Get the most recent value
                    if hasattr(facts, "value"):
                        val = float(facts.value)
                    elif hasattr(facts, "__iter__"):
                        # Multiple periods — take the latest
                        values = list(facts)
                        if values:
                            val = float(values[-1].value if hasattr(values[-1], "value") else values[-1])
                        else:
                            continue
                    else:
                        val = float(facts)
                    raw_values[tag] = val
                    financials[field] = val
            except (ValueError, TypeError, AttributeError):
                continue

        # Compute derived fields
        revenue = financials.get("annual_revenue", 0)
        cogs = financials.get("cogs", 0)
        gross_profit_raw = financials.pop("gross_profit_raw", 0)
        if revenue > 0 and cogs > 0:
            financials["gross_margin"] = round((revenue - cogs) / revenue, 4)
        elif gross_profit_raw and revenue > 0:
            financials["gross_margin"] = round(gross_profit_raw / revenue, 4)

        # Debt-to-equity
        debt = financials.get("total_debt", 0)
        equity = financials.get("equity", 0)
        if equity > 0:
            financials["debt_to_equity"] = round(debt / equity, 4)

        # EBITDA approximation
        net_income = financials.get("net_income", 0)
        depreciation = financials.get("depreciation", 0)
        if net_income:
            financials["ebitda"] = net_income + depreciation

        # Extract operating expenses sub-fields
        operating_expenses = {}
        if "sga" in financials:
            operating_expenses["sga"] = financials.pop("sga")
        if "rd" in financials:
            operating_expenses["rd"] = financials.pop("rd")
            financials["rd_spend"] = operating_expenses["rd"]
        if "depreciation" in financials:
            operating_expenses["depreciation"] = financials.pop("depreciation")
        if operating_expenses:
            financials["operating_expenses"] = operating_expenses

        # Clean up intermediate fields
        financials.pop("operating_expenses_total", None)
        financials.pop("ppe_net", None)

        return {
            "ticker": ticker.upper(),
            "financials": financials,
            "raw_xbrl_tags": raw_values,
        }

    except ImportError:
        return {"error": "edgartools not installed"}
    except Exception as e:
        return {"error": str(e)}


def map_to_profile(
    company_info: dict,
    financial_data: dict,
    base_profile: CompanyProfile | None = None,
) -> CompanyProfile:
    """Map EDGAR data directly to a CompanyProfile."""
    profile = base_profile or CompanyProfile()

    # Identity from company info
    if company_info:
        profile.identity.name = company_info.get("name", profile.identity.name)
        profile.identity.industry = company_info.get("industry", profile.identity.industry)
        profile.identity.naics_code = company_info.get("sic", profile.identity.naics_code)
        if company_info.get("state"):
            profile.identity.headquarters = company_info["state"]

    # Financials
    fin_data = financial_data.get("financials", {})
    for field in (
        "annual_revenue", "cogs", "gross_margin", "net_income", "ebitda",
        "total_assets", "total_debt", "cash", "equity", "debt_to_equity",
        "capex", "rd_spend",
    ):
        val = fin_data.get(field)
        if val is not None and hasattr(profile.financials, field):
            setattr(profile.financials, field, val)

    # Operating expenses
    opex = fin_data.get("operating_expenses", {})
    if opex:
        if "sga" in opex:
            profile.financials.operating_expenses.sga = opex["sga"]
        if "rd" in opex:
            profile.financials.operating_expenses.rd = opex["rd"]
        if "depreciation" in opex:
            profile.financials.operating_expenses.depreciation = opex["depreciation"]

    return profile
