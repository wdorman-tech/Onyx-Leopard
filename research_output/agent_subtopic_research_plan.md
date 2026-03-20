# Deep Research Plan: Mathematical Economics of Global Business (Expanded)

**Date:** 2026-03-19
**Source Document:** `references/The_Mathematical_Economics_of_Global_Business.md`
**Output Directory:** `research_output/`
**Status:** Planned — ready to execute in batches

---

## Overview

The source document is an 843-line graduate-level reference covering 11 major sections: microeconomic foundations, industrial organization, macroeconomic models, international trade, financial economics, operations research, econometrics, network economics, mechanism design, behavioral economics, and contemporary frontiers.

This research plan expands that document into 20 deep-dive subtopic files, adding:
- National and state/regional-level economics (fiscal federalism, state development, regional modeling)
- Entirely new domains not in the original (health economics, labor economics, public choice, public finance)
- Deeper coverage of existing topics (advanced game theory, causal ML, complexity economics)

Each file targets reference-quality depth with named theorists, mathematical formulations, empirical evidence, and real-world applications.

---

## Execution Strategy

Run in **4 batches of 5 agents** to avoid hitting usage limits. Each batch takes ~5-10 minutes.

### Batch 1: Micro Foundations & Markets
| # | File | Subtopic | Expands On |
|---|------|----------|------------|
| 01 | `01-production-theory-expanded.md` | Multi-output production, dynamic costs, X-inefficiency, SFA, DEA, frontier analysis | Sections I.1-I.2 |
| 02 | `02-consumer-choice-demand-estimation.md` | Discrete choice (BLP), demand estimation, behavioral extensions to utility | Section I.3 |
| 03 | `03-industrial-organization-antitrust.md` | Merger simulation, SSNIP, regulation theory, vertical restraints, digital market regulation | Section II |
| 04 | `04-advanced-game-theory.md` | Evolutionary games, cooperative games (Shapley), bargaining, global games, mean field games | Section II.1 |
| 05 | `05-growth-development-economics.md` | Growth accounting, institutions (AJR), convergence clubs, poverty traps, structural transformation | Section III.1-III.2 |

### Batch 2: Macro, Money & Public Finance
| # | File | Subtopic | Expands On |
|---|------|----------|------------|
| 06 | `06-monetary-economics-central-banking.md` | Money demand, unconventional policy (QE, YCC), CBDCs, currency crises, transmission mechanisms | Sections III.3, III.5 |
| 07 | `07-public-finance-optimal-taxation.md` | Mirrlees, Ramsey rule, tax incidence, capital taxation, behavioral public finance | NEW |
| 08 | `08-fiscal-federalism-state-local-finance.md` | Tiebout, tax competition, property tax, municipal bonds, SALT, pension obligations | NEW |
| 09 | `09-regional-urban-economics.md` | New Economic Geography, agglomeration, Rosen-Roback, housing markets, place-based policy | NEW |
| 10 | `10-trade-policy-sanctions.md` | Trade wars, US-China tariffs, sanctions economics, industrial policy revival, state-level trade exposure | Section IV expanded |

### Batch 3: Applied Fields
| # | File | Subtopic | Expands On |
|---|------|----------|------------|
| 11 | `11-financial-risk-systemic-risk.md` | VaR/CVaR, copulas, CoVaR, SRISK, contagion, stress testing, Basel framework | Section V expanded |
| 12 | `12-labor-economics-human-capital.md` | DMP search/matching, Mincer, wage inequality, minimum wage, automation/AI, immigration | NEW |
| 13 | `13-environmental-resource-economics.md` | Carbon pricing design, Hotelling, Ostrom commons, green finance, state-level environmental policy | Section XI.1 expanded |
| 14 | `14-health-economics-insurance.md` | Arrow 1963, Rothschild-Stiglitz, pharma economics, Grossman model, Medicaid, pandemic economics | NEW |
| 15 | `15-public-choice-political-economy.md` | Arrow impossibility, median voter, rent-seeking, regulatory capture, political business cycles | NEW |

### Batch 4: Methods, Policy & Computation
| # | File | Subtopic | Expands On |
|---|------|----------|------------|
| 16 | `16-national-economic-accounting-policy.md` | GDP measurement, fiscal multipliers, debt sustainability, MMT, state GDP methodology | NEW |
| 17 | `17-state-local-economic-development.md` | Economic base theory, shift-share, IMPLAN, enterprise zones, Opportunity Zones, cluster analysis | NEW |
| 18 | `18-econometrics-causal-ml.md` | Double ML, causal forests, synthetic DiD, Bayesian methods, spatial econometrics, text-as-data | Section VII expanded |
| 19 | `19-behavioral-economics-nudge-policy.md` | Choice architecture, nudge units, behavioral IO, neuroeconomics, behavioral welfare | Section X expanded |
| 20 | `20-complexity-economics-computational.md` | ABM, network science, CGE modeling, microsimulation, economic complexity index, chaos theory | Section VIII expanded |

---

## Per-File Requirements

Each output file should contain:

1. **Frontmatter**: Title, description, date, relationship to source document sections
2. **Named theorists and papers** with year citations (e.g., "Mirrlees (1971)")
3. **Mathematical formulations** in LaTeX (`$...$` for inline, `$$...$$` for display)
4. **Mechanisms and intuitions** — not just definitions, but *why* things work
5. **Real-world examples** — case studies, empirical estimates, policy applications
6. **Multi-level coverage** — global, national, and state/regional where applicable
7. **Proper markdown structure** — `##` sections, `###` subsections, tables where useful

Target length: 500-1000 lines per file (reference-quality depth).

---

## Final Deliverable

After all 20 files are written, generate `index.md` with:
- Overall title and date
- Linked table of contents to each file
- One-line description per subtopic
- Cross-references to the original source document sections

---

## Key Themes Across All Subtopics

### Global → National → State/Local Pipeline
Many economic principles scale across levels but behave differently:
- **Production functions**: firm-level vs aggregate (national accounts) vs regional (shift-share)
- **Trade theory**: international (Ricardian, H-O) vs interstate (gravity models for US states)
- **Fiscal policy**: federal multipliers vs state multipliers (smaller, open economies)
- **Tax competition**: international (BEPS) vs interstate (race to bottom)
- **Labor markets**: national Phillips curve vs state-level variation
- **Monetary policy**: uniform nationally but differential regional impact (manufacturing vs service states)

### Mathematical Depth Targets
- Optimization problems with full derivations (Lagrangian, KKT conditions)
- Equilibrium characterizations (existence, uniqueness, stability)
- Econometric estimators with asymptotic properties
- Simulation methods with algorithmic descriptions
- Calibration and estimation strategies for applied work
