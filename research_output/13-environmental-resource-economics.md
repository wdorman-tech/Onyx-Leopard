# Environmental & Resource Economics

## Overview

Environmental and resource economics applies microeconomic theory to the management of natural resources, pollution control, and climate change. This file covers carbon pricing design, resource extraction, commons governance, environmental valuation, climate policy, green finance, and state-level policy variation.

---

## 1. Carbon Pricing: Tax vs Cap-and-Trade

### 1.1 Weitzman (1974) Prices vs Quantities

Weitzman ("Prices vs. Quantities," *RES*, 1974): When marginal abatement costs are uncertain, the relative advantage of a price instrument (tax) vs quantity instrument (cap) depends on the slopes of the marginal benefit (MB) and marginal cost (MC) curves:

- If MC is steep and MB is flat → **tax is better** (quantity errors are costly; price errors are cheap)
- If MB is steep and MC is flat → **cap is better** (getting the quantity right matters more)

For climate change, MB of abatement is nearly flat (the marginal damage from one additional ton of CO₂ barely depends on the emission level in a given year because the stock is ~3 trillion tons). MC is uncertain and steep. **Weitzman's result favors a carbon tax for climate policy.** Nordhaus (2007) reached the same conclusion.

### 1.2 Carbon Tax Design

Optimal carbon tax = social cost of carbon (SCC). The SCC estimates the present value of all future damages from one additional ton of CO₂:

$$\text{SCC} = \int_0^\infty e^{-\delta t} \frac{\partial D(T(t))}{\partial E_0} dt$$

where $D$ is the damage function, $T$ is temperature, $E_0$ is current emissions, and $\delta$ is the discount rate. Estimates range from $50/ton (Nordhaus, $\delta \approx 3\%$) to $200+/ton (Stern, $\delta \approx 1.4\%$; EPA under Biden, $\delta = 2\%$, SCC ~$190). The discount rate is the dominant source of disagreement.

**Revenue recycling**: Carbon tax revenue can be recycled through (1) lump-sum dividends (carbon dividend, e.g., Canada's approach), (2) reducing distortionary taxes (the "double dividend" — Goulder 1995), or (3) funding clean energy R&D. The "strong double dividend" (carbon tax + labor tax cut improves welfare even ignoring environmental benefits) is contested; the "weak double dividend" (revenue recycling reduces the net cost of the carbon tax) is well-established.

### 1.3 Emissions Trading Systems

**EU ETS** (2005-): Cap-and-trade covering ~40% of EU emissions. Phases I-II suffered from overallocation (price collapsed to near zero). Phase III (2013+) introduced auctioning and the Market Stability Reserve (MSR) to manage surplus allowances. Phase IV (2021-2030) tightened the cap to 62% reduction by 2030. Carbon Border Adjustment Mechanism (CBAM) from 2026 prices embodied carbon in imports.

**RGGI** (Regional Greenhouse Gas Initiative, 2009-): First US cap-and-trade, covering power sector in 11 Northeast states. Modest cap with 100% auctioning. Allowance prices ~$13-15/ton. Revenue funds energy efficiency and clean energy. Estimated 45% reduction in power sector emissions 2009-2020.

**California Cap-and-Trade** (2013-): Economy-wide, linked with Quebec. Covers ~80% of state emissions. Price floor (~$22) and ceiling (~$75). Revenue funds high-speed rail, affordable housing, clean transportation. California also has a complementary Low Carbon Fuel Standard (LCFS) — a performance standard, not cap-and-trade.

---

## 2. Non-Renewable Resource Extraction: Hotelling Rule

### 2.1 Hotelling (1931)

Harold Hotelling ("The Economics of Exhaustible Resources," *JPE*, 1931) derived the fundamental principle: in competitive equilibrium, the price of an exhaustible resource net of extraction cost (the scarcity rent or Hotelling rent) must grow at the rate of interest:

$$\frac{d(p - c)}{dt} \cdot \frac{1}{p - c} = r$$

**Intuition**: The resource owner is indifferent between extracting a unit today (earning $p - c$, invested at rate $r$) and extracting tomorrow (earning the appreciated rent). If rent grew faster than $r$, all owners would delay extraction; if slower, all would extract immediately. In equilibrium, the two are equalized.

**Empirical performance**: Poor. Real commodity prices have not shown the consistent upward trend Hotelling predicts. Reasons: technological progress in extraction reduces $c$; discovery of new reserves increases the effective stock; backstop technologies cap the price. Krautkraemer (1998, *JEL*) surveys the evidence.

### 2.2 The Green Paradox (Sinn 2008)

Hans-Werner Sinn argued that *announcing* future climate policy (future carbon taxes, future bans on fossil fuels) accelerates current extraction: resource owners rush to sell before the policy bites, increasing near-term emissions. The "green paradox" — well-intentioned policy worsens the problem in the short run. The result depends on the timing and credibility of policy announcements and the extraction cost structure.

---

## 3. Renewable Resources: Fisheries

### 3.1 Gordon-Schaefer Model

The **Gordon (1954) - Schaefer (1957)** bioeconomic model of fisheries:

Stock growth: $\dot{X} = G(X) - H$ where $G(X) = rX(1 - X/K)$ is logistic growth (intrinsic rate $r$, carrying capacity $K$), and harvest $H = qEX$ (catchability $q$, effort $E$).

**Maximum Sustainable Yield (MSY)**: $H_{MSY} = rK/4$ at stock $X_{MSY} = K/2$.

**Open-access equilibrium**: Fishers enter until rent is zero: $pqX = c$ (price × catchability × stock = cost per unit effort). This occurs at $X_{OA} < X_{MSY}$ — open access leads to overfishing. The rent dissipation problem is a tragedy of the commons.

**Optimal management**: A sole owner maximizes the present value of the fishery: $\max \int_0^\infty e^{-\delta t}[pH - cE]dt$. The optimal steady state satisfies the **golden rule of renewable resources**:

$$G'(X^*) = \delta + \frac{c \cdot G(X^*)}{pqX^{*2} - cX^*/q}$$

At low discount rates, $X^* > X_{MSY}$ (conservation exceeds MSY). At very high discount rates, $X^* \to 0$ (extinction can be optimal if the resource is worth more harvested now than preserved).

---

## 4. Common-Pool Resources: Ostrom (Nobel 2009)

Elinor Ostrom (*Governing the Commons*, 1990) challenged the "tragedy of the commons" prediction (Hardin 1968) that common-pool resources inevitably degrade without privatization or government control. Through extensive field research, she documented communities successfully managing commons (irrigation systems in Spain, fisheries in Maine, forests in Nepal) through self-governance.

**Ostrom's 8 design principles** for successful commons governance:
1. Clearly defined boundaries
2. Rules match local conditions
3. Collective choice — affected parties participate in rule-making
4. Monitoring by accountable monitors
5. Graduated sanctions for rule violations
6. Accessible conflict-resolution mechanisms
7. Minimal recognition of rights to organize by external authorities
8. Nested enterprises for larger systems

**Formalization**: Ostrom's framework resists simple mathematical formalization because it emphasizes institutional heterogeneity and context-dependence. Game-theoretic models of self-governance (Sethi-Somanathan 1996) show that conditional cooperators + punishers can sustain cooperation in commons dilemmas without external enforcement, if punishment costs are low relative to cooperative gains.

---

## 5. Environmental Valuation

### 5.1 Contingent Valuation (CV)

Survey method: ask respondents their **willingness to pay (WTP)** for an environmental improvement or **willingness to accept (WTA)** compensation for a degradation. The NOAA Blue Ribbon Panel (Arrow et al. 1993) set guidelines for reliable CV: use referendum-format questions, include cheap talk to reduce hypothetical bias, remind respondents of budget constraints and substitute goods.

**Controversy**: Diamond-Hausman (1994) critique — CV measures "warm glow" (feeling good about helping the environment) rather than true economic value. WTP is insensitive to scope (WTP to save 2,000 birds ≈ WTP to save 200,000 birds — the "embedding effect"). Supporters argue scope sensitivity appears with careful design.

### 5.2 Hedonic Pricing

Rosen (1974) framework applied to environmental amenities: housing prices capitalize environmental quality. The implicit price of clean air:

$$\ln P = \alpha + \beta_1 X_1 + \cdots + \gamma \cdot \text{Pollution} + \epsilon$$

where $\gamma < 0$ measures the marginal WTP for pollution reduction. Chay and Greenstone (2005, *JPE*): Used Clean Air Act nonattainment designations as instruments; found TSP reductions capitalized into housing prices at ~$45-185 per μg/m³ per house.

### 5.3 Travel Cost Method

Value recreational sites by the cost visitors incur to reach them. Demand curve: higher travel cost → fewer visits. Consumer surplus from the site = area under the demand curve above the price line. Used for valuing national parks, beaches, wilderness areas. Assumes travel time has an opportunity cost (typically valued at 1/3 to 1/2 of the wage rate).

---

## 6. Cost-Benefit Analysis for Environmental Regulation

The basic CBA framework: a regulation is justified if $\sum_{t=0}^T \frac{B_t - C_t}{(1+r)^t} > 0$.

**Office of Management and Budget Circular A-4** (revised 2023): Federal agencies must conduct CBA for major regulations ($100M+ annual impact). Recommended discount rates: 2% (reflecting consumption rate of time preference). Distributional weights are permitted but not required.

**Value of a Statistical Life (VSL)**: The EPA uses VSL ~$11.6 million (2023 dollars) to value mortality risk reductions. Derived from hedonic wage studies (compensating differentials for risky jobs) and stated preference studies. VSL is not the "value of a life" but the marginal rate of substitution between wealth and mortality risk: $VSL = \frac{dW}{dp}$ where $p$ is the probability of death.

---

## 7. State-Level Environmental Policy Variation

US states exhibit enormous variation in environmental policy stringency:

**California**: Most aggressive state. CARB (California Air Resources Board) sets vehicle emission standards followed by ~15 other states under Clean Air Act Section 177. Cap-and-trade since 2013. 100% clean electricity target by 2045. LCFS. Offshore drilling bans.

**Texas**: Largest US emitter. No state carbon pricing. Minimal state-level emission standards beyond federal. However: largest wind power producer (>40 GW installed) driven by ERCOT market economics, not environmental regulation. Permitting reform has accelerated renewable deployment.

**RGGI states** (CT, DE, ME, MD, MA, NH, NJ, NY, PA pending, RI, VT, VA withdrew 2024): Regional cap-and-trade for power sector. Demonstrates interstate cooperation on climate without federal mandate.

**Environmental justice**: California's SB 535 (2012) directs 25% of cap-and-trade revenue to disadvantaged communities. EPA's EJScreen tool maps environmental burden by census tract. State-level EJ policies are rapidly expanding.

---

## 8. Green Finance and ESG

**Green bonds**: Fixed-income instruments where proceeds fund environmental projects. Global issuance >$500B/year by 2023. The "greenium" (green bonds trading at lower yields than conventional) is small (~2-8 basis points) but persistent, suggesting investors accept slightly lower returns for environmental alignment.

**ESG investing**: Environmental, Social, and Governance criteria integrated into investment decisions. Assets under ESG management ~$35 trillion globally. Academic evidence on ESG performance is mixed: Friede, Busch, and Bassen (2015, meta-analysis): majority of studies find non-negative relationship between ESG and financial performance. Pastor, Stambaugh, and Taylor (2021, *JFE*): ESG stocks have lower expected returns in equilibrium (investors accept lower returns for "green" holdings) but can outperform during transitions when ESG tastes strengthen.

---

## Key References

- Weitzman, M. (1974). Prices vs. Quantities. *RES*, 41(4), 477-491.
- Hotelling, H. (1931). The Economics of Exhaustible Resources. *JPE*, 39(2), 137-175.
- Nordhaus, W. (2017). Revisiting the Social Cost of Carbon. *PNAS*, 114(7), 1518-1523.
- Ostrom, E. (1990). *Governing the Commons*. Cambridge University Press.
- Gordon, H. S. (1954). The Economic Theory of a Common-Property Resource: The Fishery. *JPE*, 62(2), 124-142.
- Chay, K., Greenstone, M. (2005). Does Air Quality Matter? *JPE*, 113(2), 376-424.
- Goulder, L. (1995). Environmental Taxation and the Double Dividend. *Journal of Economic Literature*, 33(3), 1016-1065.
- Sinn, H.-W. (2008). Public Policies Against Global Warming: A Supply Side Approach. *ITAX*, 15(4), 360-394.
- Pastor, L., Stambaugh, R., Taylor, L. (2021). Sustainable Investing in Equilibrium. *JFE*, 142(2), 550-571.
