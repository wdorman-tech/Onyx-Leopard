# Health Economics & Insurance Markets

## Overview

Health economics studies the production, distribution, and consumption of healthcare — a sector with pervasive market failures. This file covers Arrow's foundational analysis, insurance market design, pharmaceutical economics, cost-effectiveness, hospital competition, and US-specific institutional details.

---

## 1. Healthcare Market Failures: Arrow (1963)

Kenneth Arrow ("Uncertainty and the Welfare Economics of Medical Care," *AER*, 1963) identified why healthcare markets deviate from the competitive ideal:

1. **Uncertainty**: Illness is unpredictable; demand for healthcare is stochastic, creating demand for insurance.
2. **Asymmetric information**: Patients cannot evaluate treatment quality; they rely on physician judgment (agency relationship). Physicians have financial incentives that may conflict with patient interests (supplier-induced demand).
3. **Barriers to entry**: Medical licensing, lengthy training, and hospital accreditation limit supply.
4. **Externalities**: Infectious disease treatment benefits others (vaccination externalities).
5. **Non-marketable goods**: Trust, the physician-patient relationship, and professional ethics are not priced.

Arrow's paper established that healthcare markets inherently fail and that some form of institutional intervention (insurance, regulation, professional norms) is necessary — the market cannot be left to its own devices.

---

## 2. Insurance Market Theory

### 2.1 Rothschild-Stiglitz (1976)

Rothschild and Stiglitz ("Equilibrium in Competitive Insurance Markets," *QJE*, 1976) showed that competitive insurance markets with adverse selection may not have an equilibrium, and when they do, it involves **separating contracts**:

- **High-risk types** get full insurance at actuarially fair (high) premiums.
- **Low-risk types** get partial insurance (deductibles, copays) at lower premiums — the partial coverage is necessary to prevent high-risk types from mimicking low-risk types.

If the proportion of low-risk types is high enough, no equilibrium exists — the separating contracts are dominated by a pooling contract, but the pooling contract is itself dominated by cream-skimming.

### 2.2 Moral Hazard in Health Insurance

The **RAND Health Insurance Experiment** (Manning et al. 1987): The gold-standard RCT of health insurance. Randomly assigned families to plans with coinsurance rates from 0% (free care) to 95%. Findings:
- Free care increased healthcare spending by ~30% relative to 95% coinsurance.
- Most additional spending on free care was for services of low clinical value.
- Health outcomes were similar across plans for most people; free care improved outcomes for the poorest and sickest.

The RAND HIE established the price elasticity of demand for healthcare at approximately -0.2 (a 10% price increase reduces utilization ~2%). This is the empirical foundation for insurance design with cost-sharing.

### 2.3 Optimal Insurance Design

Zeckhauser (1970, *AER*): The optimal coinsurance rate balances risk protection (lower coinsurance = better insurance) against moral hazard (lower coinsurance = more wasteful utilization). With CRRA utility and moral hazard:

$$c^* = \frac{\eta}{r + \eta}$$

where $\eta$ is the demand elasticity and $r$ is the coefficient of relative risk aversion. Higher risk aversion → lower optimal coinsurance (more generous insurance). Higher demand elasticity → higher optimal coinsurance (more cost-sharing needed to control moral hazard).

**Value-Based Insurance Design (VBID)**: Chernew, Rosen, and Fendrick (2007): Set cost-sharing based on clinical value rather than cost. High-value services (statins for heart disease, insulin for diabetes) get low or zero copays; low-value services get high copays. Implemented in many employer plans and in Medicare Advantage.

---

## 3. Pharmaceutical Economics

### 3.1 R&D Incentives and Patent Protection

Drug development costs ~$1-2 billion per approved drug (DiMasi et al. 2016), with ~90% failure rate in clinical trials. Patents provide 20-year exclusivity from filing, but effective market exclusivity is ~10-12 years after FDA approval. The trade-off: patents create temporary monopoly pricing (static inefficiency) to incentivize R&D investment (dynamic efficiency).

### 3.2 Generic Entry and Hatch-Waxman (1984)

The Drug Price Competition and Patent Term Restoration Act (1984) created the modern generic drug framework:
- **ANDA (Abbreviated New Drug Application)**: Generics demonstrate bioequivalence, not clinical efficacy. Dramatically reduces entry costs.
- **Paragraph IV certification**: Generic firms can challenge patents before expiration. The first successful generic challenger gets 180-day exclusivity.
- **Patent term restoration**: Compensates brand firms for time lost during FDA review.

Generic entry typically reduces prices to 10-20% of brand price within 2 years when 5+ generic competitors enter. For complex molecules (biologics), the Biologics Price Competition and Innovation Act (2009) created a "biosimilar" pathway, but biosimilar uptake has been slower due to higher manufacturing complexity and physician switching concerns.

### 3.3 International Reference Pricing

Most OECD countries use external reference pricing: set domestic drug prices based on prices in comparator countries. The US is the only major OECD country without systematic government price negotiation for drugs. The Inflation Reduction Act (2022) authorized Medicare to negotiate prices for select high-spend drugs — the first US foray into direct government drug price negotiation.

**Ramsey pricing for drugs** (Danzon 1997): Efficient price discrimination across countries with different WTP: charge higher prices in rich countries, lower in poor countries, with markups inversely proportional to demand elasticity. This is approximately what happens globally, though the degree of discrimination is constrained by parallel trade risks.

---

## 4. Cost-Effectiveness Analysis

### 4.1 QALYs and ICERs

A **Quality-Adjusted Life Year (QALY)** weights life years by health quality: 1 QALY = 1 year in perfect health; 0 = death. Health states are valued between 0 and 1 using EQ-5D, SF-6D, or other instruments.

The **Incremental Cost-Effectiveness Ratio (ICER)**:

$$\text{ICER} = \frac{C_{\text{new}} - C_{\text{comparator}}}{Q_{\text{new}} - Q_{\text{comparator}}}$$

A treatment is cost-effective if ICER < the willingness-to-pay threshold $\lambda$. **NICE** (UK) uses $\lambda \approx$ £20,000-30,000/QALY. **ICER** (US) uses $50,000-150,000/QALY depending on context. The WHO recommends $\lambda = 1-3 \times$ GDP per capita.

### 4.2 Controversy

**Disability rights critique**: QALYs value life years of disabled people less, creating discrimination. The ACA prohibits the use of QALYs by CMS for coverage decisions. ICER uses QALYs but the restriction limits their adoption in US policy.

---

## 5. Grossman (1972) Health Capital Model

Michael Grossman (*The Demand for Health*, 1972) modeled health as a durable capital stock that depreciates with age:

$$H_{t+1} = (1-\delta_t)H_t + I_t$$

where $H_t$ is health capital, $\delta_t$ is the age-dependent depreciation rate, and $I_t$ is health investment (medical care, exercise, diet). Individuals maximize lifetime utility $\sum \beta^t U(C_t, H_t)$ subject to time and budget constraints. Health is both a consumption good (provides direct utility) and an investment good (healthy days enable work).

**Predictions**: (1) Health investment increases with wage rates (higher opportunity cost of sick time). (2) Education increases the productivity of health investment (more educated people are more efficient health producers). (3) Health depreciates faster with age, so older individuals invest more in healthcare but have lower health stocks. These predictions are broadly confirmed empirically.

---

## 6. Hospital Competition and Mergers

### 6.1 Competition and Quality

Kessler and McClellan (2000, *QJE*): In areas with more hospital competition (measured by HHI), Medicare heart attack patients have lower costs and lower mortality. Competition improves both efficiency and quality. Gaynor, Moreno-Serra, and Propper (2013, *AER*): UK NHS hospital reforms introducing patient choice and competition reduced mortality with no increase in costs.

### 6.2 Merger Effects

Hospital mergers reduce competition and raise prices. Gaynor and Town (2012, *Handbook of Health Economics*) survey: mergers in concentrated markets increase prices 20-40% with no improvement in quality. The FTC has increasingly challenged hospital mergers, winning several cases (Advocate-NorthShore 2017, Geisinger-Evangelical 2024).

---

## 7. US Healthcare System Specifics

### 7.1 Medicaid Expansion

The ACA (2010) expanded Medicaid eligibility to 138% FPL. The Supreme Court (NFIB v. Sebelius, 2012) made expansion optional for states. As of 2024, 40 states + DC have expanded. Expansion states show: reduced uninsured rates (~30-50% reduction), improved financial protection (reduced medical debt), improved health outcomes (Sommers et al. 2017, *Annals of Internal Medicine*), with federal government covering 90% of costs.

### 7.2 State Variation in Costs and Outcomes

The Dartmouth Atlas documents enormous geographic variation: per-capita Medicare spending varies 2-3x across regions (Miami: ~$16,000; Minneapolis: ~$7,000) with no corresponding quality difference. Fisher et al. (2003): higher spending regions do not have better outcomes — the variation reflects practice style differences, not patient needs. This "flat of the curve" phenomenon suggests substantial waste.

### 7.3 Certificate of Need (CON)

35 states require healthcare providers to obtain a Certificate of Need before expanding capacity (new hospitals, beds, expensive equipment). Intended to prevent overinvestment and cost inflation. Evidence: CON laws reduce the number of hospitals and beds but do not reduce costs — they create barriers to entry that protect incumbents and may reduce access (Stratmann-Koopman 2016).

---

## 8. Pandemic Economics

### 8.1 SIR-Macro Models

Eichenbaum, Rebelo, and Trabandt (2021, *AER*): Integrated the epidemiological SIR (Susceptible-Infected-Recovered) model with a macroeconomic model. Rational agents reduce economic activity to lower infection risk, creating an endogenous recession even without government mandates:

$$U = \sum_{t=0}^\infty \beta^t [u(c_t, n_t) - \theta \pi_{St} \cdot (c_t n_t + \text{contacts})]$$

where $\pi_{St}$ is the probability of infection for susceptible individuals. The competitive equilibrium involves too much economic activity (individuals don't internalize that their infection infects others — an externality), justifying containment policies. The optimal lockdown is front-loaded and gradually relaxed.

---

## Key References

- Arrow, K. J. (1963). Uncertainty and the Welfare Economics of Medical Care. *AER*, 53(5), 941-973.
- Rothschild, M., Stiglitz, J. (1976). Equilibrium in Competitive Insurance Markets. *QJE*, 90(4), 629-649.
- Manning, W. et al. (1987). Health Insurance and the Demand for Medical Care. *AER*, 77(3), 251-277.
- Grossman, M. (1972). On the Concept of Health Capital and the Demand for Health. *JPE*, 80(2), 223-255.
- Gaynor, M., Town, R. (2012). Competition in Health Care Markets. *Handbook of Health Economics*, Vol. 2.
- Eichenbaum, M., Rebelo, S., Trabandt, M. (2021). The Macroeconomics of Epidemics. *RES*, 88(3), 1319-1370.
- DiMasi, J., Grabowski, H., Hansen, R. (2016). Innovation in the Pharmaceutical Industry. *Journal of Health Economics*, 47, 20-33.
