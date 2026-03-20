# Financial Risk Management & Systemic Risk

## Overview

This file covers the mathematical frameworks for measuring, managing, and regulating financial risk at the firm and system level. It extends the source document's coverage of BSM, stochastic volatility, CAPM, and factor models into applied risk management, systemic risk measurement, and financial regulation.

---

## 1. Value-at-Risk (VaR)

VaR answers: "What is the maximum loss over a given horizon at a given confidence level?" Formally, VaR at confidence level $\alpha$ is the $\alpha$-quantile of the loss distribution:

$$\text{VaR}_\alpha = \inf\{x : P(L > x) \leq 1 - \alpha\}$$

### 1.1 Historical Simulation

Use the empirical distribution of past portfolio returns. Sort $T$ historical returns; VaR at 99% is the $\lfloor T \times 0.01 \rfloor$-th worst return. No distributional assumptions. Weaknesses: limited by historical sample length; equal weighting of all past observations; cannot capture structural changes.

### 1.2 Parametric (Variance-Covariance)

Assume portfolio returns $R_p \sim N(\mu_p, \sigma_p^2)$. Then:

$$\text{VaR}_\alpha = -(\mu_p + z_\alpha \sigma_p)$$

where $z_\alpha = \Phi^{-1}(\alpha)$. For a portfolio with weight vector $w$ and covariance matrix $\Sigma$: $\sigma_p = \sqrt{w'\Sigma w}$. **RiskMetrics** (J.P. Morgan, 1996) uses EWMA to estimate $\Sigma$: $\sigma_{t}^2 = \lambda \sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2$ with $\lambda = 0.94$ (daily). Fast but assumes normality — underestimates tail risk.

### 1.3 Monte Carlo VaR

Simulate $N$ paths of risk factors from a fitted model (multivariate normal, t-copula, stochastic volatility), reprice the portfolio on each path, compute the empirical quantile of simulated P&L. Most flexible — handles nonlinear positions (options), non-normal distributions, path-dependent instruments. Computationally intensive.

---

## 2. Expected Shortfall (CVaR)

VaR tells you the threshold but nothing about losses beyond it. **Expected Shortfall** (Conditional VaR, Acerbi and Tasche 2002):

$$\text{ES}_\alpha = E[L \mid L > \text{VaR}_\alpha] = \frac{1}{1-\alpha}\int_\alpha^1 \text{VaR}_u \, du$$

ES is the average loss in the worst $(1-\alpha)$ scenarios. For normal distributions: $\text{ES}_\alpha = \mu + \sigma \cdot \frac{\phi(z_\alpha)}{1-\alpha}$.

Basel III (2016 Fundamental Review of the Trading Book) replaced VaR with ES at 97.5% for market risk capital, recognizing ES better captures tail risk.

---

## 3. Coherent Risk Measures

Artzner, Delbaen, Eber, and Heath (1999, *Mathematical Finance*) defined four axioms for a "coherent" risk measure $\rho$:

1. **Translation invariance**: $\rho(X + c) = \rho(X) - c$
2. **Subadditivity**: $\rho(X + Y) \leq \rho(X) + \rho(Y)$ (diversification reduces risk)
3. **Positive homogeneity**: $\rho(\lambda X) = \lambda \rho(X)$ for $\lambda > 0$
4. **Monotonicity**: If $X \leq Y$ a.s., then $\rho(X) \geq \rho(Y)$

**VaR violates subadditivity** for non-elliptical distributions — combining positions can increase VaR, perversely penalizing diversification. **ES is coherent.** This theoretical superiority drove the Basel III shift.

---

## 4. Copulas in Finance

A copula $C: [0,1]^n \to [0,1]$ captures the dependence structure between random variables independently of their marginal distributions. By **Sklar's theorem** (1959), any joint distribution can be written:

$$F(x_1, \ldots, x_n) = C(F_1(x_1), \ldots, F_n(x_n))$$

### 4.1 Gaussian Copula

$$C(u_1, \ldots, u_n) = \Phi_n(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_n); R)$$

where $\Phi_n$ is the multivariate normal CDF with correlation matrix $R$. **Li (2000)** applied the Gaussian copula to price CDOs by modeling default dependence. The correlation parameter $\rho$ was calibrated to market spreads.

**The CDO crisis (2007-2008):** The Gaussian copula critically underestimates joint tail events — its tail dependence is zero (extreme losses are asymptotically independent). When housing markets fell simultaneously across regions, correlated defaults far exceeded Gaussian copula predictions. Tranche losses were catastrophic. The model's failure was a central technical cause of the 2008 crisis (Salmon 2009, "The Formula That Killed Wall Street").

### 4.2 t-Copula and Alternatives

The Student-t copula has positive tail dependence: $\lambda = 2t_{\nu+1}(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)})$, capturing the tendency for joint extreme events. Clayton copula has lower-tail dependence (joint crashes); Gumbel has upper-tail dependence. Selecting the copula family is a modeling choice with enormous practical implications.

---

## 5. Credit Risk Models

### 5.1 Merton (1974) Structural Model

Firm value $V_t$ follows GBM: $dV = \mu V dt + \sigma_V V dW$. The firm has debt with face value $D$ maturing at $T$. Default occurs if $V_T < D$. Equity is a call option on firm value:

$$E = V \cdot N(d_1) - D e^{-rT} N(d_2)$$

where $d_1, d_2$ are as in BSM with $V$ replacing $S$ and $D$ replacing $K$. The probability of default is $N(-d_2)$. **KMV/Moody's Analytics** operationalized this: estimate $V$ and $\sigma_V$ from equity price and volatility, compute "distance to default" $DD = (\ln(V/D) + (\mu - \sigma_V^2/2)T) / (\sigma_V\sqrt{T})$, map DD to empirical default frequency.

### 5.2 Reduced-Form Models

**Jarrow-Turnbull (1995)**: Default is modeled as the first jump of a Poisson process with intensity $\lambda(t)$. Survival probability: $Q(t) = \exp(-\int_0^t \lambda(s) ds)$. Bond price: $P = \int_0^T e^{-rt}[-dQ(t)] \cdot R + e^{-rT}Q(T)$ where $R$ is recovery rate.

**Duffie-Singleton (1999)**: Price defaultable bonds by discounting at the risk-free rate plus a credit spread: $P = E^Q[e^{-\int_0^T (r_s + \lambda_s(1-R)) ds}]$. The intensity $\lambda$ can depend on macroeconomic state variables, producing time-varying credit spreads.

---

## 6. Systemic Risk Measures

### 6.1 CoVaR (Adrian and Brunnermeier, 2016, AER)

CoVaR measures the VaR of the financial system conditional on a specific institution being in distress:

$$P(R_{\text{system}} \leq \text{CoVaR}_q^{i|C(R_i)} \mid C(R_i)) = q$$

where $C(R_i)$ is a conditioning event (e.g., institution $i$ at its VaR). The **contribution to systemic risk**: $\Delta\text{CoVaR}_q^i = \text{CoVaR}_q^{i|\text{distress}} - \text{CoVaR}_q^{i|\text{median}}$. Estimated via quantile regression of system returns on institution returns and state variables.

### 6.2 SRISK (Acharya, Engle, Richardson, 2012; Brownlees and Engle, 2017)

SRISK is the expected capital shortfall of a firm conditional on a systemic crisis:

$$\text{SRISK}_i = E[k \cdot A_i - E_i \mid \text{crisis}]$$

where $k$ is the prudential capital ratio (~8%), $A_i$ is assets, $E_i$ is equity. Using the Long-Run Marginal Expected Shortfall (LRMES — the expected equity decline in a crisis), $\text{SRISK}_i = k \cdot D_i - (1 - k)(1 - \text{LRMES}_i) \cdot E_i$. LRMES is estimated from a DCC-GARCH model. SRISK aggregated across firms gives the total recapitalization need in a crisis. The NYU Stern V-Lab publishes real-time SRISK rankings.

### 6.3 Network-Based Systemic Risk

**Interbank network models**: Banks lend to each other; default by one bank reduces the assets of its creditors, potentially triggering cascading defaults. Eisenberg and Noe (2001, *Management Science*) formalized clearing in financial networks: given a network of interbank liabilities, the clearing payment vector is the fixed point of a monotone operator, guaranteed to exist by Tarski's theorem.

**Acemoglu, Ozdaglar, and Tahbaz-Salehi (2015, AER)**: For small shocks, dense networks absorb shocks (diversification). For large shocks exceeding a critical threshold, dense networks transmit shocks system-wide. The "robust-yet-fragile" property: the same interconnectedness that stabilizes against small shocks amplifies large ones.

---

## 7. Stress Testing

### 7.1 Dodd-Frank Act Stress Tests (DFAST) / CCAR

U.S. bank holding companies with >$100B assets undergo annual stress tests. The Federal Reserve specifies macroeconomic scenarios (baseline, adverse, severely adverse — e.g., GDP decline of 8%, unemployment at 10%, equity decline of 55%). Banks project losses, revenues, and capital ratios under each scenario using internal models. Must maintain CET1 ratio above 4.5% under severely adverse.

### 7.2 European Banking Authority (EBA)

EU-wide stress tests cover ~50 banks representing ~70% of EU banking assets. Bottom-up approach: banks use their own models with EBA-specified scenarios. Key difference from US: EBA uses a static balance sheet assumption (no management actions), producing more severe capital depletions.

**Critique**: Stress tests are only as good as their scenarios. The 2008 crisis was more severe than any pre-2008 stress scenario. Greenlaw et al. (2012): stress tests may create false confidence; they measure resilience to known risks, not unknown unknowns.

---

## 8. Liquidity Risk

### 8.1 Brunnermeier and Pedersen (2009, RFS): Funding vs Market Liquidity

Funding liquidity (a trader's ability to finance positions) and market liquidity (the ease of trading without price impact) are mutually reinforcing in spirals:

1. Losses reduce trader capital → tighter funding constraints.
2. Funding constraints force position liquidation → selling pressure.
3. Selling pressure reduces market liquidity → wider bid-ask spreads, higher margins.
4. Higher margins further tighten funding → return to step 1.

This **liquidity spiral** amplifies small shocks into crises. In equilibrium, multiple states exist: a liquid state (low margins, narrow spreads, ample funding) and an illiquid state (high margins, wide spreads, funding stress). The transition between states is sudden and self-reinforcing, matching the 2007-2008 crisis dynamics.

---

## 9. Basel Framework Evolution

**Basel I (1988)**: Flat 8% capital ratio; crude risk weights (0% for OECD sovereigns, 50% for mortgages, 100% for corporates).

**Basel II (2004)**: Three pillars — (1) minimum capital with risk-sensitive weights (internal ratings-based approach), (2) supervisory review, (3) market discipline. Introduced operational risk capital. The IRB formula for credit risk capital:

$$K = \text{LGD} \cdot [N(\frac{N^{-1}(PD) + \sqrt{\rho} N^{-1}(0.999)}{\sqrt{1-\rho}}) - PD]$$

where $\rho$ is the asset correlation parameter. This is a Vasicek (2002) one-factor model: all firms share a common systematic factor.

**Basel III (2010-2017)**: Response to the 2008 crisis. Higher capital (CET1 minimum 4.5% + 2.5% conservation buffer), leverage ratio (3% minimum), liquidity coverage ratio (LCR — sufficient HQLA for 30-day stress), net stable funding ratio (NSFR), countercyclical capital buffer (0-2.5%), G-SIB surcharges (1-3.5%).

**Basel III.1 / "Endgame" (2023+)**: Revised standardized approaches for market risk (FRTB replacing VaR with ES), credit risk, operational risk; output floor (72.5% of standardized approach). Controversial in the US: implementation debated 2023-2025.

---

## 10. The 2008 Financial Crisis: A Risk Management Post-Mortem

**Root causes through a risk lens:**
- Gaussian copula mispricing of CDO correlation risk (tail dependence = 0 assumption).
- VaR-based risk limits that ignored fat tails and liquidity risk.
- Procyclical leverage: Basel II's risk-sensitive weights amplified the boom (low measured risk → low capital → more lending) and the bust (high measured risk → capital shortfall → fire sales).
- Counterparty risk concentration (AIG wrote $450B of CDS without adequate reserves).
- Liquidity spiral (Brunnermeier-Pedersen): funding markets froze, forcing deleveraging that further depressed asset prices.

**Lehman Brothers**: September 15, 2008. $619B in assets, leverage ratio ~30:1. Filed Chapter 11 after failing to find a buyer. The bankruptcy triggered CDS settlement cascades, money market fund "breaking the buck" (Reserve Primary Fund), and a global credit freeze. VaR models at Lehman had shown the firm's risk as manageable; the models didn't capture the illiquidity of its MBS/CDO portfolio or the counterparty contagion from its failure.

---

## Key References

- Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent Measures of Risk. *Mathematical Finance*, 9(3), 203-228.
- Adrian, T., Brunnermeier, M. (2016). CoVaR. *AER*, 106(7), 1705-1741.
- Brownlees, C., Engle, R. (2017). SRISK: A Conditional Capital Shortfall Measure of Systemic Risk. *RFS*, 30(1), 48-79.
- Acemoglu, D., Ozdaglar, A., Tahbaz-Salehi, A. (2015). Systemic Risk and Stability in Financial Networks. *AER*, 105(2), 564-608.
- Brunnermeier, M., Pedersen, L. (2009). Market Liquidity and Funding Liquidity. *RFS*, 22(6), 2201-2238.
- Eisenberg, L., Noe, T. (2001). Systemic Risk in Financial Systems. *Management Science*, 47(2), 236-249.
- Li, D. (2000). On Default Correlation: A Copula Function Approach. *Journal of Fixed Income*, 9(4), 43-54.
- Merton, R. (1974). On the Pricing of Corporate Debt. *Journal of Finance*, 29(2), 449-470.
- Duffie, D., Singleton, K. (1999). Modeling Term Structures of Defaultable Bonds. *RFS*, 12(4), 687-720.
- Basel Committee on Banking Supervision (2017). Basel III: Finalising Post-Crisis Reforms.
