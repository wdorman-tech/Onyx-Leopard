# Labor Economics & Human Capital Theory

## Overview

Labor economics studies the determination of wages, employment, and the allocation of workers to jobs. This file covers search and matching theory, human capital, wage determination, inequality, minimum wages, immigration, automation, discrimination, and state-level labor market variation.

---

## 1. Search and Matching: Diamond-Mortensen-Pissarides (Nobel 2010)

### 1.1 The Matching Function

The labor market is characterized by frictions: workers searching for jobs and firms searching for workers cannot find each other instantly. The **matching function** $M(U, V)$ maps the number of unemployed workers $U$ and vacancies $V$ to the flow of new matches. Typically Cobb-Douglas:

$$M(U, V) = m_0 U^\eta V^{1-\eta}$$

where $\eta \in (0,1)$ is the elasticity of matching with respect to unemployment. Define labor market tightness $\theta = V/U$. The job-finding rate for workers: $f(\theta) = M/U = m_0 \theta^{1-\eta}$, increasing in tightness. The vacancy-filling rate for firms: $q(\theta) = M/V = m_0 \theta^{-\eta}$, decreasing in tightness.

### 1.2 The Beveridge Curve

The **Beveridge curve** plots the negative relationship between the unemployment rate $u$ and the vacancy rate $v$ in steady state. In steady state, separations equal matches: $s(1-u) = f(\theta)u$, giving:

$$u = \frac{s}{s + f(\theta)}$$

An outward shift of the Beveridge curve indicates reduced matching efficiency (structural mismatch). The US Beveridge curve shifted outward after 2009 and again after 2020, suggesting labor market mismatch.

### 1.3 Wage Bargaining: Nash Bargaining Solution

Once a worker and firm match, they split the match surplus via Nash bargaining:

$$w = \arg\max (W(w) - U)^\beta (J(w) - V)^{1-\beta}$$

where $W$ is the worker's value of employment, $U$ is the value of unemployment, $J$ is the firm's value of a filled job, $V$ is the value of a vacancy, and $\beta$ is worker bargaining power. Solution:

$$w = (1-\beta)z + \beta(p + c\theta)$$

where $z$ is the value of non-employment (UI benefits + leisure), $p$ is productivity, $c$ is vacancy posting cost, and $\theta$ is tightness. Workers get a share $\beta$ of productivity plus the expected cost the firm would incur to find another worker.

### 1.4 The DMP Model Equilibrium

Three equations determine $(w, \theta, u)$:
1. **Wage curve**: $w = (1-\beta)z + \beta(p + c\theta)$
2. **Job creation**: $\frac{c}{q(\theta)} = \frac{p - w}{r + s}$ (expected hiring cost = present value of profits)
3. **Beveridge curve**: $u = s/(s + f(\theta))$

**Shimer (2005, AER) puzzle**: The baseline DMP model generates insufficient unemployment volatility — productivity shocks produce large wage fluctuations but small employment fluctuations (wages absorb the shock). Fixes: wage rigidity (Hall 2005), alternating-offer bargaining (Hall-Milgrom 2008).

---

## 2. Human Capital Theory

### 2.1 Becker (1964)

Gary Becker (*Human Capital*, 1964, Nobel 1992) formalized education and training as investment in productive capacity. Workers invest in human capital when the present value of increased earnings exceeds the cost (direct + opportunity):

$$\sum_{t=0}^{T} \frac{w_t^{educated} - w_t^{uneducated}}{(1+r)^t} > C_{education}$$

**General vs specific human capital**: General training (transferable across firms) is financed by workers (through lower wages during training) because competitive firms cannot capture its returns. Specific training (valuable only at the current firm) creates bilateral monopoly — the surplus is shared, producing wage profiles above marginal product early and below later.

### 2.2 Mincer Earnings Equation (1974)

Jacob Mincer proposed the canonical log-linear earnings regression:

$$\ln w_i = \alpha + \beta_1 S_i + \beta_2 X_i + \beta_3 X_i^2 + \epsilon_i$$

where $S$ is years of schooling, $X$ is potential experience (age - schooling - 6). $\beta_1$ is the rate of return to education (~6-10% per year in the US); the concave experience profile reflects on-the-job human capital accumulation with diminishing returns. Heckman, Lochner, and Todd (2006, *Handbook of Economics of Education*): the Mincer equation has been remarkably stable over decades, though the return to education has risen substantially since 1980.

### 2.3 Signaling vs Human Capital

**Spence (1973)**: Education may signal innate ability rather than build productivity. Causal identification of the human capital effect requires instruments for schooling:
- **Card (1995)**: proximity to college as IV → returns ~10-13%, above OLS (~7%), suggesting ability bias makes OLS an underestimate (the marginal student gaining access has high returns).
- **Angrist-Krueger (1991)**: quarter of birth as IV for compulsory schooling → ~7% return.
- **Twin studies** (Ashenfelter-Krueger 1994): within-twin estimates ~8-10%.

The consensus: both human capital and signaling effects exist; the relative share is contested. Caplan (*The Case Against Education*, 2018) argues signaling dominates.

---

## 3. Wage Inequality and Skill-Biased Technical Change

### 3.1 The College Wage Premium

The college/high-school wage ratio rose from ~1.4 in 1980 to ~1.9 by 2020 in the US. Katz and Murphy (1992, *QJE*): modeled the premium as the outcome of a race between demand (SBTC shifting demand toward skilled workers) and supply (college graduation rates). The premium rises when demand growth outpaces supply growth. The slowdown in college attainment growth after 1980 explains much of the rising premium.

**Goldin and Katz (2008, *The Race Between Education and Technology*)**: Comprehensive historical account. The US led the world in mass education in the early 20th century, compressing the skill premium. The deceleration post-1980 reversed this, widening inequality. The "Great Compression" (1940s-1970s) and subsequent "Great Divergence" are supply-driven phenomena.

### 3.2 Task-Based Framework: Autor-Levy-Murnane (2003)

Autor, Levy, and Murnane (2003, *QJE*) decomposed jobs into tasks:
- **Routine cognitive** (bookkeeping, data entry): easily automated
- **Routine manual** (assembly line): automatable
- **Non-routine cognitive/analytical** (management, programming): complemented by computers
- **Non-routine cognitive/interpersonal** (negotiation, teaching): hard to automate
- **Non-routine manual** (janitorial, personal care): hard to automate, low-wage

Computerization substitutes for routine tasks and complements non-routine cognitive tasks. This produces **job polarization**: employment growth at the top (professional/managerial) and bottom (service), hollowing out the middle (production, clerical). Autor and Dorn (2013, *AER*) confirmed polarization empirically using commuting zone data.

---

## 4. Minimum Wage Economics

### 4.1 Competitive Model

In a competitive labor market, a binding minimum wage $w_{min} > w^*$ (market-clearing wage) creates unemployment: $L^s(w_{min}) > L^d(w_{min})$. The disemployment effect depends on labor demand elasticity. Estimates of the employment elasticity with respect to the minimum wage: -0.1 to -0.3 (Neumark-Wascher 2008 survey) for teens and low-skilled workers.

### 4.2 Card-Krueger (1994): The New Jersey Experiment

Card and Krueger (1994, *AER*) compared fast-food employment in NJ (which raised its minimum wage from $4.25 to $5.05) to eastern PA (which did not). Finding: employment in NJ fast-food restaurants *increased* relative to PA. This contradicted the competitive model and sparked intense debate.

### 4.3 Monopsony Power

If employers have wage-setting power (monopsony), a minimum wage below the competitive wage can *increase* employment. In the monopsony model, the firm faces upward-sloping labor supply $w = w(L)$. The marginal cost of labor $MCL = w + L \cdot dw/dL > w$. The monopsonist hires where $MCL = MRP_L$, resulting in lower employment and wages than competitive. A minimum wage $w_{min}$ between $w_{monopsony}$ and $w_{competitive}$ increases employment.

**Manning (2003, *Monopsony in Motion*)**: Labor markets exhibit substantial monopsony power due to search frictions, mobility costs, and employer differentiation. Dube, Lester, and Reich (2010, *ReStat*) used border discontinuities to estimate minimum wage effects, finding minimal disemployment and significant wage gains, consistent with monopsony.

---

## 5. Immigration Economics

### 5.1 Borjas Framework

George Borjas ("The Labor Demand Curve Is Downward Sloping," *QJE*, 2003): Treats immigrants and natives within education-experience cells as perfect substitutes. Using national variation in immigrant inflows across cells, estimates a wage elasticity of -0.3 to -0.4: a 10% increase in labor supply reduces wages 3-4%. The Mariel boatlift reanalysis (Borjas 2017) found large negative wage effects on low-skill workers.

### 5.2 Card Framework

David Card (1990, *ILRR*): The original Mariel boatlift study found no significant effect on Miami wages or unemployment from the 1980 influx of 125,000 Cuban workers (a 7% labor supply increase). Card argues local labor markets absorb immigrants through: (1) expansion of immigrant-intensive industries, (2) out-migration of natives, (3) capital adjustment.

**Reconciliation**: Peri and Sparber (2009, *AEJ: Applied*): immigrants and natives within skill cells specialize in different tasks (immigrants in manual tasks, natives in communication tasks), limiting direct substitution. The cross-cell complementarity can produce positive wage effects for natives in some skill groups.

### 5.3 State-Level Variation

Immigration effects vary enormously by state. California, Texas, Florida, and New York absorb the majority of immigrants. States with large immigrant inflows and elastic housing supply (Texas) accommodate growth better than those with inelastic supply (California). State-level minimum wage interactions with immigration also matter: high minimum wages may reduce immigrant employment in low-skill sectors.

---

## 6. Automation, AI, and Labor Markets

### 6.1 Acemoglu-Restrepo Framework (2018, 2020)

Acemoglu and Restrepo ("Robots and Jobs," *JPE*, 2020) formalize the displacement and reinstatement effects of automation:

$$\text{Net labor demand} = \underbrace{-\text{Displacement}}_{\text{tasks automated}} + \underbrace{\text{Reinstatement}}_{\text{new tasks for labor}} + \underbrace{\text{Productivity}}_{\text{growth raises demand}}$$

A task-based model with a continuum of tasks $[N-1, N]$: tasks below threshold $I$ are automated, tasks above $I$ use labor. Automation increases $I$ (displacement). New task creation increases $N$ (reinstatement). If displacement outpaces reinstatement, the labor share falls. Empirical estimate: one robot per 1,000 workers reduces employment-to-population ratio by 0.2 percentage points and wages by 0.42%.

### 6.2 AI and the Future of Work

Eloundou, Manning, Mishkin, Rock (2023): GPT models can affect ~80% of the US workforce through at least 10% of their tasks; ~19% of workers have at least 50% of tasks exposed. Higher-wage, more educated workers are *more* exposed to LLM automation — reversing the historical pattern of automation primarily displacing low-skill workers.

---

## 7. Labor Market Discrimination

### 7.1 Becker (1957): Taste-Based Discrimination

Employers with discriminatory tastes $d$ act as if minority workers cost $w(1+d)$. In a competitive market, discrimination is costly to the discriminator — non-discriminating firms hire undervalued minority workers and earn higher profits. Long-run prediction: competition eliminates discrimination. Empirically: discrimination persists, suggesting market imperfections.

### 7.2 Statistical Discrimination (Phelps 1972, Arrow 1973)

Employers use group membership as a signal when individual productivity is hard to observe. If group A has noisier productivity signals than group B, risk-averse employers demand a risk premium from group A, even if average productivity is identical. This produces rational but discriminatory outcomes that are self-reinforcing: lower wages reduce group A's incentive to invest in unobservable skills.

### 7.3 Audit Studies

Bertrand and Mullainathan (2004, *AER*): Sent identical resumes with "white-sounding" names (Emily, Greg) vs "African-American-sounding" names (Lakisha, Jamal). White names received 50% more callbacks. This measures discrimination at the hiring stage directly, controlling for productivity.

---

## 8. Efficiency Wages

### 8.1 Shapiro-Stiglitz (1984, AER) Shirking Model

Firms cannot perfectly monitor workers. If wages equal the market-clearing level, workers have no cost of being fired (they can immediately find an equivalent job). To deter shirking, firms pay above market-clearing wages — the **efficiency wage**:

$$w^* = \bar{w} + \frac{e}{q} \cdot (r + s + q)$$

where $e$ is effort cost, $q$ is the detection probability, $r$ is the discount rate, $s$ is the separation rate, and $\bar{w}$ is the outside option. The no-shirking condition requires involuntary unemployment as a discipline device. In equilibrium, all firms pay above market-clearing, creating persistent unemployment — a non-Walrasian result from asymmetric information.

---

## 9. State-Level Labor Market Variation

US states vary enormously in labor market outcomes:
- **Unemployment**: ranges from ~2% (Utah, New Hampshire) to ~6%+ (Nevada, Mississippi) in normal times
- **Union density**: 23% in New York vs 2.5% in South Carolina
- **Right-to-work laws**: 27 states prohibit union security agreements; associated with 3-5% lower wages (Eren-Ozbeklik 2016) but contested causality
- **Minimum wages**: from federal floor ($7.25) to $16+ (California, Washington); 30 states above federal
- **Occupational licensing**: 20-25% of workers need licenses (Kleiner-Krueger 2013); licensing requirements vary dramatically by state (e.g., barbers need 1,500 training hours in some states, 300 in others)
- **At-will employment**: most states are at-will; some recognize implied contract, public policy, or good faith exceptions

These institutional differences create natural experiments for studying labor market policies — the "laboratory of democracy."

---

## Key References

- Diamond, P. (1982). Aggregate Demand Management in Search Equilibrium. *JPE*, 90(5), 881-894.
- Mortensen, D., Pissarides, C. (1994). Job Creation and Job Destruction in the Theory of Unemployment. *RES*, 61(3), 397-415.
- Becker, G. (1964). *Human Capital*. Columbia University Press.
- Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
- Card, D., Krueger, A. (1994). Minimum Wages and Employment. *AER*, 84(4), 772-793.
- Autor, D., Levy, F., Murnane, R. (2003). The Skill Content of Recent Technological Change. *QJE*, 118(4), 1279-1333.
- Acemoglu, D., Restrepo, P. (2020). Robots and Jobs. *JPE*, 128(6), 2188-2244.
- Goldin, C., Katz, L. (2008). *The Race Between Education and Technology*. Harvard University Press.
- Borjas, G. (2003). The Labor Demand Curve Is Downward Sloping. *QJE*, 118(4), 1335-1374.
- Shapiro, C., Stiglitz, J. (1984). Equilibrium Unemployment as a Worker Discipline Device. *AER*, 74(3), 433-444.
