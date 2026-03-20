# Public Choice & Political Economy

## Overview

Public choice applies economic methodology to political decision-making. This file covers social choice theory, voting, rent-seeking, regulatory capture, political business cycles, constitutional economics, and the political economy of reform.

---

## 1. Social Choice Theory

### 1.1 Arrow's Impossibility Theorem (1951, Nobel 1972)

Arrow proved that no social welfare function $F$ mapping individual preference orderings to a social ordering can simultaneously satisfy:

1. **Unrestricted domain**: Works for any set of individual preferences
2. **Pareto principle**: If all prefer $x$ to $y$, society prefers $x$ to $y$
3. **Independence of irrelevant alternatives (IIA)**: Social ranking of $x$ vs $y$ depends only on individual rankings of $x$ vs $y$
4. **Non-dictatorship**: No single individual determines the social ordering

With 3+ alternatives, these are mutually inconsistent. The theorem shows that democratic aggregation of diverse preferences is fundamentally problematic — there is no perfect voting rule.

### 1.2 Sen's Liberal Paradox (1970)

Amartya Sen (Nobel 1998) showed that even minimal liberalism (each individual is decisive over at least one pair of social alternatives — e.g., what they read privately) conflicts with the Pareto principle. The resolution requires restricting the domain of individual rights or weakening Pareto — illustrating deep tensions between individual liberty and collective welfare.

### 1.3 Gibbard-Satterthwaite Theorem (1973/1975)

With 3+ alternatives and unrestricted preferences, the only strategy-proof (dominant-strategy incentive-compatible) and onto social choice function is dictatorship. Every non-dictatorial voting rule is manipulable — some voter can benefit by misrepresenting their preferences. This justifies the study of incentive-compatible mechanisms with weaker solution concepts (Bayesian IC) or restricted domains.

---

## 2. Voting Theory

### 2.1 Median Voter Theorem

**Black (1948)**: With single-peaked preferences over a one-dimensional policy space, majority rule selects the preferred policy of the median voter. The median voter's ideal point defeats all alternatives in pairwise majority voting.

**Downs (1957, *An Economic Theory of Democracy*)**: Two office-seeking candidates converge to the median voter's ideal point in equilibrium (Hotelling-Downs model). If candidates are policy-motivated, convergence is incomplete. Multi-dimensional policy spaces break the theorem — no Condorcet winner generally exists (McKelvey 1976 chaos theorem: majority rule can cycle through any point in the policy space).

### 2.2 Condorcet Paradox and Cycling

With three voters and three alternatives, preferences can cycle: A beats B, B beats C, C beats A (by majority rule). There is no Condorcet winner. The probability of cycling increases with the number of alternatives and voters (but not monotonically). Structure-induced equilibrium (Shepsle 1979): institutional rules (committee structure, agenda control, amendment procedures) prevent cycling by restricting the set of feasible alternatives.

### 2.3 Alternative Voting Systems

- **Borda count**: Rank alternatives; assign points (n-1 for first, n-2 for second, ..., 0 for last). Violates IIA but satisfies other desirable properties.
- **Approval voting** (Brams-Fishburn 1978): Vote for any number of candidates; most approvals wins. Strategy-proof among sincere ballots; reduces wasted votes.
- **Ranked choice voting / Instant runoff**: Eliminate the candidate with fewest first-place votes; redistribute their votes. Used in Australia, Maine, Alaska, NYC. Non-monotonic (raising a candidate in your ranking can cause them to lose).
- **Quadratic voting** (Posner-Weyl 2017): Buy votes at quadratic cost ($1 for 1 vote, $4 for 2, $9 for 3...). Elicits intensity of preference. Used in Colorado legislature experimentally.

---

## 3. Rent-Seeking

### 3.1 Tullock (1967) and Krueger (1974)

Gordon Tullock ("The Welfare Costs of Tariffs, Monopolies, and Theft," 1967) observed that the traditional Harberger triangle understates the welfare cost of monopoly because firms spend resources to acquire and maintain monopoly power. The **Tullock rectangle** — the monopoly profit that firms compete to capture — is partially or fully dissipated through lobbying, litigation, and political contributions.

Anne Krueger ("The Political Economy of the Rent-Seeking Society," *AER*, 1974) estimated that rent-seeking in India (import licenses, quotas) dissipated 7.3% of GDP, and in Turkey 15% of GNP. The total social cost of a policy distortion = Harberger triangle + resources wasted in rent-seeking.

### 3.2 Tullock Contest

$n$ players compete for a prize $V$ by spending $x_i$ on lobbying. The probability of winning for player $i$:

$$p_i = \frac{x_i^r}{\sum_j x_j^r}$$

where $r$ is the returns-to-scale parameter. With $r = 1$ (lottery contest), total rent dissipation in symmetric equilibrium is $(n-1)/n$ of the prize — nearly full dissipation with many players. With $r > 1$, there is overdissipation (players spend more than the prize is worth in expectation), which can be interpreted as escalation or conflict.

### 3.3 Grossman-Helpman (1994): Protection for Sale

Grossman and Helpman ("Protection for Sale," *AER*, 1994) modeled trade protection as the outcome of lobbying by organized industry groups. The government maximizes a weighted sum of social welfare and lobbying contributions:

$$G = aW + \sum_i C_i$$

where $W$ is aggregate welfare, $C_i$ are contributions from lobby $i$, and $a$ is the weight on welfare. In equilibrium, the tariff on industry $i$:

$$\frac{t_i}{1 + t_i} = -\frac{I_i - \alpha_L}{\alpha + \alpha_L} \cdot \frac{z_i}{e_i}$$

where $I_i$ indicates whether industry $i$ is organized, $\alpha_L$ is the fraction of population in organized lobbies, $z_i$ is the output-to-imports ratio, and $e_i$ is the import demand elasticity. Industries that are organized, have high output relative to imports, and face inelastic import demand receive higher protection. Goldberg and Maggi (1999, *AER*) confirmed the model empirically for US trade policy.

---

## 4. Regulatory Capture

**Stigler (1971, *Bell Journal*)**: "The Theory of Economic Regulation" — regulation is acquired by the industry and designed for its benefit. Industries seek regulation to restrict entry, fix prices, and suppress competition. The regulatory agency becomes the instrument of the regulated industry.

**Peltzman (1976)**: Extended Stigler — regulators balance producer and consumer interests, with the equilibrium reflecting the political influence of each group. Regulation benefits producers when they are concentrated and consumers are diffuse (Olson's logic of collective action).

**Laffont-Tirole (1991, 1993)**: Formalized regulatory capture in a principal-agent framework. The regulator (agent) has private information about the firm's costs and may be "captured" (receive side payments for favorable regulation). The optimal regulatory contract trades off the costs of capture against the benefits of the regulator's information. Higher-powered incentive schemes (price caps) are more vulnerable to capture than lower-powered schemes (cost-plus). Institutional design (separation of powers, transparency requirements, revolving door restrictions) can mitigate capture.

---

## 5. Political Business Cycles

### 5.1 Nordhaus (1975): Opportunistic Cycles

Nordhaus ("The Political Business Cycle," *RES*, 1975): Incumbent politicians stimulate the economy before elections to win votes, then contract after elections to control inflation. The model requires adaptive expectations (voters are backward-looking). With rational expectations, the cycle disappears — voters see through the manipulation.

### 5.2 Rogoff-Sibert (1988): Rational Opportunistic Cycles

Rogoff and Sibert introduce competence signaling: more competent governments produce more output for given policy. Before elections, incumbents signal competence by providing more public goods (deficit spending). Voters rationally infer competence from observable fiscal behavior. The cycle persists under rational expectations because signaling is costly and partially revealing.

### 5.3 Partisan Models

**Hibbs (1977)**: Left-wing governments prioritize low unemployment; right-wing governments prioritize low inflation (reflecting constituent preferences of labor vs capital). Policy shifts at government changes create partisan business cycles.

**Alesina (1987)**: With rational expectations and two-period overlapping wage contracts, partisan differences produce real effects in the first half of a term (when the election outcome was uncertain) but not the second half (when contracts adjust). Empirical evidence: US GDP growth is ~1.5% higher in the first two years of Democratic vs Republican administrations (Blinder-Watson 2016), though the causes are debated.

---

## 6. Constitutional Economics

### 6.1 Buchanan and Tullock: The Calculus of Consent (1962)

James Buchanan (Nobel 1986) and Gordon Tullock asked: what decision rules would rational individuals choose from behind a veil of uncertainty about their future positions? The optimal voting rule trades off:

- **Decision-making costs**: unanimity is costly (holdouts); simple majority is cheap
- **External costs**: simple majority can impose large costs on minorities; unanimity protects everyone

The optimal rule minimizes the sum. For constitutional issues (high stakes), supermajority or unanimity rules are optimal. For routine legislation, simple majority suffices. This contractarian approach provides normative foundations for constitutional design.

---

## 7. Political Economy of Reform: Fernandez-Rodrik (1991)

Fernandez and Rodrik ("Resistance to Reform: Status Quo Bias in the Presence of Individual-Specific Uncertainty," *AER*, 1991): A reform that benefits the majority can be blocked because individuals don't know ex ante whether they will be winners or losers. Even if 60% would benefit, if each individual has only a 40% chance of being a winner (with the identity of winners uncertain ex ante), the majority votes against reform. Once the reform is implemented, the actual winners would vote to keep it. This produces **status quo bias**: reforms that would pass ex post fail ex ante due to individual-level uncertainty about who gains.

---

## 8. State-Level Political Economy

### 8.1 Direct Democracy

24 US states allow citizen-initiated ballot measures. California's Proposition 13 (1978) — capping property tax at 1% of assessed value — was the most consequential initiative in US fiscal history, dramatically reducing local government revenue and shifting school funding to the state level. Matsusaka (2004, *For the Many or the Few*): initiative states have lower taxes and spending, more closely matching median voter preferences than legislatively-dominated states.

### 8.2 Gerrymandering

Redistricting creates opportunities for partisan advantage. **Packing** (concentrating opponents in few districts) and **cracking** (spreading opponents across many districts) manipulate seat shares. The efficiency gap metric (Stephanopoulos-McGhee 2015) measures wasted votes. Independent redistricting commissions (California, Arizona, Michigan) reduce partisan gerrymandering. Quantitative tools (MCMC sampling of redistricting plans — DeFord, Duchin, Solomon 2021) provide baselines for detecting gerrymandered maps.

---

## Key References

- Arrow, K. J. (1951). *Social Choice and Individual Values*. Yale University Press.
- Downs, A. (1957). *An Economic Theory of Democracy*. Harper & Row.
- Tullock, G. (1967). The Welfare Costs of Tariffs, Monopolies, and Theft. *Western Economic Journal*, 5(3), 224-232.
- Krueger, A. (1974). The Political Economy of the Rent-Seeking Society. *AER*, 64(3), 291-303.
- Grossman, G., Helpman, E. (1994). Protection for Sale. *AER*, 84(4), 833-850.
- Stigler, G. (1971). The Theory of Economic Regulation. *Bell Journal*, 2(1), 3-21.
- Buchanan, J., Tullock, G. (1962). *The Calculus of Consent*. University of Michigan Press.
- Fernandez, R., Rodrik, D. (1991). Resistance to Reform. *AER*, 81(5), 1146-1155.
- Nordhaus, W. (1975). The Political Business Cycle. *RES*, 42(2), 169-190.
- Laffont, J.-J., Tirole, J. (1993). *A Theory of Incentives in Procurement and Regulation*. MIT Press.
