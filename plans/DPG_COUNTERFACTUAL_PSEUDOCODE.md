# Counterfactual Generation Using DPG: Pseudocode

This document presents the formal pseudocode for the counterfactual generation approach using Data Point Generator (DPG) constraints. The approach leverages class-discriminative constraints learned by DPG to guide the generation of valid, actionable, and diverse counterfactual explanations.

---

## Algorithm 1: Main Counterfactual Generation

```
Algorithm: GenerateCounterfactuals
Input:
    x         : original sample (feature vector)
    f         : trained classifier
    y_target  : target class
    C         : DPG constraints for all classes
    A         : actionability constraints (optional)
    k         : number of counterfactuals to generate
    m         : overgeneration factor

Output:
    CF_set    : set of k diverse counterfactual explanations

1:  y_original ← f.predict(x)
2:  if y_original = y_target then
3:      return ERROR("Target must differ from predicted class")
4:  end if

5:  // Step 1: Analyze class boundaries for escape directions
6:  B ← AnalyzeBoundaryOverlap(C, y_original, y_target)

7:  // Step 2: Generate base counterfactual using DPG constraints
8:  x_base ← GenerateValidSample(x, C[y_target], B, A)

9:  // Step 3: Initialize population with escape-aware perturbations
10: population_size ← k × m
11: P ← InitializePopulation(x_base, x, population_size, C[y_target], B, A)

12: // Step 4: Evaluate fitness for all candidates
13: for each individual x' in P do
14:     x'.fitness ← CalculateFitness(x', x, y_target, y_original, P, C, A)
15: end for

16: // Step 5: Select best diverse counterfactuals
17: CF_set ← SelectDiverseCounterfactuals(P, x, y_target, k, m)

18: return CF_set
```

---

## Algorithm 2: Boundary Overlap Analysis

```
Algorithm: AnalyzeBoundaryOverlap
Input:
    C           : DPG constraints for all classes
    y_original  : original class
    y_target    : target class

Output:
    B           : boundary analysis containing escape directions

1:  B.non_overlapping ← ∅
2:  B.overlapping ← ∅
3:  B.escape_direction ← {}

4:  C_orig ← C[y_original]    // constraints for original class
5:  C_target ← C[y_target]    // constraints for target class

6:  for each constraint c_t in C_target do
7:      feature ← c_t.feature
8:      target_min ← c_t.min
9:      target_max ← c_t.max
10:     
11:     c_o ← FindMatchingConstraint(C_orig, feature)
12:     
13:     if c_o exists then
14:         orig_min ← c_o.min
15:         orig_max ← c_o.max
16:         
17:         // Determine if boundaries overlap
18:         if target_max ≤ orig_min then
19:             // Target requires lower values than original minimum
20:             B.non_overlapping ← B.non_overlapping ∪ {feature}
21:             B.escape_direction[feature] ← "decrease"
22:         else if target_min ≥ orig_max then
23:             // Target requires higher values than original maximum
24:             B.non_overlapping ← B.non_overlapping ∪ {feature}
25:             B.escape_direction[feature] ← "increase"
26:         else
27:             // Boundaries overlap
28:             B.overlapping ← B.overlapping ∪ {feature}
29:             B.escape_direction[feature] ← InferEscapeDirection(c_o, c_t)
30:         end if
31:     else
32:         B.overlapping ← B.overlapping ∪ {feature}
33:         B.escape_direction[feature] ← "both"
34:     end if
35: end for

36: return B
```

---

## Algorithm 3: Valid Sample Generation

```
Algorithm: GenerateValidSample
Input:
    x         : original sample
    C_target  : DPG constraints for target class
    B         : boundary analysis
    A         : actionability constraints

Output:
    x'        : valid sample within target class constraints

1:  x' ← copy(x)

2:  for each feature f in x do
3:      v_orig ← x[f]
4:      c ← FindMatchingConstraint(C_target, f)
5:      
6:      if c exists then
7:          v_min ← c.min if c.min ≠ null else -∞
8:          v_max ← c.max if c.max ≠ null else +∞
9:      else
10:         v_min ← -∞
11:         v_max ← +∞
12:     end if
13:     
14:     // Apply actionability constraints
15:     if A[f] = "no_change" then
16:         x'[f] ← v_orig
17:         continue
18:     else if A[f] = "non_decreasing" then
19:         v_min ← max(v_min, v_orig)
20:     else if A[f] = "non_increasing" then
21:         v_max ← min(v_max, v_orig)
22:     end if
23:     
24:     // Determine target value based on escape direction
25:     escape ← B.escape_direction[f]
26:     ε ← small_constant
27:     
28:     if escape = "increase" then
29:         v_target ← v_min + ε    // Just inside target's lower bound
30:     else if escape = "decrease" then
31:         v_target ← v_max - ε    // Just inside target's upper bound
32:     else
33:         if v_min ≤ v_orig ≤ v_max then
34:             v_target ← v_orig   // Keep original if within bounds
35:         else
36:             v_target ← (v_min + v_max) / 2
37:         end if
38:     end if
39:     
40:     x'[f] ← clip(v_target, v_min, v_max)
41: end for

42: // Validate prediction; if invalid, apply search refinement
43: if f.predict(x') ≠ y_target then
44:     x' ← ProgressiveDepthSearch(x, x', C_target, B, y_target)
45: end if

46: return x'
```

---

## Algorithm 4: Population Initialization

```
Algorithm: InitializePopulation
Input:
    x_base    : base counterfactual (constraint-adjusted)
    x         : original sample
    n         : population size
    C_target  : DPG constraints for target class
    B         : boundary analysis
    A         : actionability constraints

Output:
    P         : population of n candidate counterfactuals

1:  P ← {x_base}

2:  for i ← 1 to n - 1 do
3:      // Perturbation rate increases with individual index
4:      α ← 0.4 × (i / (n - 1))
5:      x' ← copy(x_base)
6:      
7:      for each feature f in x' do
8:          escape ← B.escape_direction[f]
9:          
10:         // Generate escape-aware perturbation
11:         if escape = "increase" then
12:             δ ← uniform(0, α)
13:         else if escape = "decrease" then
14:             δ ← uniform(-α, 0)
15:         else
16:             δ ← uniform(-α/2, α/2)
17:         end if
18:         
19:         x'[f] ← x_base[f] + δ
20:         
21:         // Clip to target constraint boundaries
22:         c ← FindMatchingConstraint(C_target, f)
23:         if c exists then
24:             x'[f] ← clip(x'[f], c.min, c.max)
25:         else
26:             x'[f] ← x[f]    // Reset unconstrained features
27:         end if
28:         
29:         // Enforce actionability (overrides constraints)
30:         if A[f] = "no_change" then
31:             x'[f] ← x[f]
32:         else if A[f] = "non_decreasing" then
33:             x'[f] ← max(x'[f], x[f])
34:         else if A[f] = "non_increasing" then
35:             x'[f] ← min(x'[f], x[f])
36:         end if
37:     end for
38:     
39:     P ← P ∪ {x'}
40: end for

41: return P
```

---

## Algorithm 5: Fitness Calculation

```
Algorithm: CalculateFitness
Input:
    x'          : candidate counterfactual
    x           : original sample
    y_target    : target class
    y_original  : original class
    P           : current population
    C           : DPG constraints
    A           : actionability constraints

Output:
    fitness     : fitness score (lower is better)

1:  // Check actionability - return invalid if violated
2:  if not IsActionable(x', x, A) then
3:      return INVALID_FITNESS
4:  end if

5:  // Check if identical to original
6:  if x' = x then
7:      return INVALID_FITNESS
8:  end if

9:  // Calculate core components
10: d ← EuclideanDistance(x, x')                    // Proximity
11: s ← CountChangedFeatures(x, x') / |features|   // Sparsity
12: p_c ← ConstraintViolationPenalty(x', C[y_target])
13: p_u ← UnconstrainedChangePenalty(x', x, C[y_target])

14: // Class prediction penalty (soft)
15: proba ← f.predict_proba(x')
16: p_target ← proba[y_target]
17: p_class ← β₁ × (1 - p_target)²
18: if f.predict(x') ≠ y_target then
19:     p_class ← p_class + β₂    // Hard penalty boost
20: end if

21: // Boundary escape penalty (dual-boundary)
22: p_escape ← CalculateEscapePenalty(x', x, C[y_original], y_target)

23: // Base fitness
24: fitness ← w_d × d + w_s × s + w_c × p_c + p_u + p_class + w_e × p_escape

25: // Population-based bonuses (if population available)
26: if |P| > 1 then
27:     div ← AverageDiversityToPopulation(x', P)
28:     rep ← MinDistanceToNeighbor(x', P)
29:     bnd ← BoundaryProximityBonus(x', y_target)
30:     
31:     // Cap bonuses to prevent unbounded negative fitness
32:     total_bonus ← min(w_div × div + w_rep × rep + bnd, MAX_BONUS)
33:     fitness ← fitness - total_bonus
34:     
35:     // Fitness sharing to maintain diversity
36:     niche_count ← CalculateNicheCount(x', P, σ_share)
37:     fitness ← fitness × √niche_count
38: end if

39: // Penalty multiplier for constraint violations
40: if p_c > 0 then
41:     fitness ← fitness × VIOLATION_MULTIPLIER
42: end if

43: return fitness
```

---

## Algorithm 6: Diverse Counterfactual Selection

```
Algorithm: SelectDiverseCounterfactuals
Input:
    P         : evaluated population
    x         : original sample
    y_target  : target class
    k         : number of counterfactuals to return
    m         : overgeneration factor

Output:
    CF_set    : k diverse valid counterfactuals

1:  // Filter to valid candidates only
2:  candidates ← ∅
3:  for each x' in P do
4:      if x'.fitness < INVALID_FITNESS then
5:          if f.predict(x') = y_target then
6:              margin ← CalculateProbabilityMargin(x', y_target)
7:              if margin ≥ MIN_MARGIN then
8:                  candidates ← candidates ∪ {x'}
9:              end if
10:         end if
11:     end if
12: end for

13: if candidates = ∅ then
14:     return null
15: end if

16: // Sort by fitness (best quality first)
17: candidates ← SortByFitness(candidates)
18: quality_pool ← candidates[1 : k × m]    // Top candidates

19: // Greedy diverse selection with MMR-style scoring
20: CF_set ← ∅
21: selected_arrays ← ∅
22: λ ← 0.6    // Diversity-proximity trade-off

23: // Always include best fitness candidate first
24: CF_set ← CF_set ∪ {quality_pool[1]}
25: selected_arrays ← selected_arrays ∪ {quality_pool[1]}
26: quality_pool ← quality_pool \ {quality_pool[1]}

27: while |CF_set| < k and quality_pool ≠ ∅ do
28:     best_score ← -∞
29:     best_candidate ← null
30:     
31:     for each x' in quality_pool do
32:         // Skip near-duplicates
33:         if IsNearDuplicate(x', selected_arrays) then
34:             continue
35:         end if
36:         
37:         // Calculate diversity: min distance to selected CFs
38:         min_div ← min{Distance(x', s) : s ∈ selected_arrays}
39:         norm_div ← min_div / max_diversity
40:         
41:         // Normalize fitness
42:         norm_fit ← (x'.fitness - fitness_min) / fitness_range
43:         
44:         // MMR-style score: balance diversity and quality
45:         score ← λ × norm_div - (1 - λ) × norm_fit
46:         
47:         if score > best_score then
48:             best_score ← score
49:             best_candidate ← x'
50:         end if
51:     end for
52:     
53:     if best_candidate ≠ null then
54:         CF_set ← CF_set ∪ {best_candidate}
55:         selected_arrays ← selected_arrays ∪ {best_candidate}
56:         quality_pool ← quality_pool \ {best_candidate}
57:     else
58:         break
59:     end if
60: end while

61: return CF_set
```

---

## Algorithm 7: Progressive Depth Search (Refinement)

```
Algorithm: ProgressiveDepthSearch
Input:
    x         : original sample
    x_init    : initial counterfactual (may not predict target)
    C_target  : target class constraints
    B         : boundary analysis
    y_target  : target class

Output:
    x'        : valid counterfactual predicting target class

1:  // Collect searchable features with valid bounds
2:  features ← GetSearchableFeatures(x, C_target, B)

3:  // Binary search on "depth" into target constraint space
4:  low ← 0.0       // Minimal change
5:  high ← 1.0      // Maximal change into target bounds
6:  best_valid ← null
7:  best_depth ← +∞

8:  for iter ← 1 to MAX_ITERATIONS do
9:      if |high - low| < ε then
10:         break
11:     end if
12:     
13:     mid ← (low + high) / 2
14:     x_test ← copy(x)
15:     
16:     for each (f, start, end) in features do
17:         // Interpolate between minimal and maximal change
18:         x_test[f] ← start + mid × (end - start)
19:     end for
20:     
21:     if f.predict(x_test) = y_target then
22:         if mid < best_depth then
23:             best_valid ← copy(x_test)
24:             best_depth ← mid
25:         end if
26:         high ← mid    // Search for even lower depth
27:     else
28:         low ← mid     // Need deeper into target bounds
29:     end if
30: end for

31: if best_valid ≠ null then
32:     return best_valid
33: else
34:     return RandomSampleSearch(x, C_target, y_target)
35: end if
```

---

## Key Notation Summary

| Symbol | Description |
|--------|-------------|
| $x$ | Original input sample (feature vector) |
| $x'$ | Candidate counterfactual |
| $f$ | Trained classifier |
| $y_{target}$ | Target (desired) class |
| $y_{original}$ | Original predicted class |
| $C$ | DPG constraints (per class) |
| $A$ | Actionability constraints |
| $B$ | Boundary analysis results |
| $P$ | Population of candidate counterfactuals |
| $k$ | Number of counterfactuals to generate |
| $m$ | Overgeneration factor |
| $w_d, w_s, w_c, w_e$ | Fitness component weights |
| $\lambda$ | Diversity-proximity trade-off parameter |

---

## Fitness Components

The fitness function combines multiple objectives (all minimized):

1. **Distance** ($d$): Euclidean distance from original sample
2. **Sparsity** ($s$): Ratio of changed features
3. **Constraint Penalty** ($p_c$): Violation of target class DPG constraints
4. **Unconstrained Penalty** ($p_u$): Changes to features without target constraints
5. **Class Penalty** ($p_{class}$): Soft penalty based on target class probability
6. **Escape Penalty** ($p_{escape}$): Penalty for remaining within original class bounds

Population-based components (bonuses, subtracted from fitness):
- **Diversity Bonus**: Encourages spread across feature space
- **Repulsion Bonus**: Encourages distance from nearest neighbors
- **Boundary Bonus**: Rewards proximity to decision boundary

---

## Complexity Analysis

| Phase | Time Complexity |
|-------|-----------------|
| Boundary Analysis | $O(|C| \times |F|)$ |
| Population Initialization | $O(n \times |F|)$ |
| Fitness Evaluation | $O(n \times |F| \times n)$ |
| Diverse Selection | $O(k \times m \times k)$ |
| **Total** | $O(n^2 \times |F|)$ |

Where:
- $n = k \times m$ (population size)
- $|F|$ = number of features
- $|C|$ = number of constraints
