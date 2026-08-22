# Lemmas Agent Instructions

## Uniform theorem statements

When stating claims in this directory, formulate them uniformly over the
admissible model parameters. Prefer hypotheses of the following form:

```lean
{K : Set (ℝ × ℝ)} (data : UniformATData K) {β h s : ℝ}
  (hp : (β, h) ∈ K) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
  (∀ v ∈ Set.Ico (-1 : ℝ) 1, ...)
```

- Include `data`, `hp`, and `hs` in theorem assumptions whenever the result
  concerns the uniform strict-AT regime or an interpolation parameter.
- State overlap conclusions for every `v ∈ Set.Ico (-1 : ℝ) 1` by default;
  do not silently restrict to nonnegative overlaps.
- A narrower overlap range is allowed only when it is mathematically required
  by a branch-specific argument. State that restriction explicitly and provide
  a companion theorem covering the remaining admissible ranges when possible.
- Derive positivity, bounds on `rsQ β h`, and other local facts from
  `UniformATData` inside the proof rather than adding stronger public
  assumptions.