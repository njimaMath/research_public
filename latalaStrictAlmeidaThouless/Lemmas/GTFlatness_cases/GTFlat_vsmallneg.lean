import Lemmas.GTFlatness

open MeasureTheory ProbabilityTheory Set

noncomputable section

namespace SpinGlass.AT

/-! ### Interior negative-overlap derivatives -/

/-- Extracts the numerical derivative from a supplied derivative formula for
the negative-overlap correlation. -/
lemma flatnessTildeGDeriv_eq_deriv
    (β h s v D : ℝ)
    (hD :
      HasDerivAt
        (fun u => flatnessTildeG β h (rsQ β h) s u)
        D v) :
    deriv
        (fun u => flatnessTildeG β h (rsQ β h) s u) v = D := by
  exact hD.deriv

end SpinGlass.AT
