import Lemmas.GT.Semigroup

set_option autoImplicit false

namespace SpinGlass.AT

/-- The sole half-mass identity required by the specialized GT recursion. -/
theorem poissonDirichlet_half_identity (x : ℝ) : Thalf (fun _ => x) 0 = x := by
  simp [Thalf, standardGaussianExpectation]
  ring

end SpinGlass.AT
