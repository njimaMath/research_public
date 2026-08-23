import Lemmas.GTbound.Endpoint

open MeasureTheory ProbabilityTheory Real BigOperators

noncomputable def fourOfTuple (p : ℝ × (ℝ × (ℝ × ℝ))) : Fin 4 → ℝ :=
  ![p.1, p.2.1, p.2.2.1, p.2.2.2]

lemma test_four_measure :
    Measure.pi (fun _ : Fin 4 => gaussianReal 0 1) =
      Measure.map fourOfTuple
        ((gaussianReal 0 1).prod ((gaussianReal 0 1).prod
          ((gaussianReal 0 1).prod (gaussianReal 0 1)))) := by
  apply Measure.pi_eq
  intro s hs
  have hm : Measurable fourOfTuple := by
    apply measurable_pi_lambda
    intro i
    fin_cases i <;> simp [fourOfTuple] <;> fun_prop
  rw [Measure.map_apply hm (MeasurableSet.univ_pi hs)]
  rw [show fourOfTuple ⁻¹' Set.univ.pi s =
      s 0 ×ˢ (s 1 ×ˢ (s 2 ×ˢ s 3)) by
    ext p
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const,
      Set.mem_prod]
    constructor
    · intro hp
      exact ⟨by simpa [fourOfTuple] using hp 0,
        by simpa [fourOfTuple] using hp 1,
        by simpa [fourOfTuple] using hp 2,
        by simpa [fourOfTuple] using hp 3⟩
    · rintro ⟨h0, h1, h2, h3⟩ i
      fin_cases i <;> simp [fourOfTuple, h0, h1, h2, h3]]
  simp [Measure.prod_prod, Fin.prod_univ_four]
  ring

lemma test_four_integral (f : (Fin 4 → ℝ) → ℝ)
    (hf : Integrable f (Measure.pi (fun _ : Fin 4 => gaussianReal 0 1))) :
    (∫ z : Fin 4 → ℝ, f z ∂Measure.pi (fun _ : Fin 4 => gaussianReal 0 1)) =
      ∫ z₀, ∫ z₁, ∫ z₂, ∫ z₃,
        f ![z₀, z₁, z₂, z₃] ∂gaussianReal 0 1
        ∂gaussianReal 0 1 ∂gaussianReal 0 1 ∂gaussianReal 0 1 := by
  let μ := (gaussianReal 0 1).prod ((gaussianReal 0 1).prod
    ((gaussianReal 0 1).prod (gaussianReal 0 1)))
  have hm : Measurable fourOfTuple := by
    apply measurable_pi_lambda
    intro i
    fin_cases i <;> simp [fourOfTuple] <;> fun_prop
  have hfm : Integrable f (Measure.map fourOfTuple μ) := by
    rw [← test_four_measure]
    exact hf
  have hc : Integrable (f ∘ fourOfTuple) μ := hfm.comp_measurable hm
  have hc' : Integrable (fun p => f (fourOfTuple p)) μ := by
    simpa [Function.comp_def] using hc
  rw [test_four_measure]
  rw [integral_map hm.aemeasurable hfm.aestronglyMeasurable]
  rw [integral_prod _ hc']
  apply integral_congr_ae
  filter_upwards [hc'.prod_right_ae] with z₀ hz₀
  rw [integral_prod _ hz₀]
  apply integral_congr_ae
  filter_upwards [hz₀.prod_right_ae] with z₁ hz₁
  rw [integral_prod _ hz₁]
  rfl
