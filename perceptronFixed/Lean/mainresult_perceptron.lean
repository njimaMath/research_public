import Theorem1.Theorem

open scoped Topology
open MeasureTheory Filter

namespace MainResult

noncomputable section

/-! The notation exposed by the final result is shared with its proof modules. -/

abbrev γ : Measure ℝ := Theorem1.γ
abbrev Expect (f : ℝ → ℝ) : ℝ := Theorem1.Expect f
abbrev φ : ℝ → ℝ := Theorem1.φ
abbrev Φbar : ℝ → ℝ := Theorem1.Φbar
abbrev E : ℝ → ℝ := Theorem1.E
abbrev U : ℝ → ℝ → ℝ → ℝ := Theorem1.U
abbrev F : ℝ → ℝ → ℝ → ℝ := Theorem1.F
abbrev P : ℝ → ℝ := Theorem1.P
abbrev B : ℝ → ℝ → ℝ := Theorem1.B
abbrev R : ℝ → ℝ → ℝ → ℝ := Theorem1.R
abbrev Cκ : ℝ → ℝ := Theorem1.Cκ
abbrev αc : ℝ → ℝ := Theorem1.αc
abbrev IsSolution (κ α q r : ℝ) : Prop := Theorem1.IsSolution κ α q r

/-! Existence below capacity and nonexistence at or above capacity. -/

theorem main
    (κ α : ℝ)
    (hκ : 0 ≤ κ) :
    (0 < α ∧ α < αc κ → ∃! qr : ℝ × ℝ, IsSolution κ α qr.1 qr.2) ∧
    (αc κ ≤ α → ¬ ∃ q r : ℝ, IsSolution κ α q r) := by
  constructor
  · rintro ⟨hα0, hα⟩
    exact Theorem1.theorem_main κ α hκ hα0 hα
  · intro hα
    exact Theorem1.theorem_main_no_solution κ α hκ hα

/-! Canonical solution and replica-symmetric functional below capacity. -/

abbrev sol
    (κ α : ℝ) (hκ : 0 ≤ κ) (hα0 : 0 < α) (hα : α < αc κ) : ℝ × ℝ :=
  Theorem1.sol κ α hκ hα0 hα

abbrev qSol
    (κ α : ℝ) (hκ : 0 ≤ κ) (hα0 : 0 < α) (hα : α < αc κ) : ℝ :=
  Theorem1.qSol κ α hκ hα0 hα

abbrev rSol
    (κ α : ℝ) (hκ : 0 ≤ κ) (hα0 : 0 < α) (hα : α < αc κ) : ℝ :=
  Theorem1.rSol κ α hκ hα0 hα

abbrev RSFunctional : ℝ → ℝ → ℝ → ℝ → ℝ := Theorem3.RSFunctional

abbrev RSStar
    (κ α : ℝ) (hκ : 0 ≤ κ) (hα0 : 0 < α) (hα : α < αc κ) : ℝ :=
  Theorem3.RSStar κ α hκ hα0 hα

def qAlpha (κ : ℝ) (hκ : 0 ≤ κ) (α : ℝ) : ℝ :=
  if hα : 0 < α ∧ α < αc κ then qSol κ α hκ hα.1 hα.2 else 0

def rAlpha (κ : ℝ) (hκ : 0 ≤ κ) (α : ℝ) : ℝ :=
  if hα : 0 < α ∧ α < αc κ then rSol κ α hκ hα.1 hα.2 else 0

def RSStarAlpha (κ : ℝ) (hκ : 0 ≤ κ) (α : ℝ) : ℝ :=
  if hα : 0 < α ∧ α < αc κ then RSStar κ α hκ hα.1 hα.2 else 0

private lemma qAlpha_eq_qSol
    (κ α : ℝ)
    (hκ : 0 ≤ κ)
    (hα0 : 0 < α)
    (hα : α < αc κ) :
    qAlpha κ hκ α = qSol κ α hκ hα0 hα := by
  simp [qAlpha, hα0, hα]

private lemma rAlpha_eq_rSol
    (κ α : ℝ)
    (hκ : 0 ≤ κ)
    (hα0 : 0 < α)
    (hα : α < αc κ) :
    rAlpha κ hκ α = rSol κ α hκ hα0 hα := by
  simp [rAlpha, hα0, hα]

private lemma RSStarAlpha_eq_RSStar
    (κ α : ℝ)
    (hκ : 0 ≤ κ)
    (hα0 : 0 < α)
    (hα : α < αc κ) :
    RSStarAlpha κ hκ α = RSStar κ α hκ hα0 hα := by
  simp [RSStarAlpha, hα0, hα]

/-! Sequential forms used to establish the left limits. -/

private theorem second_main_seq
    (κ : ℝ) (hκ : 0 ≤ κ)
    (α : ℕ → ℝ)
    (hα : ∀ n, 0 < α n ∧ α n < αc κ)
    (hlim : Tendsto α atTop (𝓝 (αc κ))) :
    Tendsto (fun n => rSol κ (α n) hκ (hα n).1 (hα n).2) atTop atTop ∧
      Tendsto (fun n => qSol κ (α n) hκ (hα n).1 (hα n).2) atTop (𝓝 (1 : ℝ)) := by
  exact Theorem1.theorem_second_main_seq κ hκ α hα hlim

private theorem third_main_seq
    (κ : ℝ) (hκ : 0 ≤ κ)
    (α : ℕ → ℝ)
    (hα : ∀ n, 0 < α n ∧ α n < αc κ)
    (hlim : Tendsto α atTop (𝓝 (αc κ))) :
    Tendsto (fun n => RSStar κ (α n) hκ (hα n).1 (hα n).2) atTop atBot := by
  exact Theorem3.theorem_three_seq (κ := κ) (hκ := hκ) (α := α) (hα := hα) hlim

private theorem exists_good_approx_seq
    (κ : ℝ)
    (α : ℕ → ℝ)
    (hlim : Tendsto α atTop (𝓝[<] (αc κ))) :
    ∃ α' : ℕ → ℝ,
      (∀ n, 0 < α' n ∧ α' n < αc κ) ∧
      Tendsto α' atTop (𝓝 (αc κ)) ∧
      α' =ᶠ[atTop] α := by
  let α' : ℕ → ℝ := fun n =>
    if hα : 0 < α n ∧ α n < αc κ then α n else αc κ / 2
  have hαc_pos : 0 < αc κ := Theorem1.αc_pos κ
  have hhalf : 0 < αc κ / 2 ∧ αc κ / 2 < αc κ := by
    constructor <;> linarith
  have hnhds : Tendsto α atTop (𝓝 (αc κ)) := (tendsto_nhdsWithin_iff.mp hlim).1
  have hlt : ∀ᶠ n in atTop, α n < αc κ := by
    simpa [Set.mem_Iio] using (tendsto_nhdsWithin_iff.mp hlim).2
  have hpos : ∀ᶠ n in atTop, 0 < α n := by
    have hIoi : Set.Ioi (αc κ / 2) ∈ 𝓝 (αc κ) := by
      refine IsOpen.mem_nhds isOpen_Ioi ?_
      exact hhalf.2
    have hmem : ∀ᶠ n in atTop, α n ∈ Set.Ioi (αc κ / 2) := hnhds.eventually hIoi
    refine hmem.mono ?_
    intro n hn
    exact lt_trans hhalf.1 hn
  have hgood : ∀ᶠ n in atTop, 0 < α n ∧ α n < αc κ := hpos.and hlt
  have hα'_eq : α' =ᶠ[atTop] α := by
    filter_upwards [hgood] with n hn
    simp [α', hn]
  have hα'_good : ∀ n, 0 < α' n ∧ α' n < αc κ := by
    intro n
    by_cases hα : 0 < α n ∧ α n < αc κ
    · simpa [α', hα] using hα
    · simp [α', hα, hhalf]
  have hα'_tendsto : Tendsto α' atTop (𝓝 (αc κ)) := hnhds.congr' hα'_eq.symm
  exact ⟨α', hα'_good, hα'_tendsto, hα'_eq⟩

private theorem tendsto_total_of_tendsto_seq
    {β : Type*}
    {l : Filter β}
    (κ : ℝ)
    (f : ∀ α : ℝ, 0 < α → α < αc κ → β)
    (fTotal : ℝ → β)
    (fTotal_eq : ∀ α hα0 hα, fTotal α = f α hα0 hα)
    (hseq :
      ∀ (α : ℕ → ℝ) (hα : ∀ n, 0 < α n ∧ α n < αc κ),
        Tendsto α atTop (𝓝 (αc κ)) →
          Tendsto (fun n => f (α n) (hα n).1 (hα n).2) atTop l) :
    Tendsto fTotal (𝓝[<] (αc κ)) l := by
  refine Filter.tendsto_of_seq_tendsto ?_
  intro α hlim
  obtain ⟨α', hα', hα'lim, hα'eq⟩ := exists_good_approx_seq κ α hlim
  have htotal' : Tendsto (fun n => fTotal (α' n)) atTop l := by
    convert hseq α' hα' hα'lim using 1
    funext n
    exact fTotal_eq (α' n) (hα' n).1 (hα' n).2
  exact htotal'.congr' (hα'eq.fun_comp fTotal)

/-! Divergence at the storage capacity. -/

theorem second_main
    (κ : ℝ) (hκ : 0 ≤ κ) :
    Tendsto (qAlpha κ hκ) (𝓝[<] (αc κ)) (𝓝 (1 : ℝ)) ∧
      Tendsto (rAlpha κ hκ) (𝓝[<] (αc κ)) atTop := by
  constructor
  · exact tendsto_total_of_tendsto_seq κ
      (fun α hα0 hα => qSol κ α hκ hα0 hα)
      (qAlpha κ hκ)
      (fun α hα0 hα => qAlpha_eq_qSol κ α hκ hα0 hα)
      (fun α hα hlim => (second_main_seq κ hκ α hα hlim).2)
  · exact tendsto_total_of_tendsto_seq κ
      (fun α hα0 hα => rSol κ α hκ hα0 hα)
      (rAlpha κ hκ)
      (fun α hα0 hα => rAlpha_eq_rSol κ α hκ hα0 hα)
      (fun α hα hlim => (second_main_seq κ hκ α hα hlim).1)

theorem third_main
    (κ : ℝ) (hκ : 0 ≤ κ) :
    Tendsto (RSStarAlpha κ hκ) (𝓝[<] (αc κ)) atBot := by
  exact tendsto_total_of_tendsto_seq κ
    (fun α hα0 hα => RSStar κ α hκ hα0 hα)
    (RSStarAlpha κ hκ)
    (fun α hα0 hα => RSStarAlpha_eq_RSStar κ α hκ hα0 hα)
    (third_main_seq κ hκ)

end

end MainResult
