import Main

example (β : ℝ) {h : ℝ} (hh : 0 < h) :
    ConcreteMain.ReplicaSymmetricFixedPointClaim β h :=
  ConcreteMain.replicaSymmetricFixedPointClaim_of_pos_field β hh

example : ConcreteMain.QuantitativeStrictATClaim :=
  ConcreteMain.quantitativeStrictATClaim

example (K : Set (ℝ × ℝ)) (hKcompact : IsCompact K)
    (hKsub : K ⊆ ConcreteMain.strictATRegion) :
    ConcreteMain.StrictATClaim K :=
  ConcreteMain.strictAT_main K hKcompact hKsub

example (β h : ℝ) : ConcreteMain.OverlapCLTClaim β h :=
  ConcreteMain.overlapCLTClaim β h

example (β h : ℝ) : ConcreteMain.OverlapCLTClaim β h :=
  ConcreteMain.strictAT_overlapCLT_weak β h
