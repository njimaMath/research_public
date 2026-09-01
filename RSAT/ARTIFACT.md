# RSAT artifact guide

- Artifact: RSAT: Quantitative Strict de Almeida-Thouless Theorem and Overlap CLT
- Associated paper: “A quantitative replica-symmetric bound for Sherrington--Kirkpatrick model in the entire de Almeida--Thouless region”
- arXiv: [`2608.23413v2`](https://arxiv.org/abs/2608.23413v2)
- Proof assistant: Lean `v4.32.1`
- Library: Mathlib `v4.32.1`
- License: Apache-2.0
- Public endpoint: `Main.lean`

## Correspondence with the paper

| Paper statement | Lean declaration | File |
| --- | --- | --- |
| Smart-path Hamiltonian defining $H_{N,s}$ | `H_N_s` | `Main.lean` |
| Concrete-to-abstract smart-path identification | `H_N_s_eq_smartPath` | `Main.lean` |
| Theorem 1.1 | `ConcreteMain.strictAT_main` | `Main.lean` |
| Theorem 1.2 | `ConcreteMain.strictAT_overlapCLT_weak` | `Main.lean` |

The conclusion `StrictATClaim` of `strictAT_main` consists of
`quantitativeBounds` and `repliconSusceptibility`. The first field contains
both the overlap-concentration estimate and the free-energy correction with a
common compact-set constant, as stated in Theorem 1.1.

## Fresh build

```bash
git clone https://github.com/njimaMath/research_public.git
cd research_public/RSAT
lake update
lake build QuantitativeStrictAT
lake env lean Main.lean
```

Lake obtains all dependencies within the local RSAT build environment. No
sibling repository or directory is needed. `lake-manifest.json` records the
resolved revisions of Mathlib and its transitive dependencies.

## Verification

After dependency setup, run:

```bash
./verify.sh
git diff -- Main.lean
```

The script builds the `QuantitativeStrictAT` library, checks `Main.lean`, and
rejects project Lean sources containing `sorry`, `admit`, `sorryAx`, or a
project-local `axiom` declaration. For a clean rebuild, run:

```bash
lake clean
./verify.sh
```

`Main.lean` is the stable public interface of the artifact. It defines the
paper-facing objects and states the principal exported results while importing
the proof implementation from `Lemmas/`.
