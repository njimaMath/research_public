# Agent instructions for `latalaStrictAlmeidaThouless`

## Single objective

There is one overriding goal for this package:

```text
Main.lean must compile successfully, with its theorem proved from genuine Lean proofs,
without `sorry`, `admit`, `sorryAx`, project-local `axiom` declarations, or other proof placeholders.
```

`Main.lean` is the fixed endpoint of the project.

Do not change `Main.lean`.

The contents of `Main.lean`, including its imports, theorem statements, definitions, comments, and formatting, are immutable for this task. All necessary work must be done in the files that `Main.lean` depends on.

Repository cleanup and reorganization are secondary. They are useful only insofar as they make the proof development professional, understandable, and maintainable without endangering the main objective.

If any cleanup, rename, deletion, or refactor conflicts with proving the unchanged `Main.lean`, preserve the proof path and abandon that cleanup.

## Definition of success

The task is complete only when all of the following hold from the package root:

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
```

and both commands succeed.

In addition:

```bash
git diff -- Main.lean
```

must produce no diff.

Search the project source for proof holes and project-local axioms:

```bash
rg '\b(sorry|admit)\b|sorryAx|^[[:space:]]*axiom\b' . \
  --glob '*.lean' \
  --glob '!.lake/**'
```

Every occurrence relevant to the dependency closure of `Main.lean` must be eliminated by a real proof. Prefer eliminating such occurrences throughout this package when practical.

Do not hide a proof hole behind another declaration, wrapper theorem, typeclass, opaque helper, generated file, or renamed axiom.

## `Main.lean` is immutable

Never edit `Main.lean`.

Do not:

- weaken or strengthen its theorem statement;
- change its imports;
- add assumptions;
- replace its theorem by a wrapper around an axiom;
- add local instances or declarations inside `Main.lean`;
- change comments or formatting there as part of repository cleanup.

Before finishing, verify that `git diff -- Main.lean` is empty.

If `Main.lean` does not compile, trace the failure into its imported modules and repair those modules instead.

## Proof integrity

All mathematical gaps in the dependency closure of `Main.lean` must be closed by genuine Lean proofs.

Forbidden techniques include:

```lean
sorry
admit
by exact (by sorry)
axiom someMissingTheorem : P
```

and any equivalent construction whose purpose is to assert the missing mathematics rather than prove it.

Do not introduce project-local axioms, postulates, or unsound escape hatches.

Do not use `unsafe` merely to bypass proof obligations.

Do not replace a difficult theorem with a stronger assumption supplied through a new structure field, typeclass, or parameter if that assumption was not already mathematically part of the project.

Do not weaken a theorem, definition, predicate, or final hypothesis merely to make compilation succeed.

Reusing established Mathlib theorems and already-proved repository theorems is encouraged. Lean's trusted logical foundations and ordinary Mathlib dependencies are not the target of this prohibition; the prohibition is against adding or relying on project-local unproved assumptions as substitutes for the missing proofs.

## Work from the final theorem backwards

Start by reading, without modifying:

```text
Main.lean
Lemmas/MainResult.lean
Lemmas/AGENTS.md
```

Then follow the import graph and theorem dependencies backwards from the theorem used by `Main.lean`.

Determine exactly which declarations are required to prove the final theorem. Prioritize those declarations over unrelated cleanup.

A useful dependency direction is approximately:

```text
Main.lean
    ↑
Lemmas/MainResult.lean
    ↑
strict-AT quantitative estimates
    ↑
Guerra-Talagrand / cavity / concentration estimates
    ↑
interpolation, Gaussian, fixed-point, and model infrastructure
```

The actual Lean import and declaration graph is authoritative.

When a nested `AGENTS.md` gives proof-specific instructions for a dependency required by `Main.lean`, follow those instructions unless they conflict with this root file. This root objective has priority.

## Preserve mathematical statements

The goal is to prove the existing project, not to redefine success.

Unless a statement is a purely internal helper and changing it is mathematically harmless, preserve existing public theorem statements and definitions used by the final proof.

In particular:

- do not weaken conclusions;
- do not strengthen assumptions to avoid a proof;
- do not silently narrow parameter ranges;
- do not replace quantitative bounds by qualitative ones;
- do not replace a theorem by a declaration that assumes essentially the theorem itself.

Intermediate helper lemmas may be strengthened, split, merged, renamed, moved, or replaced when that gives a cleaner genuine proof of the required downstream theorem.

## Reorganization of `Lemmas/`

A professional Lean source tree is desirable, but it is not the success criterion. Compilation and proof integrity of the unchanged `Main.lean` are the success criterion.

You may reorganize `Lemmas/` when it improves the proof architecture.

Prefer:

- topic-based directories;
- conventional Lean `UpperCamelCase` module names;
- small modules with clear mathematical responsibilities;
- direct imports instead of accidental transitive imports;
- a clear acyclic dependency graph;
- public barrel modules only where they provide a useful stable import;
- splitting very large files along real mathematical boundaries.

Potential areas for cleanup include the current AT/fixed-point, concentration, cavity, Guerra-Talagrand, Price, and smart-path developments.

Do not perform broad renames or moves merely for aesthetics while essential proof gaps remain.

When moving a Lean file, update every import and relevant documentation reference and immediately check that the affected modules still compile.

## Deleting files and lemmas

Deletion is allowed only when it cannot damage the proof of `Main.lean` or useful supporting infrastructure.

Before deleting a Lean file or declaration, check:

- whether it lies in the import/declaration dependency closure of `Main.lean`;
- direct and indirect references throughout the repository;
- `simp`, `instance`, `aesop`, coercion, notation, and other attributes;
- nested `AGENTS.md`, README, blueprint, and `.tex` references;
- whether it is a compatibility module still imported anywhere;
- whether its removal changes which theorem or instance Lean finds;
- whether the relevant modules and final package still build after deletion.

If uncertain, keep the declaration. Proof completion is more important than aggressive pruning.

Compatibility-import shims may be removed after all importers have been migrated to the canonical module and the full build succeeds.

## Temporary and generated files

Clean obvious generated or scratch artifacts when safe. Examples currently worth inspecting include:

```text
.lake/
.tmp_check.lean
.tmp_four.lean
.tmp_names2.lean
ScratchCavityCheck.lean
ScratchFinCheck.lean
```

Before deleting a scratch Lean file, inspect it for a proof fragment or useful theorem that has not yet been migrated into the real source tree.

If it contains useful mathematics needed for the final proof, move that mathematics into an appropriately named source module first.

Tracked build output such as `.lake/` should not be treated as source.

Do not delete reproducibility files such as:

```text
lakefile.lean
lake-manifest.json
lean-toolchain
```

Do not delete mathematical documentation such as `Lemmas/Cavity/blueprint.tex` merely because Lean does not compile it.

## Git safety

At the beginning, run:

```bash
git status --short
```

Preserve all unrelated user work and uncommitted changes.

Never use destructive commands such as:

```text
git reset --hard
git clean -fd
forced checkout of user changes
```

Use `git mv` for intentional file renames when practical.

Do not modify `Main.lean`, even temporarily.

## Build discipline

Do not wait until the end to discover that the reorganization broke the project.

After each coherent proof or structural change, compile the smallest affected module when practical.

Periodically run:

```bash
lake build LatalaMeetsAT
```

The final mandatory checks are:

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
git diff -- Main.lean
rg '\b(sorry|admit)\b|sorryAx|^[[:space:]]*axiom\b' . \
  --glob '*.lean' \
  --glob '!.lake/**'
```

A successful build obtained by inserting an axiom or proof placeholder is a failure.

A beautifully reorganized repository in which `Main.lean` still fails is also a failure.

An unchanged `Main.lean` that compiles because all of its mathematical dependencies have genuine proofs is success.

## Final report

When finished, report:

- confirmation that `Main.lean` was not changed;
- the result of `lake build LatalaMeetsAT`;
- the result of `lake env lean Main.lean`;
- all `sorry`, `admit`, `sorryAx`, or project-local `axiom` occurrences removed from the relevant proof dependency graph;
- the main mathematical proof gaps that were closed;
- structural reorganization performed in support of those proofs;
- files or declarations deleted and why their deletion was safe;
- any cleanup intentionally skipped because it could have endangered the proof objective.

Do not claim completion unless the unchanged `Main.lean` actually compiles.
