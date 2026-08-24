# Latała Meets the Strict Almeida–Thouless Condition

Lean formalization of a quantitative strict Almeida–Thouless result for the Sherrington–Kirkpatrick model.

## Primary goal

The single non-negotiable goal of this project is:

> `Main.lean` must compile exactly as it is, using complete Lean proofs.

`Main.lean` is the fixed public endpoint of the formalization. Do not modify it to make the project compile.

The final development must not rely on proof holes or project-local assumptions inserted to bypass missing mathematics. In particular, do not use:

- `sorry`
- `admit`
- `sorryAx`
- new project-local `axiom` declarations used to replace proofs
- equivalent shortcuts whose purpose is to assume an unproved mathematical result

Reorganization, refactoring, file deletion, renaming, and cleanup are useful only insofar as they support this goal.

## Entry point

The public theorem is exposed through:

```text
Main.lean
```

It imports:

```text
Lemmas.MainResult
```

and packages the quantitative strict-AT conclusions proved by the library.

Treat `Main.lean` as read-only.

## Repository organization

The Lean package contains the main proof development under:

```text
Lemmas/
SpinGlass/
```

Important mathematical components currently include topics such as:

```text
Lemmas/ATDefs.lean
Lemmas/Cavity/
Lemmas/GTbound/
Lemmas/Price/
Lemmas/GTFlatness_cases/
Lemmas/smart_path/
Lemmas/MainResult.lean
```

The directory structure may be improved as the formalization matures. Prefer a professional Lean module layout:

- organize files by mathematical topic;
- use stable, descriptive module names;
- keep imports acyclic and as local as practical;
- split very large files when there is a genuine mathematical boundary;
- remove obsolete compatibility modules after all importers have migrated;
- remove scratch files, temporary checks, and tracked build artifacts;
- preserve useful local `AGENTS.md` instructions and mathematical blueprints.

Any move or rename must update all affected Lean imports.

## Proof policy

When a theorem is missing, prove it or reorganize the supporting theory so that it can be proved.

Do not weaken theorem statements merely to make Lean accept them.

Do not replace a difficult theorem by an assumption.

Do not introduce a stronger hypothesis into a public theorem unless the mathematics genuinely requires it and the change remains compatible with the fixed `Main.lean` endpoint.

Before deleting a lemma, check that it is not used through:

- direct imports or theorem references;
- namespace-qualified references;
- typeclass instances;
- attributes;
- simplification lemmas;
- notation or coercions;
- downstream public theorems.

If usefulness is uncertain, keep the declaration until the dependency is understood.

## Local instructions

Subdirectories may contain their own `AGENTS.md` files with more specific mathematical or implementation requirements.

Always follow the most specific applicable instructions while preserving the global objective:

```text
Main.lean must remain unchanged and compile without proof holes.
```

## Building

Run commands from the `latalaStrictAlmeidaThouless` package directory.

Build the library:

```bash
lake build LatalaMeetsAT
```

Check the fixed entry point explicitly:

```bash
lake env lean Main.lean
```

Both commands must succeed after structural changes.

## Final verification

Before considering a reorganization or proof task complete, verify all of the following:

```text
Main.lean is byte-for-byte unchanged.
lake build LatalaMeetsAT succeeds.
lake env lean Main.lean succeeds.
No sorry remains in the project proof development.
No admit remains.
No sorryAx is used as a proof shortcut.
No new project-local axiom has been introduced to replace a missing proof.
All renamed or moved modules have correct imports.
No required theorem was accidentally deleted.
Temporary and generated files are not part of the mathematical source tree.
```

A cleaner directory tree is desirable. A complete proof of the unchanged `Main.lean` is mandatory.
