# AFM submission packet

This file records the remaining author actions for submitting the associated
paper to the Annals of Formalized Mathematics (AFM). It is not part of the
formal proof.

## Submission metadata

- Title: A quantitative replica-symmetric bound for Sherrington--Kirkpatrick
  model in the entire de Almeida--Thouless region
- Authors: Seiichiro Kusuoka and Shuta Nakajima
- Open-archive record: [arXiv:2608.23413v2](https://arxiv.org/abs/2608.23413v2)
- Formal artifact: [RSAT](https://github.com/njimaMath/research_public/tree/main/RSAT)
- Proof assistant: Lean 4.32.1
- Library: Mathlib 4.32.1
- Artifact license: Apache-2.0
- Suggested subjects: probability, mathematical physics, formalized
  mathematics, interactive theorem proving
- Suggested MSC 2020 codes: 60K35, 82B44, 68V20
- Suggested handling editor: Sébastien Gouëzel, subject to the authors'
  conflict-of-interest check

AFM is an overlay journal. The manuscript is already deposited on an accepted
open archive and the artifact is public and open source. The journal submission
itself must be made through the
[AFM submission page](https://afm.episciences.org/submit/index).

## Required manuscript change before submission

AFM asks that each main result link directly to its formal statement. The
formalization section in arXiv v2 names the declarations, but its displayed
repository reference does not provide direct code links. Replace the relevant
sentences by the following text, or add the links as footnotes:

```latex
Theorem~\ref{thm:main} corresponds to the Lean theorem
\href{https://github.com/njimaMath/research_public/blob/cf9ae297cb90c990611fe5ba4be14c2e79602ff4/RSAT/Main.lean#L332-L380}
{\texttt{Main.strictAT\_main}}. Its conclusion is represented by
\href{https://github.com/njimaMath/research_public/blob/cf9ae297cb90c990611fe5ba4be14c2e79602ff4/RSAT/Main.lean#L307-L325}
{\texttt{Main.StrictATClaim}}.

Theorem~\ref{thm:overlap-clt-intro} corresponds to
\href{https://github.com/njimaMath/research_public/blob/cf9ae297cb90c990611fe5ba4be14c2e79602ff4/RSAT/Main.lean#L521-L595}
{\texttt{Main.strictAT\_overlapCLT\_weak}}.
```

These URLs use the immutable source commit containing the public declarations,
so later edits to the default branch cannot change what the manuscript links
to. Confirm that `hyperref` is loaded and that all three links are clickable in
the deposited PDF.

The paper should also cite the artifact itself, using `CITATION.cff`, rather
than referring to it only in prose.

## Recommended manuscript strengthening

The current abstract reports the mathematical results but does not mention the
formalization. Add a sentence such as:

> We formalize the main results in Lean 4 using Mathlib, including a concrete
> Gaussian model and checked bridges between the paper's notation and a
> reusable proof backend.

Add `Lean 4`, `formalized mathematics`, and `interactive theorem proving` to
the keywords, and include MSC 2020 code `68V20`.

The present formalization section gives the theorem correspondence. For a
stronger fit with AFM's stated evaluation criteria, it should also discuss:

- why a countable product Gaussian space was chosen and how the finite disorder
  variables are recovered from it;
- which measurability and integration arguments required the most formal work;
- how the paper-facing `Main` namespace is separated from the reusable backend;
- what the formal proof revealed about hidden uniformity, positivity, and
  finite-size assumptions in the informal argument;
- the weak-convergence formulation chosen for Theorem 1.2 and the harmless
  shift from positive system sizes to natural-number indexing;
- the standard Lean axioms and the `native_decide` trust boundary documented in
  `ARTIFACT.md`;
- the artifact's current integration status with Mathlib and the earlier
  `SpinGlass` code acknowledged in `NOTICE`.

These additions need not reproduce substantial Lean code. Their purpose is to
explain the mathematical and engineering lessons of the formalization.

A software citation can be added to the bibliography in this form:

```bibtex
@software{KusuokaNakajimaRSAT2026,
  author  = {Seiichiro Kusuoka and Shuta Nakajima},
  title   = {RSAT: Quantitative Strict de Almeida--Thouless Theorem
             and Overlap CLT},
  year    = {2026},
  url     = {https://github.com/njimaMath/research_public/tree/main/RSAT},
  note    = {Lean 4 formalization associated with arXiv:2608.23413v2}
}
```

## Suggested cover note

> Dear Editors,
>
> We submit “A quantitative replica-symmetric bound for
> Sherrington--Kirkpatrick model in the entire de Almeida--Thouless region” for
> consideration in the Annals of Formalized Mathematics. The paper proves
> quantitative overlap concentration, a finite-size free-energy correction,
> the finite-volume replicon susceptibility, and an overlap central limit
> theorem throughout the strict de Almeida--Thouless region. Its main results
> are formalized in Lean 4.32.1 with Mathlib 4.32.1. The public Lean interface
> gives paper-facing definitions and direct counterparts of Theorems 1.1 and
> 1.2; the accompanying artifact contains the complete dependency graph and
> reproducible verification scripts. The manuscript is available as
> arXiv:2608.23413v2 and the artifact is openly available under Apache-2.0.
>
> The manuscript is original, has not been published previously, and is not
> under consideration by another journal. [Add the authors' funding and
> conflict-of-interest declarations here.]
>
> Sincerely,
> Seiichiro Kusuoka and Shuta Nakajima

## Release checklist

- Commit the current documentation and verification changes.
- Run `./verify.sh` on Linux or macOS, or `./verify.ps1` on Windows, from a
  fresh clone after `lake update`.
- Create a new annotated release tag for the AFM artifact. Do not reuse
  `rsat-artifact-v1`, which predates the current public theorem names.
- Push the commit and release tag, then confirm that every manuscript link and
  every link in `README.md` and `ARTIFACT.md` resolves on GitHub.
- Upload a new arXiv version containing the direct formal-statement links and
  an artifact citation. The PDF must not be wrapped in a zip archive when
  selected on Episciences.
- Confirm that both authors approve the submitted version and the suggested
  handling editor.
- Supply funding and conflict-of-interest declarations. Do not infer these
  declarations from the artifact.
- Confirm that the manuscript is not simultaneously submitted elsewhere.
- Request archival of the release from Software Heritage. Record the resulting
  SWHID in this file, `CITATION.cff`, and the final accepted manuscript.
- At acceptance, follow AFM's copyediting instructions, upload the AFM-formatted
  version to the same open-archive record, and use the required CC BY 4.0
  publication license.

## Software Heritage identifier

Pending. AFM invites authors to provide a Software Heritage persistent
identifier with the accepted version.

## Official venue pages

- [Instructions for authors](https://afm.episciences.org/page/instructions-for-authors)
- [Aims and scope](https://afm.episciences.org/page/aims-and-scope)
- [Publishing policies](https://afm.episciences.org/page/publishing-policies)
- [Editorial board](https://afm.episciences.org/page/comite-editorial)
