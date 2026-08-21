import Lemmas.GTFlatness

open MeasureTheory ProbabilityTheory Set

open SpinGlass.AT

open Lean Elab Command

run_meta do
  let env ← getEnv
  for (name, _) in env.constants.toList do
    if name.toString.toLower.contains "path" then
      logInfo m!"{name}"
