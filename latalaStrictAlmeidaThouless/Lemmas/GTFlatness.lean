import Lemmas.GTFlatnessCore
import Lemmas.GTFlatness_cases.GTFlatvsmallpos
import Lemmas.GTFlatness_cases.GTFlat_vlargepos
import Lemmas.GTFlatness_cases.GTFlat_vsmallneg
import Lemmas.GTFlatness_cases.GTFlatvlargeneg

/-!
# GT flatness

This public module collects the shared GT-flatness theory and the four
overlap-specific consequences. Case modules depend only on
`GTFlatnessCore`, avoiding an import cycle while preserving the original
single-import API.
-/
