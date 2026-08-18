// Lean compiler output
// Module: SpinGlassAT.Mathlib.Probability.Distributions.GaussianIntegrationByParts
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Lie.OfAssociative public import Mathlib.Data.Real.StarOrdered public import Mathlib.Order.CompletePartialOrder public import Mathlib.Probability.Distributions.Gaussian.Real public import Mathlib.Topology.Algebra.Module.ModuleTopology public import Mathlib.Topology.EMetricSpace.Paracompact public import Mathlib.Topology.Separation.CompletelyRegular
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Lie_OfAssociative(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Real_StarOrdered(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompletePartialOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Probability_Distributions_Gaussian_Real(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_ModuleTopology(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_EMetricSpace_Paracompact(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Separation_CompletelyRegular(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LatalaMeetsAT_SpinGlassAT_Mathlib_Probability_Distributions_GaussianIntegrationByParts(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Lie_OfAssociative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Real_StarOrdered(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompletePartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Probability_Distributions_Gaussian_Real(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_ModuleTopology(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_EMetricSpace_Paracompact(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Separation_CompletelyRegular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
